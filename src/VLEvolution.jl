module VLEvolution

export evolve, VLGroup, AbstractMetaVariable, Point,randomPoint, similar_population,
AbstractOptimParameters, OptimParameters, Param, AbstractModel, fitness_factory, Model,
groupparams, best_by_size, KnotModel, random_population, model_function_factory, get_n_metavariables

using Dates
using StatsBase
using Statistics
using Polyester: @batch


include("structures.jl")
include("operators.jl")
include("nondominated.jl")


function hmss(dt)
    dt = dt.value
    (h,r) = divrem(dt,60*60*1000)
    (m,r) = divrem(r, 60*1000)
    (s,r) = divrem(r, 1000)
    string(Int(h),":",Int(m),":",s)
end

function best_by_size(pop::AbstractVector{<:AbstractModel},perf::AbstractVector{<:Number})
    sizes = get_n_metavariables.(pop)
    u = sort(unique(sizes))
    return Dict([(v,pop[findmax([ s == v ? p : -Inf for (s,p) in collect(zip(sizes,perf)) ])[2]]) for v in u])
end

function fitness_factory(functional::Function, x::AbstractVector,y::AbstractVector,weigths::AbstractVector)
    function fitness(state::OptimParameters,m::AbstractModel)
        residuals = y .- functional(x,m)
        if state.helper == :more
            [- residuals' * (weigths .* residuals) , sum(get_n_metavariables(m))]
        elseif state.helper == :less
            [- residuals' * (weigths .* residuals) , - sum(get_n_metavariables(m))]
        elseif state.helper == :none
            [- residuals' * (weigths .* residuals)]
        elseif state.helper == :both
            v = sum(get_n_metavariables(m))
            [- residuals' * (weigths .* residuals) , - v, v]
        else
            throw(error("Unexpected helper: $(state.helper)"))
        end
    end
    return fitness
end

fitness_factory(functional::Function, x::AbstractVector,y::AbstractVector) = fitness_factory(functional, x, y, ones(length(x)))



function evaluate(state, pop, fitness_function)
    pop_size = length(pop)
    perf = Vector{Vector{Float64}}(undef,pop_size)
    @batch for i in 1:pop_size
        perf[i] = fitness_function(state,pop[i])
    end
    return perf
end


@inline scheduler(gen::Integer,p::AbstractOptimParameters) = p # rel_change::Number,

function evolve(pop, fitness_function,state::AbstractOptimParameters; max_gen=nothing,max_time=nothing, info_every=50) # stopping_tol=nothing,
    @assert any([!isnothing(c) for c in [max_gen,max_time]]) "Please define at least one stopping criterium" #,stopping_tol

    start_time = now()
    gen = 0
    # no_change_counter = 0
    # previous_convergence_metric = Inf

    if !isnothing(info_every) @info "Starting evolution with state $state" end
    while true
        if (!isnothing(max_gen)) && (gen >= max_gen)
            if !isnothing(info_every) @info "Finished: Reached maximum generation $(max_gen)" end
            break
        end
        
        t = now() - start_time
        if (!isnothing(max_time)) && (t >= max_time)
            if !isnothing(info_every) @info "Finished: Reached maximum execution time $max_time" end
            break
        end

        gen +=  1

        @debug "Current state: $state"
        @debug "Evaluating parent population"
        pop_perf = evaluate(state,pop,fitness_function)
        constraint_violation = constraints(state, pop, pop_perf)
        rank, _ = fast_non_dominated_sort(pop_perf,constraint_violation)
        distance = crowding_distance(pop_perf,rank)

        @debug "Generating candidate child population"
        mating_pool = selection(rank,distance)
        pop_candidate = crossover(state,pop[mating_pool],state.pop_size)

        mutate!(state,pop_candidate)

        @debug "Evaluating candidate child population performance"
        pop_candidate_perf = evaluate(state,pop_candidate,fitness_function)

        @debug "Ranking joint pool - NSGA II + Main objective elitism"
        pool = vcat(pop,pop_candidate)
        pool_perf = vcat(pop_perf,pop_candidate_perf)
        constraint_violation = constraints(state, pool, pool_perf)
        _, fronts = fast_non_dominated_sort(pool_perf,constraint_violation)
        

        selected = Int[]
        # Main obj elitism
        if (state.n_main_obj_elitism != 0) & any(constraint_violation .== 0)
            append!(selected, sortperm(
                [(isnan(v[1]) || c != 0) ? -Inf : v[1] for (c,v) in zip(constraint_violation,pool_perf)],rev=true
                )[1:state.n_main_obj_elitism]
            )
            
        end
        # Determine non-dominated rank that complitely fits in pop_size
        ind = 0
        for (i,v) in enumerate(length.(fronts))
            if (length(selected) + v) > state.pop_size
                ind = i
                break
            end
            selected = unique(vcat(selected,fronts[i]))
        end

        # Append the remaining base on crowding_distance
        remaining = state.pop_size-length(selected)
        if remaining != 0
            distance = crowding_distance(pool_perf[fronts[ind]])
            append!(selected,fronts[ind][sortperm(distance,rev=true)[1:remaining]])
        end
        
        pop = pool[selected]

        # # Checking for convergence
        # current_convergence_metric = Statistics.median([isnan(v[1]) ? 0 : v[1] for v in pool_perf[selected]])

        # rel_change = abs(current_convergence_metric  / previous_convergence_metric - 1)

        # if (!isnothing(stopping_tol)) 
        #     if  rel_change < stopping_tol[1]
        #         no_change_counter += 1
        #     else
        #         no_change_counter = 0 # reset
        #     end
        #     if no_change_counter >= stopping_tol[2]
        #         if !isnothing(info_every) @info "Finished: No change bellow $(stopping_tol[1]) for the last $(stopping_tol[2]) generations" end
        #         break
        #     end
        #     if !isnothing(info_every) && mod(gen,info_every) == 0 @info "No change counter: $(no_change_counter)" end
        # end

        # previous_convergence_metric = current_convergence_metric

        if !isnothing(info_every) && mod(gen,info_every) == 0
            @info begin
                sizes = [get_n_metavariables(v) for v in pop]
                l = length(sizes[1])
                mins = [minimum(getindex.(sizes,i)) for i in 1:l]
                maxs = [maximum(getindex.(sizes,i)) for i in 1:l]
                
                """
                Generation $(gen) - $(hmss(t))
                Best per objective: $([maximum([isnan(v[i]) ? -Inf : v[i] for v in pool_perf]) for i in 1:length(pool_perf[1])])
                Minimum metavariables: $mins
                Maximum metavariables: $maxs
                """
                # Relative change: $rel_change
                # Current metric: $current_convergence_metric
            end
        end

        @debug "Number of ranks: $(maximum(rank))"
        scheduler(gen,state) # rel_change,
    end

    return pop
    
end

end