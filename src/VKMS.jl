module VKMS

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

function fitness_factory(functional::Function, x::AbstractVector,y::AbstractVector)
    function fitness(state::OptimParameters,m::AbstractModel)
        residuals = y .- functional(x,m)
        if state.helper == :more
            [- residuals' * residuals , sum(get_n_metavariables(m))]
        elseif state.helper == :less
            [- residuals' * residuals , - sum(get_n_metavariables(m))]
        elseif state.helper == :none
            [- residuals' * residuals]
        elseif state.helper == :both
            v = sum(get_n_metavariables(m))
            [- residuals' * residuals , - v, v]
        else
            throw(error("Unexpected helper: $(state.helper)"))
        end
    end
    return fitness
end


function evaluate(state, pop, fitness_function)
    pop_size = length(pop)
    perf = Vector{Vector{Float64}}(undef,pop_size)
    @batch for i in 1:pop_size
        perf[i] = fitness_function(state,pop[i])
    end
    return perf
end


@inline scheduler(gen::Integer,pop::AbstractVector,perf::AbstractVector,p::AbstractOptimParameters) = pop,p # rel_change::Number,

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

        pop, state = scheduler(gen,pop,pool_perf[selected],state)

        if !isnothing(info_every) && mod(gen-1,info_every) == 0
            @info begin
                sizes = [get_n_metavariables(v) for v in pop[constraint_violation .== 0.]]
                l = length(sizes[1])
                mins = [minimum(getindex.(sizes,i)) for i in 1:l]
                maxs = [maximum(getindex.(sizes,i)) for i in 1:l]
                
                """
                Generation $(gen) - $(hmss(t))
                Best per objective: $([maximum([isnan(v[i]) ? -Inf : v[i] for v in pop_perf[constraint_violation .== 0.]]) for i in 1:length(pop_perf[1])])
                Minimum metavariables: $mins
                Maximum metavariables: $maxs
                """
                # Relative change: $rel_change
                # Current metric: $current_convergence_metric
            end
        end

        @debug "Number of ranks: $(maximum(rank))"


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
        try
            if (state.n_main_obj_elitism != 0)
                append!(selected, sortperm(
                    [isnan(v[1]) ? -Inf : v[1] for v in pool_perf[constraint_violation .== 0.]],rev=true
                    )[1:state.n_main_obj_elitism]
                )
                
            end
        catch 
            nothing
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

        # Emigration
        n_emigrants = 3
        emigrants = [mutate_element(MutationParameters(1,0.9,0.3),pop[1]) for _ in 1:3]
        pop[end-n_emigrants:end] = emigrants
    end

    return pop
    
end

end