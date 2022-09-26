module VKMS

export evolve, VLGroup, AbstractMetaVariable, Point,randomPoint, similar_population,
AbstractOptimParameters, OptimParameters, Param, AbstractModel, fitness_factory, Model,
groupparams, best_by_size, KnotModel, random_population, model_function_factory, get_n_metavariables, first_front

using Dates
using StatsBase
using Statistics
using FLoops
using FoldsThreads
using ConstructionBase


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

function fitness_factory(functional::Function, x::AbstractVector,y::AbstractVector,weigths::AbstractVector; sigdigits=5)
    function fitness(state::AbstractOptimParameters,m::AbstractModel)
        residuals = y .- functional(x,m)
        ssr = round.(- residuals' * (weigths .* residuals),sigdigits=sigdigits)
        ssr = isnan(ssr) ? -Inf : ssr
        if state.helper == :more
            [ssr , sum(get_n_metavariables(m))]
        elseif state.helper == :less
            [ssr , - sum(get_n_metavariables(m))]
        elseif state.helper == :none
            [ssr]
        elseif state.helper == :both
            v = sum(get_n_metavariables(m))
            [ssr , - v, v]
        else
            throw(error("Unexpected helper: $(state.helper)"))
        end
    end
    return fitness
end

function fitness_factory(functional::Function, x::AbstractVector,y::AbstractVector; sigdigits=5)
    function fitness(state::AbstractOptimParameters,m::AbstractModel)
        residuals = y .- functional(x,m)
        ssr = round.(- residuals' * residuals; sigdigits=sigdigits)
        ssr = isnan(ssr) ? -Inf : ssr
        if state.helper == :more
            [ssr , sum(get_n_metavariables(m))]
        elseif state.helper == :less
            [ssr , - sum(get_n_metavariables(m))]
        elseif state.helper == :none
            [ssr]
        elseif state.helper == :both
            v = sum(get_n_metavariables(m))
            [ssr , - v, v]
        else
            throw(error("Unexpected helper: $(state.helper)"))
        end
    end
    return fitness
end


function evaluate(state, pop, fitness_function)
    pop_size = length(pop)
    perf = Vector{Vector{Float64}}(undef,pop_size)
    @floop WorkStealingEx() for i in 1:pop_size
        @inbounds perf[i] = fitness_function(state,pop[i])
    end
    return perf
end


function identity_scheduler(
    gen::Integer,
    pop::AbstractVector,
    perf::AbstractVector,
    constraint_violation::AbstractVector,
    rank::AbstractVector,
    p::AbstractOptimParameters)
    
    @debug("Identity scheduler")
    return pop,p
end

function first_front(state::OptimParameters, pop::AbstractVector{T},fitness_function) where {T <: AbstractModel}
    pop_perf = evaluate(state,pop,fitness_function)
    constraint_violation  = constraints.(pop)
    fronts = fast_non_dominated_sort(pop_perf,constraint_violation)
    return unique_dict(pop_perf[fronts[1]],fronts[1])
end

function evolve(pop::AbstractVector{T}, fitness_function::Function,parameters::OptimParameters; max_gen=nothing,max_time=nothing,terminate_on_front_collapse = true, info_every=50)::Tuple{Vector{T},Int} where {T<:AbstractModel} # stopping_tol=nothing, #scheduler=identity_scheduler
    @assert any([!isnothing(c) for c in [max_gen,max_time]]) "Please define at least one stopping criterium" #,stopping_tol
    @assert length(pop) == parameters.pop_size "Inconsistancy between the legth of the population and the population size in the state"
    @assert rem(parameters.pop_size,2) == 0 "Population size must be divisible by 2"
    start_time = now()
    gen = 0
    # no_change_counter = 0
    # previous_convergence_metric = Inf

    if !isnothing(info_every) @info "Starting evolution with parameters $parameters" end
    
    η = 2. .^(2:10) # Similar to simulated annealing
    state = _OptimParameters(parameters.pop_size,2.,parameters.p_change_length,2.,parameters.pc,parameters.window,parameters.helper)
    # Initialization of the pop candidate
    pop_candidate = deepcopy(pop)
    mutate!(state,pop_candidate)
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
        
        pool = vcat(pop,pop_candidate)
        pool_perf = evaluate(state,pool,fitness_function)
        constraint_violation = constraints(state, pool, pool_perf)
        fronts = fast_non_dominated_sort(pool_perf,constraint_violation)

        if terminate_on_front_collapse && (length(fronts[1]) >= state.pop_size)
            if isempty(η)
                @info "Finished all η steps. Terminating..."
                break
            end
            v = popfirst!(η)
            if !isnothing(info_every) @info "Increasing η to $v" end
            state = setproperties(state,(ηm=v,ηc=v))
            # Delete from first front so that most elements of other fronts are included (reintroduces diversity)
            # Keep one or more copies of the unique elements of the first front
            # More than one copie might be necessary depending on the number of elements in the first front
            n_missing = state.pop_size - sum(length.(fronts[2:end])) + 1
            u = unique_dict(pool_perf[fronts[1]],fronts[1])
            to_keep = Int[]
            while length(to_keep) < n_missing
                for v in values(u)
                    push!(to_keep,pop!(v))
                end
                filter!(p -> !isempty(p.second),u)
            end
            filter!(x -> x in to_keep,fronts[1])
        end        

        selected = Set{Int}()
        # # Main obj elitism
        # if (state.n_main_obj_elitism != 0)
        #     union!(selected, sortperm(
        #         [isnan(v[1]) || constraint_violation[i] != 0. ? -Inf : v[1] for (i,v) in enumerate(pool_perf)],rev=true
        #         )[1:state.n_main_obj_elitism]
        #     )
        # end
        
        # Determine non-dominated rank that complitely fits in pop_size
        ind = 0
        F = [Set{FitnessEvaluation{ eltype(first(pool_perf))}}() for _ in 1:length(fronts)]
        for (i,fi) in enumerate(fronts)
            ufit = unique_dict(pool_perf[fi],fi)
            dist = crowding_distance(collect(keys(ufit)))
            union!(F[i],[FitnessEvaluation(k,i,dist[j],Set(v)) for (j,(k,v)) in  enumerate(ufit)])
            
            if (length(selected) + length(fi)) > state.pop_size
                ind = i
                break
            end
            union!(selected,fronts[i])
        end
        
        # Append the remaining base on distance
        remaining = state.pop_size-length(selected)
        if remaining == 0
            ind -= 1
        else
            S = last_front_selection(remaining, collect(F[ind]))
            for v in F[ind] # remove elements not selected from last front
                filter!(x -> x in S,v.elems)
            end
            filter!(x -> !isempty(x.elems), F[ind])

            union!(selected,S)
        end
        
        indexes = collect(selected)
        pop = pool[indexes]

        mating_pool = selection(state.pop_size,union(F[1:ind]...))
        crossover!(state,pop_candidate,pool[mating_pool])
        mutate!(state,pop_candidate)


        if !isnothing(info_every) && mod(gen-1,info_every) == 0
            @info begin
                "\nGeneration $(gen) - η = $(state.ηm) - $(hmss(t)) \nFirst front ($(sum(length(v.elems) for v in F[1]))/$(state.pop_size)):\n" * join(sort([v.val => length(v.elems) for v in F[1]],rev=true),"\n")
            end
        end


        #pop, state = scheduler(gen,pop,pool_perf[indexes],constraint_violation[indexes],rank[indexes],state)

        # # Emigration
        # n_emigrants = 3
        # emigrants = [mutate_element(MutationParameters(1.,0.9,0.3),pop[1]) for _ in 1:n_emigrants]
        # pop[end-n_emigrants+1:end] = emigrants
    end

    return pop, gen
    
end

end