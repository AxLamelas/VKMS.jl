module VKMS

export evolve, VLGroup, AbstractMetaVariable, Point, randomPoint, similar_population,
    AbstractOptimParameters, OptimParameters, Param, AbstractModel, KnotModel,
    random_population, model_function_factory, get_n_metavariables

export LessFitness, MoreFitness, NoneFitness # BothFitness,


using Dates
using StatsBase
using Statistics
using ConstructionBase
using ProgressMeter
using ThreadsX
using StaticArrays
using Flatten


include("structures.jl")
include("operators.jl")
include("nondominated.jl")


abstract type AbstractFitness{N,T} end

struct LessFitness{N,T,F} <: AbstractFitness{N,T}
    functional::F
    x::Vector{T}
    y::Vector{T}
    weights::Vector{Float64}
    sigdigits::Int
    function LessFitness{N}(functional::F, x::Vector{T}, y::Vector{T}, weights::Vector{Float64}, sigdigits::Int) where {N,F,T}
        new{N,T,F}(functional, x, y, weights, sigdigits)
    end
end


function (f::LessFitness)(_::AbstractOptimParameters, m::AbstractModel)
    r = f.functional(f.x, m)
    nssr = round(sum(-f.weights[i] * (f.y[i] - r[i])^2 for i in eachindex(f.y)), sigdigits=f.sigdigits)
    return SVector((isnan(nssr) ? -Inf : nssr), (-convert(Float64, v) for v in get_n_metavariables(m))...)
end

struct MoreFitness{N,T,F} <: AbstractFitness{N,T}
    functional::F
    x::Vector{T}
    y::Vector{T}
    weights::Vector{Float64}
    sigdigits::Int
    function MoreFitness{N}(functional::F, x::Vector{T}, y::Vector{T}, weights::Vector{Float64}, sigdigits::Int) where {N,F,T}
        new{N,T,F}(functional, x, y, weights, sigdigits)
    end
end

function (f::MoreFitness)(_::AbstractOptimParameters, m::AbstractModel)
    r = f.functional(f.x, m)
    nssr = round(sum(-f.weights[i] * (f.y[i] - r[i])^2 for i in eachindex(f.y)), sigdigits=f.sigdigits)
    return SVector((isnan(nssr) ? -Inf : nssr), (convert(Float64, v) for v in get_n_metavariables(m))...)
end

struct NoneFitness{N,T,F} <: AbstractFitness{N,T}
    functional::F
    x::Vector{T}
    y::Vector{T}
    weights::Vector{Float64}
    sigdigits::Int
    function NoneFitness{N}(functional::F, x::Vector{T}, y::Vector{T}, weights::Vector{Float64}, sigdigits::Int) where {N,F,T}
        new{N,T,F}(functional, x, y, weights, sigdigits)
    end
end

function (f::NoneFitness)(_::AbstractOptimParameters, m::AbstractModel)
    r = f.functional(f.x, m)
    nssr = round(sum(-f.weights[i] * (f.y[i] - r[i])^2 for i in eachindex(f.y)), sigdigits=f.sigdigits)
    return SVector((isnan(nssr) ? -Inf : nssr),)
end


function evaluate!(perf, state, pop, fitness_function::AbstractFitness)
    ThreadsX.map!(perf, pop) do p
        fitness_function(state, p)
    end
    return nothing
end

function evaluate(state, pop, fitness_function::AbstractFitness{N,T}) where {T,N}
    perf = Vector{SVector{N,T}}(undef, length(pop))
    evaluate!(perf, state, pop, fitness_function)
    return perf
end



function evolve(pop::AbstractVector{T}, fitness_function::AbstractFitness{N,W}, parameters::OptimParameters; max_gen=nothing, max_time=nothing, terminate_on_front_collapse=true, progress=true)::Tuple{Vector{T},Int} where {T<:AbstractModel,W,N} # stopping_tol=nothing, #scheduler=identity_scheduler
    @assert any([!isnothing(c) for c in [max_gen, max_time]]) "Please define at least one stopping criterium" #,stopping_tol
    @assert length(pop) == parameters.pop_size "Inconsistancy between the legth of the population and the population size in the state"
    @assert rem(parameters.pop_size, 2) == 0 "Population size must be divisible by 2"

    η = 2.0 .^ (2:10) # Similar to simulated annealing
    state = _OptimParameters(parameters.pop_size, 2.0, parameters.p_change_length, 2.0, parameters.pc, parameters.window, parameters.helper)

    @info "Starting evolution with parameters $parameters"
    prog = if isnothing(max_gen)
        ProgressUnknown(dt=1e-9, desc="Evolving: ", showspeed=true, enabled=progress)
    else
        Progress(max_gen, dt=1e-9, desc="Evolving: ", showspeed=true, enabled=progress)
    end

    generate_showvalues(state, F, constraint_violation) = () -> [
        (:n, state.ηm),
        (Symbol("First front"), "$(join(sort([v.val => (length(v.elems),mean(constraint_violation[i] for i in v.elems)) for v in F[1]],rev=true),", ")) ($(sum(length(v.elems) for v in F[1]))/$(state.pop_size))")
    ]

    # Initialization of the pop candidate
    pool = vcat(pop, pop)
    pop_candidate = deepcopy(pop)
    mutate!(state, pop_candidate)


    pool_perf = Vector{SVector{N,W}}(undef, 2 * length(pop))
    selected = Vector{Int}(undef, state.pop_size)

    old_best_fitness = -Inf

    gen = 1
    start_time = now()
    while true
        if (!isnothing(max_gen)) && (gen >= max_gen)
            ProgressMeter.finish!(prog)
            @info "Finished: Reached maximum generation $(max_gen)"
            break
        end

        t = now() - start_time
        if (!isnothing(max_time)) && (t >= max_time)
            ProgressMeter.finish!(prog)
            @info "Finished: Reached maximum execution time $max_time"
            break
        end


        @debug "Current state: $state"

        pool[1:state.pop_size] .= pop
        pool[(state.pop_size+1):end] .= pop_candidate

        evaluate!(pool_perf, state, pool, fitness_function)
        constraint_violation = constraints(state, pool, pool_perf)
        fronts = fast_non_dominated_sort(pool_perf, constraint_violation)

        f1 = Tuple(pool_perf[i][1] for i in fronts[1] if constraint_violation[i] ≈ 0.0)
        best_fitness = isempty(f1) ? old_best_fitness : maximum(f1)

        if best_fitness < old_best_fitness
            @info "Best fitness decreased from $(old_best_fitness) by $(old_best_fitness-best_fitness)"
        end

        old_best_fitness = best_fitness

        if terminate_on_front_collapse && (length(fronts[1]) >= state.pop_size)
            if isempty(η)
                @info "Finished all η steps. Terminating..."
                break
            end
            v = popfirst!(η)
            @info "Increasing η to $v"
            state = setproperties(state, (ηm=v, ηc=v))
            # Delete from first front so that most elements of other fronts are included (reintroduces diversity)
            # Keep one or more copies of the unique elements of the first front
            # More than one copy might be necessary depending on the number of elements in the first front
            n_missing = state.pop_size - sum(length.(fronts[2:end])) + 1
            u = unique_dict(pool_perf[fronts[1]], fronts[1])
            to_keep = Int[]
            while length(to_keep) < n_missing
                for v in values(u)
                    push!(to_keep, pop!(v))
                end
                filter!(p -> !isempty(p.second), u)
            end
            filter!(x -> x in to_keep, fronts[1])
        end

        cursor = 0

        # Determine non-dominated rank that completely fits in pop_size
        ind = 0
        F = [Set{FitnessEvaluation{eltype(first(pool_perf))}}() for _ in 1:length(fronts)]
        for (i, fi) in enumerate(fronts)
            ufit = unique_dict(view(pool_perf, fi), fi)
            dist = crowding_distance(collect(keys(ufit)))
            union!(F[i], [FitnessEvaluation(k, i, dist[j], Set(v)) for (j, (k, v)) in enumerate(ufit)])

            if (cursor + length(fi)) > state.pop_size
                ind = i
                break
            end
            selected[(cursor+1):(cursor+length(fronts[i]))] .= fronts[i]
            cursor += length(fronts[i])
        end

        # Append the remaining base on distance
        remaining = state.pop_size - cursor
        if remaining == 0
            ind -= 1
        else
            S = last_front_selection(remaining, collect(F[ind]))
            for v in F[ind] # remove elements not selected from last front
                filter!(x -> x in S, v.elems)
            end
            filter!(x -> !isempty(x.elems), F[ind])

            selected[(cursor+1):end] .= S
        end

        for (i, j) in enumerate(selected)
            pop[i] = pool[j]
        end

        mating_pool = selection(state.pop_size, union((F[i] for i in 1:ind)...))
        crossover!(state, pop_candidate, pool, mating_pool)
        mutate!(state, pop_candidate)

        gen += 1
        ProgressMeter.next!(prog, showvalues=generate_showvalues(state, F, constraint_violation))

    end

    return pop, gen

end

end
