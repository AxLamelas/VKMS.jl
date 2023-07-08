module VKMS

export evolve, VLGroup, AbstractMetaVariable, Point,randomPoint, similar_population,
AbstractOptimParameters, OptimParameters, Param, AbstractModel, best_by_size, KnotModel, random_population, model_function_factory, get_n_metavariables, first_front

export LessFitness, MoreFitness,  NoneFitness # BothFitness,


using Dates
using StatsBase
using Statistics
using ConstructionBase
using ProgressMeter
using ThreadsX
using StaticArrays


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

function best_by_size(pop::AbstractVector{<:AbstractModel},perf::AbstractVector{<:Number},::Val{true})
    sizes = [sum(get_n_metavariables(p)) for p in pop]
    u = sort(unique(sizes))
    return Dict([(v,pop[argmax([ s == v ? p : -Inf for (s,p) in zip(sizes,perf)])]) for v in u])
end

function best_by_size(pop::AbstractVector{<:AbstractModel},perf::AbstractVector{<:Number})
    sizes = [sum(get_n_metavariables(p)) for p in pop]
    u = sort(unique(sizes))
    return Dict([(v,argmax([ s == v ? p : -Inf for (s,p) in zip(sizes,perf)])) for v in u])
end

abstract type AbstractWorkspace end

abstract type AbstractFitness{N,T} end

struct LessFitness{N,T,F,C<:AbstractWorkspace} <: AbstractFitness{N,T}
    functional::F
    ws::C
    y::Vector{T}
    weights::Vector{Float64}
    sigdigits::Int
    function LessFitness{N}(functional::F, ws::C, y::Vector{T}, weights::Vector{Float64}, sigdigits::Int) where {N,F,C,T}
        new{N,T,F,C}(functional,ws,y,weights,sigdigits)
    end
end


function (f::LessFitness)(_::AbstractOptimParameters,m::AbstractModel)
    r = f.functional(f.ws,m)
    nssr = round(sum(-f.weights[i] * (f.y[i] - r[i])^2 for i in eachindex(f.y)), sigdigits=f.sigdigits)
    return SVector((isnan(nssr) ? -Inf : nssr), (-convert(Float64,v) for v in get_n_metavariables(m))...)
end

struct MoreFitness{N,T,F,C<:AbstractWorkspace} <: AbstractFitness{N,T}
    functional::F
    ws::C
    y::Vector{T}
    weights::Vector{Float64}
    sigdigits::Int
    function MoreFitness{N}(functional::F, ws::C, y::Vector{T}, weights::Vector{Float64}, sigdigits::Int) where {N,F,C,T}
        new{N,T,F,C}(functional,ws,y,weights,sigdigits)
    end
end

function (f::MoreFitness)(_::AbstractOptimParameters,m::AbstractModel)
    r = f.functional(f.ws,m)
    nssr = round(sum(-f.weights[i] * (f.y[i] - r[i])^2 for i in eachindex(f.y)), sigdigits=f.sigdigits)
    return SVector((isnan(nssr) ? -Inf : nssr), (convert(Float64,v) for v in get_n_metavariables(m))...)
end

struct NoneFitness{N,T,F,C<:AbstractWorkspace} <: AbstractFitness{N,T}
    functional::F
    ws::C
    y::Vector{T}
    weights::Vector{Float64}
    sigdigits::Int
    function NoneFitness{N}(functional::F, ws::C, y::Vector{T}, weights::Vector{Float64}, sigdigits::Int) where {N,F,C,T}
        new{N,T,F,C}(functional,ws,y,weights,sigdigits)
    end
end

function (f::NoneFitness)(_::AbstractOptimParameters,m::AbstractModel)
    r = f.functional(f.ws,m)
    nssr = round(sum(-f.weights[i] * (f.y[i] - r[i])^2 for i in eachindex(f.y)), sigdigits=f.sigdigits)
    return SVector((isnan(nssr) ? -Inf : nssr),)
end
   

# struct BothFitness{F,T,C<:AbstractWorkspace} <: AbstractFitness
#     functional::F
#     ws::C
#     y::T
#     weights::Vector{Float64}
#     sigdigits::Int
# end
#
# function (f::BothFitness)(_::AbstractOptimParameters,m::AbstractModel)
#     r = f.functional(f.ws,m)
#     nssr = round(sum(-f.weights[i] * (f.y[i] - r[i])^2 for i in eachindex(f.y)), sigdigits=f.sigdigits)
#     return [isnan(nssr) ? -Inf : nssr, sum(get_n_metavariables(m)), -sum(get_n_metavariables(m))]
# end
 
function evaluate!(perf, state, pop, thread_fitness::AbstractVector{<:AbstractFitness})
    ThreadsX.map!(perf,pop) do p 
        thread_fitness[Threads.threadid()](state,p)
    end
    return nothing
end
    
function evaluate!(perf,state, pop, fitness_function::AbstractFitness)
    for i in 1:length(pop)
        @inbounds perf[i] = fitness_function(state,pop[i])
    end
    return nothing
end

function evaluate(state,pop,fitness_function::AbstractFitness{N,T})  where {T,N}
    perf = Vector{SVector{N,T}}(undef,length(pop))
    evaluate!(perf,state,pop,fitness_function)
    return perf
end


# function identity_scheduler(
#     gen::Integer,
#     pop::AbstractVector,
#     perf::AbstractVector,
#     constraint_violation::AbstractVector,
#     rank::AbstractVector,
#     p::AbstractOptimParameters)
#     
#     @debug("Identity scheduler")
#     return pop,p
# end

function first_front(state::OptimParameters, pop::AbstractVector{T},fitness_function) where {T <: AbstractModel}
    pop_perf = evaluate(state,pop,fitness_function)
    constraint_violation  = constraints.(pop)
    fronts = fast_non_dominated_sort(pop_perf,constraint_violation)
    return unique_dict(pop_perf[fronts[1]],fronts[1])
end

function evolve(pop::AbstractVector{T}, fitness_function::AbstractFitness{N,W},parameters::OptimParameters; max_gen=nothing,max_time=nothing,terminate_on_front_collapse = true, progress=true)::Tuple{Vector{T},Int} where {T<:AbstractModel, W, N} # stopping_tol=nothing, #scheduler=identity_scheduler
    @assert any([!isnothing(c) for c in [max_gen,max_time]]) "Please define at least one stopping criterium" #,stopping_tol
    @assert length(pop) == parameters.pop_size "Inconsistancy between the legth of the population and the population size in the state"
    @assert rem(parameters.pop_size,2) == 0 "Population size must be divisible by 2"

    η = 2. .^(2:10) # Similar to simulated annealing
    state = _OptimParameters(parameters.pop_size,2.,parameters.p_change_length,2.,parameters.pc,parameters.window,parameters.helper)
    
    @info "Starting evolution with parameters $parameters"
    prog = if isnothing(max_gen) 
        ProgressUnknown(dt=1e-9, desc="Evolving: ", showspeed=true,enabled=progress)
    else
        Progress(max_gen,dt=1e-9, desc="Evolving: ", showspeed=true, enabled=progress)
    end

    generate_showvalues(state,F,constraint_violation) = () -> [
            (:n,state.ηm),
            (Symbol("First front"),"$(join(sort([v.val => (length(v.elems),mean(constraint_violation[i] for i in v.elems)) for v in F[1]],rev=true),", ")) ($(sum(length(v.elems) for v in F[1]))/$(state.pop_size))")
        ]

    thread_fitness = [deepcopy(fitness_function) for _ in 1:Threads.nthreads()]
   
    # Initialization of the pop candidate
    pool = vcat(pop,pop)
    pop = view(pool,1:state.pop_size)
    pop_candidate = view(pool,state.pop_size + 1 : 2state.pop_size)
    mutate!(state,pop_candidate)


    pool_perf = Vector{SVector{N,W}}(undef,2*length(pop))
   
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
        
        evaluate!(pool_perf,state,pool,thread_fitness)
        constraint_violation = constraints(state, pool, pool_perf)
        fronts = fast_non_dominated_sort(pool_perf,constraint_violation)

        if terminate_on_front_collapse && (length(fronts[1]) >= state.pop_size)
            if isempty(η)
                @info "Finished all η steps. Terminating..."
                break
            end
            v = popfirst!(η)
            @info "Increasing η to $v"
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
            ufit = unique_dict(view(pool_perf,fi),fi)
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
        
        for (i,j) in enumerate(selected)
            pop[i] = pool[j]
        end

        mating_pool = selection(state.pop_size,union((F[i] for i in 1:ind)...))
        crossover!(state,pop_candidate,pool[mating_pool])
        mutate!(state,pop_candidate)

        gen +=  1
        ProgressMeter.next!(prog,showvalues = generate_showvalues(state,F,constraint_violation))
        
    end

    return pop, gen
    
end

end
