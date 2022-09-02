using Flatten

function polynomial_mutation(s::Param; p::P=0.5, η::E = 2) where {E <: Real, P <: Real}
    if (rand() < p)
        if s.val isa Bool return Param(!s.val,s.lb,s.ub) end
        r = rand()
        δ = r < 0.5 ? (2r)^(1/(η+1))-1 : 1 - (2*(1-r))^(1/(η+1))
        typeof(s.val) <: Integer ? convert(typeof(s),round(s + δ*(s.ub-s.lb))) : s + δ*(s.ub-s.lb)
    else
        s
    end 
end


function mutate_element(state::AbstractOptimParameters, elem::AbstractModel; pr::T = 0.5) where {T <: Real}
    new::typeof(elem) = modify(v -> isfixed(v) ? v : polynomial_mutation(v,η=state.ηm,p=state.pm), elem, Param)

    for g in flatten(elem,VLGroup)
        if rand() < pr
            n_remove = floor(Int,log(rand())/log(state.p_change_length))
            removable = [i for i in 1:length(g) if  !any(isfixed.(g.metavariables[i]))]
            # Only remove if there will remain at least two non-fixed nodes, because, otherwise the metavariable recombination does nothing
            if (n_remove != 0) && (length(removable) > (n_remove + 2))
                @debug "Removing $n_remove random knot(s)"
                for _ in 1:n_remove
                    new = deleteat(new,g.id,rand(removable)) 
                end
            end
        else

            n_add = floor(Int,log(rand())/log(state.p_change_length))

            if n_add != 0
                @debug "Inserting $n_add random knot(s)"            
                for _ in n_add
                    # Mutate an existing non-fixed knot
                    new = push(
                        new,
                        g.id,
                        modify(
                            v -> polynomial_mutation(v,p=state.pm,η=state.ηm),
                            rand([v for v in g.metavariables if !any(isfixed.(v))]),
                            Param
                        )
                    )

                end
            end
        end
    end
    return new
end


function similar_population(initial_s::AbstractModel, pop_size::Int; η::E = 500., pl::P = 0.2) where {E <: Real, P <: Real}
    pop = Vector{typeof(initial_s)}(undef,pop_size)
    @floop for i in 1:pop_size
        pop[i] = mutate_element(MutationParameters(promote(η,1.,pl)...), initial_s)
    end
    return pop
end

function similar_population(initial_s::AbstractModel, pop_size::Int, metric::Function; gen_multiplier::Int = 10, η::E = 500.,pl::P = 0.2) where {E <: Real, P <: Real}
    @assert gen_multiplier >= 1 "Generation multiplier should be at least 1"
    if gen_multiplier == 1 return similar_population(initial_s,pop_size, η = η) end

    size = pop_size*gen_multiplier
    pop = Vector{typeof(initial_s)}(undef,size)
    @floop for i in 1:(size)
        pop[i] = mutate_element(MutationParameters(promote(η,1.,pl)...), initial_s)
    end

    m = Vector{Number}(undef,size)
    @floop for i in 1:size
        m[i] = metric(pop[i])
    end


    return pop[sortperm([isnan(v) ? -Inf : v for v in m],rev=true)[1:pop_size]]
end


function mutate!(state, pop)
    @floop for i in 1:state.pop_size
        pop[i] = mutate_element(state,pop[i])
    end
end

# Tournament selection
function selection(rank, distance, number_in_tournament=2)
    pop_size = length(rank)

    mating_pool = Vector{Int}(undef,pop_size)
    @floop for i in 1:pop_size
        best = rand(1:pop_size)
        for _ in 1:number_in_tournament-1
            candidate = rand(1:pop_size)
            if nondominated_better(rank[candidate],rank[best],distance[candidate],distance[best])
                best = candidate
            end
        end
        mating_pool[i] = best
    end
    
    return mating_pool
end



function cut_and_slice_recombination(m1::AbstractVector{T},m2::AbstractVector{T}) where {T <: AbstractMetaVariable}
    points1 = sort(rand(1:length(m1),2))
    points2 = sort(rand(1:length(m2),2))
    
    # The +1 are required so that the same element is not considered multiple times
    vcat(m1[1:points1[1]],m2[points2[1]+1:points2[2]],m1[points1[2]+1:end]), vcat(m2[1:points2[1]],m1[points1[1]+1:points1[2]],m2[points2[2]+1:end])

end

# Similar-metavariable recombination - Reyerkerk et al. 2017
function similar_metavariable_recombination(m1::AbstractVector{T},m2::AbstractVector{T}) where {T <: AbstractMetaVariable}
    n1 = length(m1)
    n2 = length(m2)

    W = number_type(T)
    lb::Vector{W} = minimum(hcat([[v.lb for v in mi] for mi in vcat(m1,m2)]...),dims=2)[:]
    ub::Vector{W} = maximum(hcat([[v.ub for v in mi] for mi in vcat(m1,m2)]...),dims=2)[:]
    variable_range = ub-lb

    dissimilarity = Matrix{W}(undef,n1,n2)
    
    for k in 1:n2, j in 1:n1
        dissimilarity[j,k] = 0.5 * sum(abs.(m1[j] .- m2[k]) ./ variable_range) 
    end

    preference_p1 = argmin.(eachrow(dissimilarity))
    preference_p2 = argmin.(eachcol(dissimilarity))

    groups = Dict{Set{Int},Set{Int}}()
    missing_association = Set(1:n2)
    while !isempty(missing_association)
        p2_elem = pop!(missing_association)
        # For termination
        old_g_p1 = Set{Int}()
        # All elements of parent 2 that prefer the p1 element
        g_p1 = Set(findall(x -> x == p2_elem, preference_p1))
        # If none of parent 2 prefer the p1 element skip it
        if isempty(g_p1)
            continue
        end
        # To store associated elements of parent 1
        g_p2 = Set{Int}()
        while true
            # If the associated elements of parent one do not change break
            if old_g_p1 == g_p1
                break
            end
            
            # Updated in case they have changed
            old_g_p1 = g_p1

            # Add all elements of parent 1 that have the same preference in parent 2
            union!(g_p2, findall(x -> x in g_p1, preference_p2))

            # Add all elements of parent 2 that prefere one of the elements in the parent 1 group
            union!(g_p1,findall(x -> x in g_p2, preference_p1))
        end

        # Dead end
        if isempty(g_p2)
            continue
        end
        # Remove new association from missing
        for v in g_p2
            delete!(missing_association,v)
        end
        # Create new association
        groups[g_p1] = g_p2
    end


    nm = length(groups)
    n_swap = nm <= 3 ? 1 : rand(1:div(nm,2)) # How many to swap
    to_swap = rand(1:nm,n_swap) # Which ones to swap

    
    c1 = T[]
    c2 = T[]
    for (i,(k,v)) in enumerate(groups)
        if i in to_swap
            push!(c1,[m2[vi] for vi in v] ...)
            push!(c2,[m1[ki] for ki in k] ...)
        else
            push!(c1,[m1[ki] for ki in k] ...)
            push!(c2,[m2[vi] for vi in v] ...)
        end
    end

    return c1,c2
end



function simulated_binary_crossover(p1::Param{T},p2::Param{T};p::P=0.5,η::E=2) where {T,P <: Real,E <: Real}
    if rand() >= p return p1,p2 end
    u = rand()
    β = u <= 0.5 ? (2u)^(1/(η+1)) : (2(1-u))^(-1/(η+1))
    return 0.5((1-β)*p1+(1+β)*p2),  0.5((1+β)*p1+(1-β)*p2)
end

function uniform_crossover(p1::Param{T},p2::Param{T};p::P=0.5) where {T,P <: Real}
    if rand() < p
        return p2,p1
    else
        return p1,p2
    end
end


function crossover_elements(state, p1::T,p2::T) where {T <: AbstractModel}
    # Crossover fixed length fields
    fixed_p1 = flatten(p1,Param,VLGroup)
    fixed_p2 = flatten(p2,Param,VLGroup)

    fixed_v1,fixed_v2 = zip(map(v -> begin
            if isfixed(v[1]) || isfixed(v[2]) || !(typeof(v[1].val) <: AbstractFloat && typeof(v[2].val) <: AbstractFloat)
                return uniform_crossover(v[1],v[2],p=state.pc)
            else
                return simulated_binary_crossover(v[1],v[2],p=state.pc, η=state.ηc)
            end   
        end,
    zip(fixed_p1,fixed_p2))
    ...)

    fixed_c1 = reconstruct(p1,fixed_v1,Param,VLGroup)
    fixed_c2 = reconstruct(p2,fixed_v2,Param,VLGroup)

    vl_p1 = flatten(fixed_c1, VLGroup)
    vl_p2 = flatten(fixed_c2, VLGroup)

    vl_v1,vl_v2 = zip(map(g -> begin
            m1 = flatten(g[1],AbstractMetaVariable)
            m2 = flatten(g[2],AbstractMetaVariable)
            
            fixed_m1 = [any(isfixed.(v)) for v in m1]
            fixed_m2 = [any(isfixed.(v)) for v in m2]
            n1,n2 = similar_metavariable_recombination(
                collect(m1[map(!,fixed_m1)]),
                collect(m2[map(!,fixed_m2)])
            )
            VLGroup(g[1].id,(m1[fixed_m1]...,n1...)), VLGroup(g[2].id,(m2[fixed_m2]...,n2...))
        end,
        zip(vl_p1,vl_p2)
    )...)

    c1::T = reconstruct(fixed_c1,vl_v1,VLGroup)
    c2::T = reconstruct(fixed_c2,vl_v2,VLGroup)

    return c1,c2
end

function crossover(state,mating_pool,pop_size)
    new_pop = Vector{eltype(mating_pool)}(undef,pop_size)
    for i in 1:2:pop_size
        new_pop[i:i+1] .= crossover_elements(state,StatsBase.sample(mating_pool,2,replace=false)...)
    end

    return new_pop
end

                                        

# Has to return 0 if no constraint is violated or the ammount of violation if it violest some constraints
# Element wise constraints
function constraints(s::AbstractModel)
    #Add up distance bellow lower bound and above upper bound
    constraint_violation = sum(map(p -> isfixed(p) ? 0. : clamp((p.lb-p.val)/(p.ub-p.lb),0,Inf) + clamp((p.val-p.ub)/(p.ub-p.lb),0,Inf),flatten(s,Param)))
    
    return constraint_violation
end

# Has to return 0 if no constraint is violated or the ammount of violation if it violest some constraints
# Population wise constraints
function constraints(
    state::AbstractOptimParameters,
    pop::Vector{T},
    pop_perf::AbstractVector
    ) where T <: AbstractModel
    c_violation = constraints.(pop)
    if isnothing(state.window) return c_violation end

    best = findmax([isnan(v[1]) ? -Inf : v[1] for v in pop_perf])[2]
    n = get_n_metavariables(pop[best])
    @debug "Number of parmeters of best: $n"

    c_violation += clamp.([sum(abs.(n .- p)) for p in get_n_metavariables.(pop)] .- state.window,0,Inf)
    
    return c_violation
end



