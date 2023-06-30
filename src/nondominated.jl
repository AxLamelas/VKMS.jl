 

dominates(p,q) = all(p .>= q) & any(p .> q)

function constraint_dominates(p_perf,q_perf,p_constraint_violation,q_constraint_violation)
    if (p_constraint_violation == 0) & !(q_constraint_violation == 0)
        return true
    elseif (p_constraint_violation == 0) & (q_constraint_violation == 0)
        return dominates(p_perf,q_perf)
    elseif p_constraint_violation < q_constraint_violation 
        return true
    else
        return false
    end
end

function fast_non_dominated_sort(pop_perf,constraint_violation)
    fronts = [Int[] for _ in 1:2*length(pop_perf)]
    Sp = [Int[] for _ in 1:length(pop_perf)]
    np = zeros(Int,length(pop_perf))
    for (i,p) in enumerate(pop_perf)
        for (j,q) in enumerate(pop_perf)
            if constraint_dominates(p,q,constraint_violation[i],constraint_violation[j])
                push!(Sp[i],j)
            elseif constraint_dominates(q,p,constraint_violation[j],constraint_violation[i])
                np[i] += 1
            end
        end
        if np[i] == 0
            push!(fronts[1],i)
        end
    end
        
    i = 1
    while !isempty(fronts[i])
        Q = []
        for j in fronts[i]
            for k in Sp[j]
                np[k] -= 1
                if np[k] == 0
                    push!(Q,k)
                end
            end
        end
        i += 1
        fronts[i] = Q
    end
    return filter!(x -> !isempty(x),fronts)
end
            
function crowding_distance(front)
    l = length(front)
    distance = zeros(l)
    matrix = vcat(front'...)
    for m in 1:length(front[1])
        perm = sortperm(matrix[:,m])
        distance[perm[1]] = Inf
        distance[perm[end]] = Inf
        r = (matrix[perm[end],m]-matrix[perm[1],m])
        if r == 0
            distance[perm[2:l-1]] .= 0.
            continue
        end
        for i in 2:l-1
            distance[perm[i]] += (matrix[perm[i+1],m]-matrix[perm[i-1],m])/r
        end
    end
    return  distance
end

function crowding_distance(pop_perf,ranks)
    d = Vector{Float64}(undef,length(pop_perf))
    for r in unique(ranks)
        roi = ranks .== r
        d[roi] = crowding_distance(pop_perf[roi])
    end
    return d
end

function euclidian_distance(front)
    n = length(front)
    dist = Inf * ones(n,n)
    for i in 1:n
        for j in (i+1):n
            v = front[i] .- front[j]
            dist[i,j] = v' * v
            dist[j,i] = dist[i,j]
        end
    end
    return minimum(dist;dims=2)[:]
end

function euclidian_distance(pop_perf,ranks)
    d = Vector{Float64}(undef,length(pop_perf))
    for r in unique(ranks)
        roi = ranks .== r
        d[roi] = euclidian_distance(pop_perf[roi])
    end
    return d
end

nondominated_better(ranki,rankj,distancei,distancej) = (ranki < rankj) | ((ranki == rankj) & (distancei > distancej))