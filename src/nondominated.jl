 

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
    fronts = [Int[] for i in 1:2*length(pop_perf)]
    Sp = [Int[] for i in 1:length(pop_perf)]
    np = zeros(Int,length(pop_perf))
    rank = zeros(Int,length(pop_perf))
    for (i,p) in enumerate(pop_perf)
        for (j,q) in enumerate(pop_perf)
            if constraint_dominates(p,q,constraint_violation[i],constraint_violation[j])
                push!(Sp[i],j)
            elseif constraint_dominates(q,p,constraint_violation[j],constraint_violation[i])
                np[i] += 1
            end
        end
        if np[i] == 0
            rank[i] = 1
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
                    rank[k] = i+1
                    push!(Q,k)
                end
            end
        end
        i += 1
        fronts[i] = Q
    end
    return rank, [v for v in fronts if !isempty(v)]
end
            
function crowding_distance(front)
    l = length(front)
    distance = zeros(l)
    matrix = hcat(front...)
    for m in 1:length(front[1])
        perm = sortperm(matrix[m,:])
        distance[perm[1]] = Inf
        distance[perm[end]] = Inf
        for i in 2:l-1
            distance[i] += (matrix[m,perm[i+1]]-matrix[m,perm[i-1]])/(matrix[m,perm[end]]-matrix[m,perm[1]])
        end
    end
    return  distance
end

function crowding_distance(pop_perf,ranks)
    ref_inds = 1:length(pop_perf)
    inds = Int[]
    d = []
    for r in unique(ranks)
        append!(inds,ref_inds[ranks .== r])
        append!(d,crowding_distance(pop_perf[ranks .== r]))
    end
    return d[sortperm(inds)]
end


nondominated_better(ranki,rankj,distancei,distancej) = (ranki < rankj) | ((ranki == rankj) & (distancei > distancej))