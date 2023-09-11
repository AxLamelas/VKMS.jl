"""
    similar_population(initial_s, pop_size[, metric]; η, pl[, gen_multiplier])

Generate a population of `pop_size` by mutating the given the element `initial_s`.

If metric is given, `pop_size * gen_multiplier` elements are generated and are ranked according to `metric`.
The `pop_size` best are then selected and return. `metric` must be a function that return a single value given an element.

`η` is the polynomial mutation index (higher `η` means higher variance in the population) and `pl`
is the probability to change the number of metavariables in the element.
"""
function similar_population(initial_s::AbstractModel, pop_size::Int, metric::Function;
    gen_multiplier::Int=10, η::E=500.0, pl::P=0.2) where {E<:Real,P<:Real}
    @assert gen_multiplier >= 1 "Generation multiplier should be at least 1"
    if gen_multiplier == 1
        return similar_population(initial_s, pop_size, η=η)
    end

    size = pop_size * gen_multiplier
    pop = ThreadsX.map(1:size) do _
        mutate_element(MutationParameters(promote(η, pl)...), initial_s)
    end

    m = ThreadsX.map(metric, pop)

    return pop[sortperm([isnan(v) ? -Inf : v for v in m], rev=true)[1:pop_size]]
end

function similar_population(initial_s::AbstractModel,
    pop_size::Int; η::E=500.0, pl::P=0.2) where {E<:Real,P<:Real}
    return ThreadsX.map(1:pop_size) do _
        mutate_element(MutationParameters(promote(η, pl)...), initial_s)
    end
end

"""
    random_population(xbounds::AbstractVector{<:Tuple}, ybounds::AbstractVector{<:Tuple},
    mbounds::Tuple, bbounds::Tuple, pop_size)

Generate a population of `pop_size` of `KnotModel` with `xbounds` and `ybounds` for each
knot, `mmbounds` for the slope and `bbounds` for the y-intercept.
"""
function random_population(
    xbounds::AbstractVector{<:Tuple},
    ybounds::AbstractVector{<:Tuple},
    mbounds::Tuple,
    bbounds::Tuple,
    pop_size::Int
)
    @assert length(xbounds) == length(ybounds) "Bounds must have the same length"
    mm, Mm = mbounds
    mb, Mb = bbounds
    n_knots = length(xbounds)
    return [
        begin
            knot_x = [(Mx - mx) * rand() + mx for (mx, Mx) in xbounds]
            knot_y = [(My - my) * rand() + my for (my, My) in ybounds]
            KnotModel(Param((Mm - mm) * rand() + mm, mbounds...), Param((Mb - mb) * rand() + mb, bbounds...),
                VLGroup(Point, n_knots, knot_x, xbounds, knot_y, ybounds))
        end for _ in 1:pop_size
    ]
end

"""
    random_population(xbounds::AbstractVector{<:Tuple}, ybounds::AbstractVector{<:Tuple},
    mbounds::Tuple, bbounds::Tuple, pop_size, metric; gen_multiplier)

Generate a population of `pop_size` of `KnotModel` with `xbounds` and `ybounds` for each
knot, `mmbounds` for the slope and `bbounds` for the y-intercept.

`pop_size * gen_multiplier` elements are generated and are ranked according to `metric`.
The `pop_size` best are then selected and return. `metric` must be a function that return a single value given an element.
"""
function random_population(
    xbounds::AbstractVector{<:Tuple},
    ybounds::AbstractVector{<:Tuple},
    mbounds::Tuple,
    bbounds::Tuple,
    pop_size::Int,
    metric::Function;
    gen_multiplier::Int=10
)
    @assert length(xbounds) == length(ybounds) "Bounds must have the same length"
    mm, Mm = mbounds
    mb, Mb = bbounds
    total = gen_multiplier * pop_size
    n_knots = length(xbounds)
    pop = [
        begin
            knot_x = [(Mx - mx) * rand() + mx for (mx, Mx) in xbounds]
            knot_y = [(My - my) * rand() + my for (my, My) in ybounds]
            KnotModel(Param((Mm - mm) * rand() + mm, mbounds...), Param((Mb - mb) * rand() + mb, bbounds...),
                VLGroup(Point, n_knots, knot_x, xbounds, knot_y, ybounds))
        end for _ in 1:total
    ]

    m = ThreadsX.map(metric, pop)

    return pop[sortperm([isnan(v) ? -Inf : v for v in m], rev=true)[1:pop_size]]

end


"""
    random_population(xbounds::Tuple, ybounds::Tuple,
    mbounds::Tuple, bbounds::Tuple, pop_size)

Generate a population of `pop_size` of `KnotModel` with each element having `n_knots`
all with equal `xbounds` and `ybounds`. The slope is bounde by `mbounds` and the y-intercept
by `bbounds`.
"""
function random_population(
    n_knots::Integer,
    xbounds::Tuple,
    ybounds::Tuple,
    mbounds::Tuple,
    bbounds::Tuple,
    pop_size::Int,
)
    mx, Mx = xbounds
    my, My = ybounds
    mm, Mm = mbounds
    mb, Mb = bbounds
    pop = [
        begin
            knot_x = range(mx, Mx, length=n_knots)
            bounds_x = vcat((mx, mx), fill((mx, Mx), n_knots - 2), (Mx, Mx))
            knot_y = (My - my) .* rand(n_knots) .+ my
            bounds_y = fill((my, My), n_knots)
            KnotModel(Param((Mm - mm) * rand() + mm, mbounds...), Param((Mb - mb) * rand() + mb, bbounds...),
                VLGroup(Point, n_knots, knot_x, bounds_x, knot_y, bounds_y))
        end for _ in 1:pop_size
    ]

    return pop

end

"""
    random_population(xbounds::Tuple, ybounds::Tuple,
    mbounds::Tuple, bbounds::Tuple, pop_size)

Generate a population of `pop_size` of `KnotModel` with each element having `n_knots`
all with equal `xbounds` and `ybounds`. The slope is bounde by `mbounds` and the y-intercept
by `bbounds`.

`pop_size * gen_multiplier` elements are generated and are ranked according to `metric`.
The `pop_size` best are then selected and return. `metric` must be a function that return a single value given an element.

"""
function random_population(
    n_knots::Integer,
    xbounds::Tuple,
    ybounds::Tuple,
    mbounds::Tuple,
    bbounds::Tuple,
    pop_size::Int,
    metric::Function;
    gen_multiplier::Int=10
)
    total = gen_multiplier * pop_size
    mx, Mx = xbounds
    my, My = ybounds
    mm, Mm = mbounds
    mb, Mb = bbounds
    pop = [
        begin
            knot_x = range(mx, Mx, length=n_knots)
            bounds_x = vcat((mx, mx), fill((mx, Mx), n_knots - 2), (Mx, Mx))
            knot_y = (My - my) .* rand(n_knots) .+ my
            bounds_y = fill((my, My), n_knots)
            KnotModel(Param((Mm - mm) * rand() + mm, mbounds...), Param((Mb - mb) * rand() + mb, bbounds...),
                VLGroup(Point, n_knots, knot_x, bounds_x, knot_y, bounds_y))
        end for _ in 1:total
    ]

    m = ThreadsX.map(metric, pop)

    return pop[sortperm([isnan(v) ? -Inf : v for v in m], rev=true)[1:pop_size]]

end








