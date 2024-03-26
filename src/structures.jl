using UUIDs
using Flatten
using Distributions: Uniform
using AbstractNumbers
using Interpolations


struct FitnessEvaluation{T<:Real}
    val::AbstractVector{T}
    rank::Int
    distance::T
    elems::Set{Int}
end


abstract type AbstractOptimParameters end

struct MutationParameters{T<:Number} <: AbstractOptimParameters
    ηm::T
    p_change_length::T
end

struct _OptimParameters{T<:Number} <: AbstractOptimParameters
    pop_size::Integer
    ηm::T
    p_change_length::T
    ηc::T
    pc::T
    window::Union{Nothing,Integer}
    helper::Symbol
    #n_main_obj_elitism::Integer
end

struct OptimParameters{T<:Number} <: AbstractOptimParameters
    pop_size::Integer
    p_change_length::T
    pc::T
    window::Union{Nothing,Integer}
    helper::Symbol
    #n_main_obj_elitism::Integer
end


const uuid = UUIDs.uuid1


"""
    Param(val, lb, ub)

Parameter to be optimized, `val`, and its lower, `lb`, and upper, `ub`, bounds.
"""
struct Param{T} <: AbstractNumbers.AbstractNumber{T}
    val::T
    lb::T
    ub::T
end

Param(val::A, lb::B, ub::C) where {A,B,C} = Param(promote(val, lb, ub)...)
Param(val) = Param(val, typemin(val), typemax(val))
Param{T}(val::T) where {T<:Number} = Param(val, typemin(val), typemax(val))

Base.convert(::Type{Param{T}}, x::Param) where {T} = Param(convert(T, x.val), convert(T, x.lb), convert(T, x.ub))
Base.convert(T::Type{Number}, x::Param) = convert(T, x.val)
AbstractNumbers.basetype(::Type{<:Param}) = Param
AbstractNumbers.number(x::Param) = x.val
AbstractNumbers.like(num::Param, x::Number) = Param(promote(x, num.lb, num.ub)...)


Base.:+(a::Param, b::Param) = Param(promote(a.val + b.val, a.lb >= b.lb ? a.lb : b.lb, a.ub <= b.ub ? a.ub : b.ub)...)
Base.:-(a::Param, b::Param) = Param(promote(a.val - b.val, a.lb >= b.lb ? a.lb : b.lb, a.ub <= b.ub ? a.ub : b.ub)...)
Base.:*(a::Param, b::Param) = Param(promote(a.val * b.val, a.lb >= b.lb ? a.lb : b.lb, a.ub <= b.ub ? a.ub : b.ub)...)
Base.:/(a::Param, b::Param) = Param(promote(a.val / b.val, a.lb >= b.lb ? a.lb : b.lb, a.ub <= b.ub ? a.ub : b.ub)...)

number_type(::Param{T}) where {T} = T
number_type(::Type{Param{T}}) where {T} = T

abstract type AbstractMetaVariable end

Base.iterate(m::AbstractMetaVariable, state=1) = state > nfields(m) ? nothing : (getfield(m, fieldname(typeof(m), state)), state + 1)

"Metavariable to represent a 2-dimensional point"
struct Point{T} <: AbstractMetaVariable
    x::Param{T}
    y::Param{T}
end

"""
    Point(x, xbound, y, ybounds)

Return a Point with coordinates `x` and `y`, bounded by `xbounds` and `ybounds`, respectively.
"""
function Point(x::Number, xbounds::Tuple{Number,Number}, y::Number, ybounds::Tuple{Number,Number})
    x, xlb, xub, y, ylb, yub = promote(x, xbounds..., y, ybounds...)
    Point(
        Param(x, xlb, xub),
        Param(y, ylb, yub)
    )
end

"""
    Point(x, y)

Return an unbounded Point with coordinates `x` and `y`.
"""
Point(x::Number, y::Number) = Point(x, (typemin(x), typemax(x)), y, (typemin(y), typemax(y)))

"""
    Point(range)

Return a Point with coordinates (0,0) bound in both dimensions by range.
"""
Point(range::Tuple{Number,Number}) = Point(0, range, 0, range)


"""
    Point()

Return a unbounded Point with coordinates (0,0).
"""
Point() = Point(0.0, 0.0)

Base.convert(::Type{Point{T}}, x::Point{W}) where {T,W} = Point(convert.(Param{T}, x)...)

random_point(xrange::Tuple{Number,Number}, yrange::Tuple{Number,Number}) = Point(rand(Uniform(xrange...)), xrange, rand(Uniform(yrange...)), yrange)
random_point(range::Tuple{Number,Number}) = random_point(range, range)

Base.length(::Point) = 2
number_type(::Point{T}) where {T} = T
number_type(::Type{Point{T}}) where {T} = T

"Variable length group of subtype of `AbstractMetaVariable`."
struct VLGroup{W<:AbstractMetaVariable}
    id::UUID
    metavariables::NTuple{N,W} where {N}
end

Base.iterate(m::VLGroup, state=1) = iterate(m.metavariables, state)
@inline Base.length(x::VLGroup) = length(x.metavariables)
Base.getindex(x::VLGroup{W}, elems...) where {W} = getindex(x.metavariables, elems...)
Base.convert(::Type{VLGroup{T}}, x::VLGroup{W}) where {T,W} = VLGroup(x.id, convert.(T, x.metavariables))


VLGroup(meta_constructor::Union{Function,Type{T}}, n::Integer, args...) where {T<:AbstractMetaVariable} = VLGroup(uuid(), Tuple([meta_constructor([length(a) == n ? a[i] : a for a in args]...) for i in 1:n]))

abstract type AbstractModel{T} end


Base.eltype(_::AbstractModel{T}) where {T} = T
Base.length(x::AbstractModel) = length(flatten(x)) # Generic fallback, overwrite for better performance

push(g::VLGroup, meta::AbstractMetaVariable) = VLGroup(g.id, (g.metavariables..., meta))
delete(g::VLGroup, meta::AbstractMetaVariable) = VLGroup(g.id, Tuple([m for m in g.metavariables if m != meta]))
deleteat(g::VLGroup, ind::Integer) = VLGroup(g.id, Tuple([m for (i, m) in enumerate(g.metavariables) if i != ind]))
deleteat(g::VLGroup, inds::AbstractVector{Int}) = VLGroup(g.id, Tuple([m for (i, m) in enumerate(g.metavariables) if !(i in inds)]))


push(m::AbstractModel, id::UUID, meta::AbstractMetaVariable) = modify(g -> g.id == id ? push(g, meta) : g, m, VLGroup)
delete(m::AbstractModel, id::UUID, meta::AbstractMetaVariable) = modify(g -> g.id == id ? delete(g, meta) : g, m, VLGroup)
deleteat(m::AbstractModel, id::UUID, ind::Integer) = modify(g -> g.id == id ? deleteat(g, ind) : g, m, VLGroup)
deleteat(m::AbstractModel, id::UUID, inds::AbstractVector{Int}) = modify(g -> g.id == id ? deleteat(g, inds) : g, m, VLGroup)

isfixed(p::Param) = (p.lb ≈ p.ub)
get_n_metavariables(m::AbstractModel) = (length(flatten(m, AbstractMetaVariable)),)

"""
Simple model consisting of one `VLGroup`, a slope `m` and a y-intercept `b`.
"""
struct KnotModel{T} <: AbstractModel{T}
    m::Param{T}
    b::Param{T}
    knots::VLGroup{Point{T}}
end

Base.length(x::KnotModel) = nfields(x) - 1 + length(x.knots)
get_n_metavariables(x::KnotModel) = (length(x.knots),)
number_type(::KnotModel{T}) where {T} = T
number_type(::Type{KnotModel{T}}) where {T} = T




"""
    model_function_factory(m::AbstractModel)


Return a callable to evaluate the model `m` at given `x`.
"""
model_function_factory(m::AbstractModel) = error("Must implement `model_function_factory` for `$(typeof(m))`.")


function model_function_factory(m::KnotModel)
    xs = map(v -> v.x.val, m.knots)
    ys = map(v -> v.y.val, m.knots)
    perm = sortperm(xs)
    xs = xs[perm]
    ys = ys[perm]
    Interpolations.deduplicate_knots!(xs; move_knots=true)
    return extrapolate(interpolate(xs, ys, SteffenMonotonicInterpolation()), Flat())
end


