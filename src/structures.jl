using UUIDs
using Flatten
using Distributions: Uniform
using AbstractNumbers
using Interpolations

# abstract type AbstractMutationProbability end

# struct FixedMutationProbability{T<:Number} <: AbstractMutationProbability
#     pm::T
# end

# struct BinomialMutationProbability{T<:Number} <: AbstractMutationProbability
#     mean::T
# end


# @inline mutation_probability(_::AbstractModel, m::FixedMutationProbability) = m.pm
# mutation_probability(elem::AbstractModel, m::FixedMutationProbability) = m.mean / length(elem)


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


uuid = UUIDs.uuid1


struct Param{T}  <: AbstractNumbers.AbstractNumber{T}
    val::T
    lb::T
    ub::T
end

Param(val::A,lb::B,ub::C) where {A,B,C}= Param(promote(val,lb,ub)...)
Param(val) = Param(val,typemin(val),typemax(val))
Param{T}(val::T) where T<: Number = Param(val,typemin(val),typemax(val))

Base.convert(::Type{Param{T}}, x::Param) where T = Param(convert(T,x.val),convert(T,x.lb),convert(T,x.ub))
Base.convert(::Type{Number}, x::Param) = convert(T,x.val)
AbstractNumbers.basetype(::Type{<: Param}) = Param
AbstractNumbers.number(x::Param) = x.val
AbstractNumbers.like(num::Param,x::Number) = Param(promote(x,num.lb,num.ub)...)


Base.:+(a::Param,b::Param) = Param(promote(a.val+b.val, a.lb >= b.lb ? a.lb : b.lb, a.ub <= b.ub ? a.ub : b.ub)...)
Base.:-(a::Param,b::Param) = Param(promote(a.val-b.val, a.lb >= b.lb ? a.lb : b.lb, a.ub <= b.ub ? a.ub : b.ub)...)
Base.:*(a::Param,b::Param) = Param(promote(a.val*b.val, a.lb >= b.lb ? a.lb : b.lb, a.ub <= b.ub ? a.ub : b.ub)...)
Base.:/(a::Param,b::Param) = Param(promote(a.val/b.val, a.lb >= b.lb ? a.lb : b.lb, a.ub <= b.ub ? a.ub : b.ub)...)

number_type(::Param{T}) where {T} = T
number_type(::Type{Param{T}}) where {T} = T

abstract type AbstractMetaVariable end

Base.iterate(m::AbstractMetaVariable,state=1) = state > nfields(m) ? nothing : (getfield(m,fieldname(typeof(m),state)),state+1)
#Base.convert(::Type{T},m::AbstractMetaVariable) where T = typeof(m)(convert.(T,m)...)

struct Point{T} <: AbstractMetaVariable
    x::Param{T}
    y::Param{T}
end

function Point(x::Number,xbounds::Tuple{Number,Number},y::Number,ybounds::Tuple{Number,Number}) 
    x,xlb,xub,y,ylb,yub = promote(x,xbounds...,y,ybounds...)
    Point(
        Param(x,xlb,xub),
        Param(y,ylb,yub)
    )
end

Point(x::Number,y::Number) = Point(x,(typemin(x),typemax(x)),y,(typemin(y),typemax(y)))
Point(range::Tuple{Number,Number}) = Point(0,range,0,range)
Point() = Point(0.,0.)
Base.convert(::Type{Point{T}},x::Point{W}) where {T,W} = Point(convert.(Param{T},x)...)

randomPoint(xrange::Tuple{Number,Number},yrange::Tuple{Number,Number}) = Point(rand(Uniform(xrange...)),xrange,rand(Uniform(yrange...)),yrange)
randomPoint(range::Tuple{Number,Number}) = randomPoint(range,range)

Base.length(::Point) = 2
number_type(::Point{T}) where {T} = T
number_type(::Type{Point{T}}) where {T} = T

struct VLGroup{W<: AbstractMetaVariable}
    id::UUID 
    metavariables::NTuple{N,W} where N
end

Base.iterate(m::VLGroup,state=1) = iterate(m.metavariables,state)
@inline Base.length(x::VLGroup) = length(x.metavariables)
Base.getindex(x::VLGroup{W},elems...) where {W} = getindex(x.metavariables,elems...)
Base.convert(::Type{VLGroup{T}},x::VLGroup{W}) where {T,W} = VLGroup(x.id,convert.(T,x.metavariables)) 


VLGroup(meta_constructor::Union{Function,Type{T}},n::Integer,args...) where T <: AbstractMetaVariable = VLGroup(uuid(),Tuple([meta_constructor([length(a) == n ? a[i] : a for a in args]...) for i in 1:n]))

abstract type AbstractModel{T} end


Base.eltype(_::AbstractModel{T}) where {T} = T
Base.length(x::AbstractModel) = length(flatten(x)) # Generic fallback, ovewrite for better performance

push(g::VLGroup,meta::AbstractMetaVariable) = VLGroup(g.id,(g.metavariables...,meta))
delete(g::VLGroup,meta::AbstractMetaVariable) = VLGroup(g.id,Tuple([m for m in g.metavariables if m != meta ]))
deleteat(g::VLGroup,ind::Integer) = VLGroup(g.id,Tuple([m for (i,m) in enumerate(g.metavariables) if i != ind ]))
deleteat(g::VLGroup,inds::AbstractVector{Int}) = VLGroup(g.id,Tuple([m for (i,m) in enumerate(g.metavariables) if !(i in inds)]))


push(m::AbstractModel,id::UUID, meta::AbstractMetaVariable) = modify(g -> g.id == id ? push(g,meta) : g, m,VLGroup)
delete(m::AbstractModel,id::UUID, meta::AbstractMetaVariable) = modify(g -> g.id == id ? delete(g,meta) : g, m,VLGroup)
deleteat(m::AbstractModel,id::UUID, ind::Integer) = modify(g -> g.id == id ? deleteat(g,ind) : g, m,VLGroup)
deleteat(m::AbstractModel,id::UUID, inds::AbstractVector{Int}) = modify(g -> g.id == id ? deleteat(g,inds) : g, m,VLGroup)

isfixed(p::Param) = (p.lb ≈ p.ub)
get_n_metavariables(m::AbstractModel) = (length(flatten(m,AbstractMetaVariable)),)

struct KnotModel{T} <: AbstractModel{T}
    m::Param{T}
    b::Param{T}
    knots::VLGroup{Point{T}}
end

Base.length(x::KnotModel) = nfields(x)-1+length(x.knots)
get_n_metavariables(x::KnotModel) = (length(x.knots),)
number_type(::KnotModel{T}) where {T} = T
number_type(::Type{KnotModel{T}}) where {T} = T


function correct_same_x!(xs::AbstractVector)
    unique_xs = unique(xs)
    if unique_xs == xs return nothing end
    for v in unique_xs
        inds = findall(x->x==v,xs)
        if length(inds) == 1 continue end
        for k in 2:length(inds)
            xs[inds[k]] = xs[inds[k-1]]+eps(xs[inds[k-1]])
        end
    end
    correct_same_x!(xs) # Might cause another same x when moving
end

function model_function_factory(m::KnotModel)
    xs = map(v -> v.x.val, m.knots)
    ys = map(v -> v.y.val, m.knots)
    perm = sortperm(xs)
    xs = xs[perm]
    ys = ys[perm]
    Interpolations.deduplicate_knots!(xs; move_knots = true)
    return extrapolate(interpolate(xs,ys,SteffenMonotonicInterpolation()), Flat())
end


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
            knot_x = [(Mx-mx) * rand() + mx for (mx,Mx) in xbounds]
            knot_y = [(My-my) * rand() + my for (my,My) in ybounds]
            KnotModel(Param((Mm-mm) * rand() + mm,mbounds...),Param((Mb-mb) * rand() + mb,bbounds...),
                VLGroup(Point,n_knots,knot_x,xbounds,knot_y,ybounds))
        end for _ in 1:pop_size
    ]   
end

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
    total = gen_multiplier*pop_size
    n_knots = length(xbounds)
    pop = [
        begin
            knot_x = [(Mx-mx) * rand() + mx for (mx,Mx) in xbounds]
            knot_y = [(My-my) * rand() + my for (my,My) in ybounds]
            KnotModel(Param((Mm-mm) * rand() + mm,mbounds...),Param((Mb-mb) * rand() + mb,bbounds...),
                VLGroup(Point,n_knots,knot_x,xbounds,knot_y,ybounds))
        end for _ in 1:total
    ]

    m = ThreadsX.map(metric,pop)

    return pop[sortperm([isnan(v) ? -Inf : v for v in m],rev=true)[1:pop_size]]
    
end


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
    total = gen_multiplier*pop_size
    mx,Mx = xbounds
    my,My = ybounds
    mm, Mm = mbounds
    mb, Mb = bbounds
    pop = [
        begin
            knot_x = range(mx,Mx,length=n_knots)
            bounds_x = vcat((mx,mx),fill((mx,Mx),n_knots-2),(Mx,Mx))
            knot_y = (My-my) .* rand(n_knots) .+ my
            bounds_y = fill((my,My),n_knots)
            KnotModel(Param((Mm-mm) * rand() + mm,mbounds...),Param((Mb-mb) * rand() + mb,bbounds...),
                VLGroup(Point,n_knots,knot_x,bounds_x,knot_y,bounds_y))
        end for _ in 1:total
    ]

    m = ThreadsX.map(metric,pop)

    return pop[sortperm([isnan(v) ? -Inf : v for v in m],rev=true)[1:pop_size]]
    
end



function random_population(
    n_knots::Integer,
    xbounds::Tuple,
    ybounds::Tuple,
    mbounds::Tuple,
    bbounds::Tuple,
    pop_size::Int,
    )
    mx,Mx = xbounds
    my,My = ybounds
    mm, Mm = mbounds
    mb, Mb = bbounds
    pop = [
        begin
            knot_x = range(mx,Mx,length=n_knots)
            bounds_x = vcat((mx,mx),fill((mx,Mx),n_knots-2),(Mx,Mx))
            knot_y = (My-my) .* rand(n_knots) .+ my
            bounds_y = fill((my,My),n_knots)
            KnotModel(Param((Mm-mm) * rand() + mm,mbounds...),Param((Mb-mb) * rand() + mb,bbounds...),
                VLGroup(Point,n_knots,knot_x,bounds_x,knot_y,bounds_y))
        end for _ in 1:pop_size
    ]

    return pop
    
end





