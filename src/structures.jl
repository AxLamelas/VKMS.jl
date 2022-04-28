using UUIDs
using Flatten
using Distributions: Uniform
using AbstractNumbers

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
Base.:-(a::Param,b::Param) = Param(promote(a.val-b.val, a.lb >= b.lb ? a.lb : b.lb, a.ub <= b.ub ? a.ub : b.ub)...)

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

struct VLGroup{N,W<: AbstractMetaVariable}
    id::UUID 
    metavariables::NTuple{N,W}
end

Base.iterate(m::VLGroup,state=1) = iterate(m.metavariables,state)
@inline Base.length(x::VLGroup) = length(x.metavariables)
Base.getindex(x::VLGroup{N,W},elems...) where {N,W} = getindex(x.metavariables,elems...)
Base.convert(::Type{VLGroup{N,T}},x::VLGroup{N,W}) where {N,T,W} = VLGroup(x.id,convert.(T,x.metavariables)) 


VLGroup(meta_constructor::Union{Function,Type{T}},n::Integer,args...) where T <: AbstractMetaVariable = VLGroup(uuid(),Tuple([meta_constructor([length(a) == n ? a[i] : a for a in args]...) for i in 1:n]))

abstract type AbstractModel end

Base.length(x::AbstractModel) = length(flatten(x)) # Generic fallback, ovewrite for better performance

push(g::VLGroup,meta::AbstractMetaVariable) = VLGroup(g.id,(g.metavariables...,meta))
delete(g::VLGroup,meta::AbstractMetaVariable) = VLGroup(g.id,Tuple([m for m in g.metavariables if m != meta ]))
deleteat(g::VLGroup,ind::Integer) = VLGroup(g.id,Tuple([m for (i,m) in enumerate(g.metavariables) if i != ind ]))


push(m::AbstractModel,id::UUID, meta::AbstractMetaVariable) = modify(g -> g.id == id ? push(g,meta) : g, m,VLGroup)
delete(m::AbstractModel,id::UUID, meta::AbstractMetaVariable) = modify(g -> g.id == id ? delete(g,meta) : g, m,VLGroup)
deleteat(m::AbstractModel,id::UUID, ind::Integer) = modify(g -> g.id == id ? deleteat(g,ind) : g, m,VLGroup)

isfixed(p::Param) = p.lb == p.ub
get_n_metavariables(m::AbstractModel)::Int = length(flatten(m,AbstractMetaVariable))


abstract type AbstractOptimParameters end

struct MutationParameters{T<:Number} <: AbstractOptimParameters
    ηm::T
    pm::T
    p_change_length::T
end

struct OptimParameters{T<:Number} <: AbstractOptimParameters
    pop_size::Integer
    ηm::T
    pm::T
    p_change_length::T
    ηc::T
    pc::T
    window::Union{Nothing,Integer}
    helper::Symbol
    n_main_obj_elitism::Integer
end