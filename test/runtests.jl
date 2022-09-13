using VKMS
using Test

@testset "Structure" begin
    p = Param(0.,-1,1)
    @test p.val == 0.
    @test p.lb == -1.
    @test p.ub == 1.
    @test typeof(p.lb) <: AbstractFloat

end

@testset "Fit" begin
    state = OptimParameters(100, 0.05, 0.1, 2, :less)
    functional(x::AbstractVector,f) = @. 2f(x) + x + 1
    functional(x::AbstractVector,m::KnotModel) = functional(x,model_function_factory(m)) .+ m.m.val .+ m.b.val
    target_function = sin
    x = -10:0.2:10
    y = functional(x,target_function)

    fitness = fitness_factory(functional,x,y)

    pop = random_population(10,(minimum(x),maximum(x)),(-1.5,1.5),(-1.,1.),(-1.,1.),state.pop_size)
    @test all([all(p .== (10,)) for p in get_n_metavariables.(pop)])

    final_pop, gen = evolve(pop, fitness, state, max_gen = 2000, info_every=10)
    
    @test typeof(pop) == typeof(final_pop)
    @test length(pop) == length(final_pop) == state.pop_size
end