using VKMS
using Statistics: mean
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
    x = -10:0.2:10 |> collect
    y = functional(x,target_function)

    fitness = LessFitness{2}(functional,x,y,ones(length(x)),5)

    n_knots=10
    xbounds = vcat((minimum(x),minimum(x)),fill((minimum(x),maximum(x)),n_knots-2),(maximum(x),maximum(x)))
    ybounds = fill((-1.5,1.5),n_knots)
    pop = random_population(xbounds,ybounds,(-1.,1.),(-1.,1.),state.pop_size)
    @test count(!=(2), [length(filter(v -> v.x.lb == v.x.ub, p.knots.metavariables)) for p in pop]) == 0
    @test all([all(p .== (10,)) for p in get_n_metavariables.(pop)])

    initial_ssr = mean(begin
        residual = target_function.(x) .- model_function_factory(p).(x)
        residual' * residual
    end for p in pop)

    final_pop, gen = evolve(pop, fitness, state, max_gen = 2000)
    
    @test typeof(pop) == typeof(final_pop)
    @test length(pop) == length(final_pop) == state.pop_size
    @test count(!=(2), [length(filter(v -> v.x.lb == v.x.ub, p.knots.metavariables)) for p in final_pop]) == 0

    final_ssr = mean(begin
        residual = target_function.(x) .- model_function_factory(p).(x)
        residual' * residual
    end for p in final_pop)

    @test initial_ssr > final_ssr 
    println("Mean initial ssr: $(initial_ssr)")
    println("Mean final ssr: $(final_ssr)")

end
