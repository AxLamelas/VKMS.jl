using VLEvolution
using Test

@testset "Structure" begin
    p = Param(0.,-1,1)
    @test p.val == 0.
    @test p.lb == -1.
    @test p.ub == 1.
    @test typeof(p.lb) <: AbstractFloat

end

@testset "Fit" begin
    
end