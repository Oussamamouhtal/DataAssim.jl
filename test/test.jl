using Pkg
using Test

#Pkg.add("LinearOperators")
include("../src/lorenz95.jl")
include("../src/operators.jl")

T = Float64
n = 40
nt = 10
dt = 0.01
F = 8.0
model = Lorenz95Model(F, dt)      # Modèle de Lrenz95

# Pour les tests de l'observation
sigmaR = 0.5
total_space_obs = 10
total_time_obs = 5
space_inds_obs = round.(Int, range(1, n; length=total_space_obs))
time_inds_obs = round.(Int, range(1, nt-1; length=total_time_obs))
m =  total_space_obs*total_time_obs         
obs = ObsOperator(sigmaR, space_inds_obs, n, time_inds_obs, nt, m, 
model)          # L'opérateur d'observation H 

@testset "TLM-ADJ (Model : M)" begin
    x = randn(n)
    dx = randn(n)
    a = randn(n)

    y1 = tlm_traj(model, x, dx, nt)     # Le jacobien du modèle au point x : J_M(x)
    y2 = ad_traj(model, x, a, nt)       # Le transposé du jacobien au point x : J_M(x)^T   

    inner1 = dot(y1, a)                 # < J_M(x) dx, a >
    inner2 = dot(dx, y2)                # < dx, J_M(x)^T a >

    @test inner1 ≈ inner2 atol=1e-7     # < J_M(x) dx, a > = < dx, J_M(x)^T a >
end

@testset "TLM vs Finite Differences (Model : M)" begin
    x = randn(n)
    dx = randn(n)
    ϵ = 1e-6

    fx = traj(model, x , nt)
    fx_perturbed = traj(model, x .+ ϵ .* dx, nt)
    fd_approx = (fx_perturbed .- fx) ./ ϵ

    tlm_val = tlm_traj(model, x, dx, nt)

    @test fd_approx ≈ tlm_val atol=10*ϵ
end

@testset "TLM-ADJ Consistency (Opérateur d'observation H)" begin
    x = randn(n)
    dx = randn(n)
    ay = randn(length(space_inds_obs))

    y1 = tlm_hop(obs, dx)           # J_H(x) = J_H (L'opérateur d'observation H est linéaire)
    y2 = adj_hop(obs, ay)           # J_H^T

    inner1 = dot(y1, ay)            # < J_H*dx, ay >
    inner2 = dot(dx, y2)            # < dx, J_H^T*ay >

   @test inner1 ≈ inner2 atol=1e-7  # < J_H*dx, ay > = < dx, J_H^T*ay >
end

@testset "TLM vs Finite Differences (Opérateur d'observation : H)" begin
    x = randn(n)
    dx = randn(n)
    ϵ = 1e-6

    fx = hop(obs, x)
    fx_perturbed = hop(obs, x .+ ϵ .* dx)
    fd_approx = (fx_perturbed .- fx) ./ ϵ

    tlm_val = tlm_hop(obs, dx)

    @test fd_approx ≈ tlm_val atol=10*ϵ
end

@testset "TLM vs Finite Differences (G = H o M Composition du model et Obs)" begin
    x = randn(n)
    dx = randn(n)
    ϵ = 1e-6
    

    fx = gop(obs, x)
    fx_perturbed = gop(obs, x .+ ϵ .* dx)
    fd_approx = (fx_perturbed .- fx) ./ ϵ

    tlm_val = tlm_gop(obs, x, dx)

    @test fd_approx ≈ tlm_val atol=10*ϵ
end

@testset "gop: TLM-ADJ Consistency (G operator)" begin
    xt = randn(n)
    dx = randn(length(xt))
    a = randn(length(space_inds_obs) * length(time_inds_obs))

    y1 = tlm_gop(obs, xt, dx)
    y2 = adj_gop(obs, xt, a)

    inner1 = dot(y1, a)
    inner2 = dot(dx, y2)

    @test inner1 ≈ inner2 atol=1e-7
end