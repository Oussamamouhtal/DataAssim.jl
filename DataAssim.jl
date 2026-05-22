using Pkg

Pkg.add("DataFrames")
Pkg.activate("Data_Assimilation")
Pkg.develop(path="../Krylov.jl")   # change path to local Krylov fork
Pkg.develop(path="../JSOSolvers.jl") 
using PrettyTables

Pkg.add("ADNLPModels")

# Inclure tes fichiers locaux
include("../DataAssim.jl/src/lorenz95.jl")
include("../DataAssim.jl/src/operators.jl")

using NLPModels, ADNLPModels
using ForwardDiff
using StaticArrays
using JSOSolvers

using DataFrames
function run_solver(nlp, subsolver; memory=nothing)

    if memory === nothing
        stats = trunk(nlp,
            max_time=10000.0,
            max_iter=500,
            verbose=0,
            subsolver=subsolver
        )
    else
        stats = trunk(nlp,
            max_time=10000.0,
            max_iter=500,
            verbose=0,
            subsolver=subsolver,
            subsolver_kwargs=(memory=memory,)
        )
    end

    row = (
        solver = string(subsolver) * (memory === nothing ? "" : "_m$(memory)"),
        status = stats.status,
        norm_sol = norm(stats.solution),
        objective = stats.objective,
        iter = stats.iter,
        obj_eval = nlp.counters.neval_obj,
        grad_eval = nlp.counters.neval_grad,
        hprod = nlp.counters.neval_hprod,
        time = stats.elapsed_time
    )

    reset!(nlp)

    return row
end

function nonlinear_funcval(x, y, obs, xb, B, R)
    eo = y .- gop(obs, x)
    eb = x .- xb
    fx = eltype(x).(1/2) * eb'* invdot(B, eb) + eltype(x).(1/2) * eo'*invdot(R, eo)
    return fx
end


n = 1000
nt = 10
dt = 0.025
F = 10.0
rng = MersenneTwister(1234)

sigmaR =  0.2
total_space_obs = 500
total_time_obs = 2   # m = total_space_obs*total_time_obs
sigmaB =  0.8

# Model
model = Lorenz95Model(F, dt)

# Observations

space_inds_obs = round.(Int, range(1, n; length=total_space_obs))
time_inds_obs = round.(Int, range(1, nt-1; length=total_time_obs))
m = total_space_obs*total_time_obs
obs = ObsOperator(sigmaR, space_inds_obs, n, time_inds_obs, nt, m, model)
R = RMatrix(sigmaR)

# Background
B = BMatrix(n, sigmaB)

# Spin-up
xt = 3 .* ones(n) #.+ randn(rng, n)
xt = traj(model, xt, 5000)

xb = xt .+ sqrtdot(B, randn(rng, n))

# Observations
y = generate_obs(obs, xt)

# Construct ADNLPModel
f(x) = nonlinear_funcval(x, y, obs, xb, B, R)

x0 = copy(xb)

results = DataFrame()

nlp = ADNLPModel(f, x0, backend = :optimized)
res = run_solver(nlp, :cg)
push!(results, res)
nlp1 = ADNLPModel(f, copy(x0), backend=:optimized)
res1 = run_solver(nlp1, :lbfgs; memory=50)
push!(results, res1)
nlp2 = ADNLPModel(f, copy(x0), backend=:optimized)
res2 = run_solver(nlp2, :lbfgs; memory=100)
push!(results, res2)
nlp3 = ADNLPModel(f, copy(x0), backend=:optimized)
res3 = run_solver(nlp3, :diom; memory=50)
push!(results, res3)
nlp4 = ADNLPModel(f, copy(x0), backend=:optimized)
res4 = run_solver(nlp4, :diom; memory=100)
push!(results, res4)
pretty_table(results)