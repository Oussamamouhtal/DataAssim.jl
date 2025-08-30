using Pkg

Pkg.add("LinearOperators")
Pkg.add("Krylov")
Pkg.add("Plots")
Pkg.add("PrettyTables")

include("lorenz95.jl")
include("operators.jl")

using LinearAlgebra, Random, Krylov, LinearOperators, Plots
using PrettyTables
function fourdvar(n, nt, dt, F , sigmaR, total_space_obs, total_time_obs, sigmaB)
# Model
model = Lorenz95Model(F, dt)

# Observations

space_inds_obs = round.(Int, range(1, n; length=total_space_obs))
time_inds_obs = round.(Int, range(1, nt-1; length=total_time_obs))
m = length(space_inds_obs)*length(time_inds_obs)
obs = ObsOperator(sigmaR, space_inds_obs, n, time_inds_obs, nt, m, model)
R = RMatrix(sigmaR)

# Background
B = BMatrix(sigmaB, n)

# Spin-up
xt = 3.0 .* ones(n) .+ randn(n)
xt = traj(model, xt, 5000)

# Background state
xb = xt .+ randn(n) .* sigmaB

# Observations
y = generate_obs(obs, xt)


function nonlinear_funcval(x, y, obs, xb, B, R)
    eo = y .- gop(obs, x)
    eb = x .- xb
    return eb'* invdot(B, eb) + eo'*invdot(R, eo)
end

function quadratic_funcval(x, dx, y, obs, xb, B, R)
    eo = gop(obs, x) .- y .+ tlm_gop(obs, x, dx)
    eb = x .- xb .+ dx
    return dot(eb, invdot(B, eb)) + dot(eo, invdot(R, eo))
end

xa = copy(xb) 
max_outer = 30
tol_grad = 1e-6

iterates_GN = Dict{String, Vector{Float64}}()

# Initialisation des itérées de Gauss-Newton
for name in ["minres_qlp", "minres", "cg_lanczos", "cg"]
    iterates_GN[name] = copy(xa)
end


row_names = vcat("Itérations GN", "Norme résidu finale", "Coût non linéaire", "Erreur relative à xt")
global stats_GN_table = row_names
for name in ["minres_qlp", "minres", "cg_lanczos", "cg"]
    iterates_GN[name] = copy(xa)
    nb_iter = 0
    b = zeros(n) 
    println("Résolution su problème avec le sous-solveur $name")
    global stats_subsolver = vcat("nombre d'itérations", "résidu initiale", "résidu finale", "temps")
    header_subsolver = [""]
    for iter_outer in 1:max_outer
        d = misfit(obs, y, iterates_GN[name])
        b = invdot(B, xb .- iterates_GN[name]) .+ adj_gop(obs, iterates_GN[name], invdot(R, d))
        A = Hessian4DVar(obs, R, B, iterates_GN[name])
        solver = getproperty(Krylov, Symbol(name))
        dx, stats = solver(A, b, history=true)
        iterates_GN[name] .+= dx
        nb_iter += 1
        if norm(b) < tol_grad
            break
        end
        if iter_outer%5 == 0 || iter_outer == 1
            push!(header_subsolver, "GN iter $iter_outer")
            global stats_subsolver = hcat(stats_subsolver, vcat(stats.niter, stats.residuals[1], stats.residuals[end], stats.timer))
        end
        
    end

    pretty_table(stats_subsolver, header=header_subsolver)

    
    println("statstiques sur GN en utilisons différents sous-solveurs")

    res_final = norm(b)
    fval = nonlinear_funcval(iterates_GN[name], y, obs, xb, B, R)
    err_xt = norm(xt - iterates_GN[name]) / norm(xt)
    global stats_GN_table = hcat(stats_GN_table, vcat(nb_iter, res_final, fval, err_xt))
end


header = ["", "minres_qlp",  "minres", "cg_lanczos", "cg"]
pretty_table(stats_GN_table, header=header)
    return xt, xb, y, iterates_GN, model, time_inds_obs
 
end 

n = 100
nt = 10
dt = 0.025
F = 8.0
sigmaR =  0.1
total_space_obs = 10
total_time_obs = 5
sigmaB = 0.8

xtruth, xback, y, iterates_GN, model, time_inds_obs = fourdvar(n, nt, dt, F, sigmaR, 
total_space_obs, total_time_obs, sigmaB)

x_truth_full = Vector{Vector{Float64}}()
push!(x_truth_full, xtruth)
for _ in 1:nt
    global xtruth = traj(model, xtruth, 1)
    push!(x_truth_full, xtruth)
end

# Trajectoire background

x_b_full = Vector{Vector{Float64}}()
push!(x_b_full, xback)
for _ in 1:nt
    global xback = traj(model, xback, 1)
    push!(x_b_full, xback)
end

trajectories = Dict{String, Vector{Vector{Float64}}}()
# Trajectoire analysée
for name in ["minres_qlp", "minres", "cg_lanczos", "cg"]
    trajectories[name] = Vector{Vector{Float64}}() 
    global xanalysis = copy(iterates_GN[name])
    push!(trajectories[name], xanalysis)
    for _ in 1:nt
        global xanalysis = traj(model, xanalysis, 1)
        push!(trajectories[name], xanalysis)
    end
end

# Observations
y_obs = reshape(y, total_space_obs, total_time_obs)

# Plot d'un état : par exemple la variable n°1 en fonction du temps
idx = 1  # index de la variable à tracer

times = 0:nt
gr() 
plot(times, [x[idx] for x in x_truth_full], label="Truth", lw=2)
plot!(times, [x[idx] for x in x_b_full], label="Background", lw=2, ls=:dash)
for name in ["minres_qlp", "minres", "cg_lanczos", "cg"]
    plot!(times, [x[idx] for x in trajectories[name]], label=name, lw=2)
end
scatter!(time_inds_obs, y_obs[1, :], label="Observations", ms=4, c=:black)
xlabel!("Time step")
ylabel!("State x[$idx]")
savefig("assimilation_result.png")
