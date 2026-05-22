using Pkg

Pkg.activate("mnist")
Pkg.add("MLDatasets")
Pkg.add("Images")
Pkg.add("DataFrames")
using MLDatasets
using Images
using Random
using NLPModels, ADNLPModels
using LinearAlgebra
using SparseArrays
using JSOSolvers
using DataFrames
using PrettyTables

Pkg.develop(path="../Krylov.jl")   # change path to local Krylov fork
Pkg.develop(path="../JSOSolvers.jl") 

F = MLDatasets.MNIST(split = :train)
A = F.features  # images d'entrée
b = F.targets   # labels
println("$(length(b)) images")


# extraction des parties de A et b pertinentes
digits = (1, 7)
index_digits = findall(x -> x ∈ digits, b)

#n_samples = 6000  # taille  souhaitée
#idx_sub = index_digits[randperm(length(index_digits))[1:n_samples]]
b_digits = b[index_digits]
A_digits = A[:, :, index_digits]

# définition des deux classes
b_digits[b_digits .== digits[1]] .= 1
b_digits[b_digits .== digits[2]] .= -1

# reformulation de A sous forme d'une matrice
# chaque colonne du nouveau A_digits est la vectorisation d'une des images,
# i.e., l'empilement de ses colonnes les unes par-dessus les autres
A_digits = reshape(A_digits, size(A_digits, 1) * size(A_digits, 2), size(A_digits, 3))
A_digits = convert(Matrix{Float64}, A_digits) ./ 255

size(A_digits)


Ahat = Diagonal(b_digits) * sparse(A_digits)'

function f(x)
    r = Ahat * x               
    y = 100.0 .* (1 .- tanh.(r))
    return 0.5 * dot(y, y)
end
   
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

x0 = ones(size(A_digits, 1))
nlp = ADNLPModel(f, x0, backend = :optimized)
results = DataFrame()

push!(results, run_solver(nlp, :cg))
push!(results, run_solver(nlp, :lbfgs, memory=100))
push!(results, run_solver(nlp, :diom, memory=100))
push!(results, run_solver(nlp, :lbfgs, memory=50))
push!(results, run_solver(nlp, :diom, memory=50))

pretty_table(results)