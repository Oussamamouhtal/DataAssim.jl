using LinearAlgebra, Random, Statistics

export ObsOperator

export Hessian4DVar

""" 
Structure représentant l'opérateur d'observation pour 3D-Var ou 4D-Var.

# Champs
- `sigmaR` : écart-type du bruit d'observation (diagonale de R)
- `space_inds` : indices des variables d'état observées (spatialement)
- `n` : taille de l'état total
- `time_inds` : indices temporels des observations (pour 4D-Var)
- `nt` : nombre total de pas de temps dans la trajectoire
- `model` : le modèle de dynamique (ex. Lorenz95Model)
"""
struct ObsOperator
    sigmaR::Float64
    space_inds::Vector{Int}
    n::Int
    time_inds::Vector{Int}
    nt::Int
    m ::Int
    model::Lorenz95Model
end

"""
Génère des observations bruitées à partir d'un état de vérité xt.
"""
function generate_obs(op::ObsOperator, xt::Vector{Float64})
    x = copy(xt)
    y = Vector{Vector{Float64}}()
    if op.nt > 0
        # 4DVAR
        counter = 0
        for ii in 0:op.nt-1
            if ii in op.time_inds
                new_obs = hop(op, x) .+ op.sigmaR .* randn(length(op.space_inds))
                push!(y, new_obs)
                counter += 1
            end
            counter == length(op.time_inds) && break
            x = traj(op.model,x, 1)
        end
        return reduce(vcat, y)
    else
        # 3DVAR
        return hop(op, x) .+ op.sigmaR .* randn(length(op.space_inds))
    end
end

""" Extraction de l'observation (opérateur H appliqué à x). """
function hop(op::ObsOperator, x::Vector{Float64})
    return x[op.space_inds]
end

""" Modèle tangent de l'opérateur d'observation (H appliqué à dx). """
function tlm_hop(op::ObsOperator, dx::Vector{Float64})
    return dx[op.space_inds]
end

""" Modèle adjoint de l'opérateur d'observation (H^T appliqué à ay). """

function adj_hop(op::ObsOperator, ay::Vector{Float64})
    ax = zeros(op.n)
    ax[op.space_inds] .= ay
    return ax
end
"""
Calcule le résidu d'observation : y - H(x), en 3D ou 4D.
"""
function misfit(op::ObsOperator, y::Vector{Float64}, xt::Vector{Float64})
    x = copy(xt)
    l = length(op.time_inds)
    nx = length(op.space_inds)
    if l == 0
        return y .- hop(op, x)
    else
        yy = reshape(copy(y), (nx, l))
        d = Vector{Vector{Float64}}()
        counter = 0
        for ii in 0:op.nt-1
            if ii in op.time_inds
                push!(d, yy[:, counter+1] .- hop(op, x))
                counter += 1
            end
            counter == l && break
            x = traj(op.model,x, 1)
        end
        return reduce(vcat, d)
    end
end

""" Applique l'opérateur G(x) = H o M(x) avec M model dynamique (traj)"""

function gop(op::ObsOperator, xt::Vector{Float64})
    x = copy(xt)
    gx = Vector{Vector{Float64}}()
    counter = 0
    for ii in 0:op.nt-1
        if ii in op.time_inds
            push!(gx, hop(op, x))
            counter += 1
        end
        counter == length(op.time_inds) && break
        x = traj(op.model,x, 1)
    end
    return reduce(vcat, gx)
end

function tlm_gop(op::ObsOperator, xt::Vector{Float64}, dxt::Vector{Float64})
    x = copy(xt)
    dx = copy(dxt)
    dgx = Vector{Vector{Float64}}()
    counter = 0

    for ii in 0:op.nt-1
        if ii in op.time_inds
            push!(dgx, tlm_hop(op, dx))
            counter += 1
        end
        counter == length(op.time_inds) && break
        dx = tlm_traj(op.model, x, dx, 1)
        x = traj(op.model, x, 1)
    end

    return reduce(vcat, dgx)
end

function adj_gop(op::ObsOperator, xt::Vector{Float64}, axt::Vector{Float64})
    x = copy(xt)
    l = length(op.time_inds)
    nx = length(op.space_inds)
    agx = reshape(copy(axt), (nx, l))
    ax = [zeros(length(xt)) for _ in 1:(op.time_inds[end]+1)]
    counter = 0

    # Forward pass: stocke la trajectoire complète
    traj_xx = Vector{Vector{Float64}}()
    for _ in 1:op.nt
        push!(traj_xx, copy(x))
        x = traj(op.model, x, 1)
    end

    # Backward pass
    for ii in reverse(0:op.time_inds[end])
        if ii in reverse(op.time_inds)
            idx = findfirst(x -> x == ii, op.time_inds)
            ax[ii+1] .+= adj_hop(op, agx[:, idx])
            counter += 1
        end
        counter == l+1 && break
        if ii > 0
            ax[ii] = ad_traj(op.model, traj_xx[ii], ax[ii+1], 1)
        end
    end
    return ax[1]
end


# -------------------------------------------------------------------------
# Matrices d’erreur de covariance d’observation et de background
# -------------------------------------------------------------------------

""" Structure représentant R, la matrice de covariance des observations. """
struct RMatrix
    sigmaR::Float64
end

""" Applique R⁻¹ à un vecteur. """
function invdot(R::RMatrix, d::Vector{Float64})
    return d ./ (R.sigmaR^2)
end

function sqrtinvdot(R::RMatrix, d::Vector{Float64})
    return d ./ (R.sigmaR)
end

""" Structure représentant la matrice de covariance B. """

mutable struct BMatrix
    sigmaB::Float64
    n::Int
end

function invdot(B::BMatrix, x::Vector{Float64})
    return x ./ (B.sigmaB^2)
end

function sqrtinvdot(B::BMatrix, x::Vector{Float64})
    return x ./ (B.sigmaB)
end

function Bdot(B::BMatrix, x::Vector{Float64})
    return x .* (B.sigmaB^2)
end


# Hessienne 4DVAR
struct Hessian4DVar
    obs::ObsOperator
    R::RMatrix
    B::BMatrix
    xt::Vector{Float64}
end

function  LinearAlgebra.mul!(y, H::Hessian4DVar, dx::Vector{Float64})
    w = invdot(H.R, tlm_gop(H.obs, H.xt, dx))
    gtrinv_dx = adj_gop(H.obs, H.xt, w)
    binv_dx = invdot(H.B, dx)
    y .= binv_dx .+ gtrinv_dx
    return y
end

Base.size(H::Hessian4DVar) = (n, n)
Base.eltype(::Hessian4DVar) = Float64

# Jacobian 4DVAR
struct Jacobian4DVar
    obs::ObsOperator
    R::RMatrix
    B::BMatrix
    xt::Vector{Float64}
end


function  LinearAlgebra.mul!(y, J::Jacobian4DVar, dx::Vector{Float64})
        v = sqrtinvdot(J.R, tlm_gop(J.obs, J.xt, dx))
        w = sqrtinvdot(J.B, dx)
        y .= [v; w]
    return y
end

Base.size(J::Jacobian4DVar) = (J.obs.n+J.obs.m, J.obs.n)
Base.eltype(J::Jacobian4DVar) = Float64

# Wrapper pour l’adjoint
struct Jacobian4DVarAdj
    J::Jacobian4DVar
end

LinearAlgebra.adjoint(J::Jacobian4DVar) = Jacobian4DVarAdj(J)

# Multiplication par l’adjoint : y ← J' * z
function LinearAlgebra.mul!(y::AbstractVector, JT::Jacobian4DVarAdj, z::AbstractVector)
    J = JT.J
    m, n = J.obs.m, J.obs.n
    @assert length(z) == m + n
    @assert length(y) == n

    z1 = z[1:m]       # partie "observation"
    z2 = z[m+1:end]   # partie "background"

    # Appliquer R^{-1/2} et B^{-1/2}
    rhalf_z1 = sqrtinvdot(J.R, z1)
    bhalf_z2 = sqrtinvdot(J.B, z2)

    t = adj_gop(J.obs, J.xt, rhalf_z1)

    # Somme des deux contributions
    y .= t .+ bhalf_z2
    return y
end

Base.size(JT::Jacobian4DVarAdj) = (JT.J.obs.n, JT.J.obs.m + JT.J.obs.n)
Base.eltype(::Jacobian4DVarAdj) = Float64


"""
    revd(A::AbstractMatrix, k; p=10, q=0)

Randomized EVD pour matrice symétrique A (n×n).
Retourne (U, λ, λmin, resnorms) avec les k plus grandes valeurs propres.
Toutes les multiplications par A sont réalisées en boucle via `mul!`.
"""
function revd(A, k::Integer; p::Integer=10, q::Integer=0)
    n, m = size(A)
    @assert n == m "A doit être carrée n×n"
    ℓ = k + p

    # 1) Matrice aléatoire Ω (n×ℓ)
    Ω = randn(n, ℓ)

    # 2) Y = A * Ω 
    Y   = Matrix{eltype(A)}(undef, n, ℓ)
    Q   = Matrix{eltype(A)}(undef, n, ℓ)
    tmp = similar(Ω, n)  # vecteur de travail de longueur n
    for j in 1:ℓ
        mul!(tmp, A, Ω[:,j])      # tmp = A * Ω[:,j]
        copyto!(view(Y, :, j), tmp)
    end

    # 3) QR 
    F = qr(Y)
    Q = Matrix(F.Q)                      # n×ℓ

    # 4) Itérations de puissance 
    for _ in 1:q
        for j in 1:ℓ
            mul!(tmp, A, Q[:,j])  # tmp = A * Q[:,j]
            copyto!(view(Y, :, j), tmp)
        end
        F = qr(Y)
        Q = Matrix(F.Q)
    end

    # 5) AQ = A * Q 
    AQ = Matrix{eltype(A)}(undef, n, ℓ)
    for j in 1:ℓ
        mul!(tmp, A, Q[:,j])
        copyto!(view(AQ, :, j), tmp)
    end

    # 6) Petite matrice de Rayleigh et EVD
    T = Symmetric(Q' * AQ)               # ℓ×ℓ
    E = eigen(T)                         # valeurs triées ↑
    λ = E.values
    W = E.vectors
    idx = (ℓ-k+1):ℓ                      # indices des k plus grandes
    λk = λ[idx]
    Uk = Q * W[:, idx]                   # U ≈ Q * W_k
    # 7) Résidus ‖A*Uk[:,i] - λk[i]*Uk[:,i]‖₂ (en boucle)
    resnorms = similar(λk)
    for i in 1:k
        mul!(tmp, A, Uk[ :, i])     # tmp = A * u_i
        resnorms[i] = norm(tmp .- λk[i] .* view(Uk, :, i))
    end

    return Uk, λk, minimum(λk), resnorms
end




struct Prec
    S::Matrix{Float64}
    Λ::Vector{Float64} 
    θ::Float64   
end

"""
P = In + U(tetas-1/2 - Il)Ut
"""
function LinearAlgebra.mul!(y::AbstractVector, P::Prec, x::AbstractVector)
    t = P.S' * x 
    t .= (sqrt(P.θ)./sqrt.(P.Λ) .- 1) .* t
    t = P.S * t
    y .= t .+ x  
    return y
end

Base.size(P::Prec) = (size(P.S, 1), size(P.S, 1))
Base.eltype(::Prec) = Float64
