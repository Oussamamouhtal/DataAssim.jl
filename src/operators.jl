Pkg.add("RegularizedProblems")  # définit un type qui nous sera utile

using LinearAlgebra, Random, Statistics, RegularizedProblems
using SparseArrays

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
struct ObsOperator{T}
    sigmaR::T
    space_inds::Vector{Int}
    n::Int
    time_inds::Vector{Int}
    nt::Int
    m::Int
    model::Lorenz95Model
end

"""
Génère des observations bruitées à partir d'un état de vérité xt.
"""
function generate_obs(op::ObsOperator, xt::AbstractVector{T}) where T
    x = copy(xt)
    y = Vector{Vector{T}}()
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
function hop(op::ObsOperator, x::AbstractVector{T}) where T
    return x[op.space_inds]
end

""" Modèle tangent de l'opérateur d'observation (H appliqué à dx). """
function tlm_hop(op::ObsOperator, dx::AbstractVector{T}) where T
    return dx[op.space_inds]
end

""" Modèle adjoint de l'opérateur d'observation (H^T appliqué à ay). """

function adj_hop(op::ObsOperator, ay::AbstractVector{T}) where T
    ax = zeros(T, op.n)
    ax[op.space_inds] .= ay
    return ax
end
"""
Calcule le résidu d'observation : y - H(x), en 3D ou 4D.
"""
function misfit(op::ObsOperator, y::AbstractVector{T}, xt::AbstractVector{T}) where T
    x = copy(xt)
    l = length(op.time_inds)
    nx = length(op.space_inds)
    if l == 0
        return y .- hop(op, x)
    else
        yy = reshape(copy(y), (nx, l))
        d = Vector{Vector{T}}()
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

function gop(op::ObsOperator, xt::AbstractVector{T}) where T
    x = copy(xt)
    gx = Vector{Vector{T}}()
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

function tlm_gop(op::ObsOperator, xt::AbstractVector{T}, dxt::AbstractVector{T}) where T
    x = copy(xt)
    dx = copy(dxt)
    dgx = Vector{Vector{T}}()
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

function adj_gop(op::ObsOperator, xt::AbstractVector{T}, axt::AbstractVector{T}) where T
    x = copy(xt)
    l = length(op.time_inds)
    nx = length(op.space_inds)
    agx = reshape(copy(axt), (nx, l))
    ax = [zeros(length(xt)) for _ in 1:(op.time_inds[end]+1)]
    counter = 0

    # Forward pass: stocke la trajectoire complète
    traj_xx = Vector{typeof(x)}()
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
    sigmaR
end

function invdot(R::RMatrix, d::AbstractVector) 
    σ = convert(eltype(d), R.sigmaR)
    return d ./ (σ^2)
end

function sqrtinvdot(R::RMatrix, d::AbstractVector)
    return d ./ R.sigmaR
end

#mutable struct BMatrix
#   sigmaB
#    n
#end
using LinearAlgebra
using SparseArrays

# ============================================================
# B MATRIX
# ============================================================

mutable struct BMatrix{T,S}
    sigmaB::T
    n::Int
    kind::String
    D::Int
    M::Int
    h::T
    Tmat::SparseMatrixCSC{T,Int}
    solver::S
    normalization::T
end

# ============================================================
# CONSTRUCTOR
# ============================================================

function BMatrix(
    n::Int,
    sigmaB;
    kind::String="diagonal",
    D=5,
    M=4
)

    T = typeof(float(sigmaB))

    # --------------------------------------------------------
    # diffusion with D=0 -> diagonal
    # --------------------------------------------------------

    if kind == "diffusion" && D == 0
        kind = "diagonal"
    end

    h = one(T)

    # --------------------------------------------------------
    # default values
    # --------------------------------------------------------

    Tmat = spdiagm(0 => ones(T, n))

    solver = factorize(Tmat)

    normalization = one(T)

    # --------------------------------------------------------
    # diffusion covariance
    # --------------------------------------------------------

    if kind == "diffusion"

        @assert D > 0
        @assert M >= 2
        @assert iseven(M)

        l = D / sqrt(2M - 3)

        α = (l / h)^2

        main_diag = (1 + 2α) .* ones(T, n)
        off_diag  = (-α) .* ones(T, n-1)

        Tmat = spdiagm(
            -1 => off_diag,
             0 => main_diag,
             1 => off_diag
        )

        # periodic BCs
        Tmat[1,n] = -α
        Tmat[n,1] = -α

        Tmat = sparse(Tmat)

        # prefactorization
        solver = factorize(Tmat)

        # ----------------------------------------------------
        # normalization
        # ----------------------------------------------------

        dirac = zeros(T, n)
        dirac[n ÷ 2] = one(T)

        tmp = copy(dirac)

        for _ in 1:(M ÷ 2)
            tmp = solver \ tmp
        end

        tmp = tmp / h

        for _ in 1:(M ÷ 2)
            tmp = solver \ tmp
        end

        normalization = inv(sqrt(maximum(tmp)))
    end

    return BMatrix(
        sigmaB,
        n,
        kind,
        D,
        M,
        h,
        Tmat,
        solver,
        normalization
    )
end

# ============================================================
# B^{-1} x
# ============================================================

function invdot(B::BMatrix, x::AbstractVector)

    y = x / B.sigmaB

    if B.kind == "diffusion"

        y = y / B.normalization

        # apply T^{M/2}
        for _ in 1:(B.M ÷ 2)
            y = B.Tmat * y
        end

        y = y * B.h

        # apply T^{M/2}
        for _ in 1:(B.M ÷ 2)
            y = B.Tmat * y
        end

        y = y / B.normalization
    end

    y = y / B.sigmaB

    return y
end

# ============================================================
# B^{-1/2} x
# ============================================================

function sqrtinvdot(B::BMatrix, x::AbstractVector)

    y = x / B.sigmaB

    if B.kind == "diffusion"

        y = y / sqrt(B.normalization)

        for _ in 1:(B.M ÷ 2)
            y = B.Tmat * y
        end

        y = y * sqrt(B.h)

        y = y / sqrt(B.normalization)
    end

    return y
end

# ============================================================
# B x
# ============================================================

function Bdot(B::BMatrix, x::AbstractVector)

    y = x * B.sigmaB

    if B.kind == "diffusion"

        y = y * B.normalization

        # apply T^{-M/2}
        for _ in 1:(B.M ÷ 2)
            y = B.solver \ y
        end

        y = y / B.h

        # apply T^{-M/2}
        for _ in 1:(B.M ÷ 2)
            y = B.solver \ y
        end

        y = y * B.normalization
    end

    y = y * B.sigmaB

    return y
end

# ============================================================
# B^{1/2} x
# ============================================================

function sqrtdot(B::BMatrix, x::AbstractVector)

    y = x

    if B.kind == "diffusion"

        # inverse sqrt(h)
        y = y / sqrt(B.h)

        # apply T^{-M/2}
        for _ in 1:(B.M ÷ 2)
            y = B.solver \ y
        end

        # normalization
        y = y * B.normalization
    end

    # apply sigmaB
    y = y * B.sigmaB

    return y
end
#function invdot(B::BMatrix, x::AbstractVector) 
#    return x ./ (eltype(x)(B.sigmaB)^2)
#end

#function sqrtinvdot(B::BMatrix, x::AbstractVector)
#    return x ./ B.sigmaB
#end

#function Bdot(B::BMatrix, x::AbstractVector)
#    return x .* (B.sigmaB^2)
#end


# Hessienne 4DVAR
struct Hessian4DVar{T}
    obs::ObsOperator{T}
    R::RMatrix
    B::BMatrix
    xt::AbstractVector{T}
end

function  LinearAlgebra.mul!(y, H::Hessian4DVar{T}, dx::AbstractVector{T}) where T
    w = invdot(H.R, tlm_gop(H.obs, H.xt, dx))
    gtrinv_dx = adj_gop(H.obs, H.xt, w)
    binv_dx = invdot(H.B, dx)
    y .= binv_dx .+ gtrinv_dx
    return y
end

Base.size(H::Hessian4DVar) = (n, n)
Base.eltype(::Hessian4DVar) = T

# Jacobian 4DVAR
struct Jacobian4DVar{T}
    obs::ObsOperator
    R::RMatrix
    B::BMatrix
    xt::AbstractVector{T}
end


function  LinearAlgebra.mul!(y, J::Jacobian4DVar{T}, dx::AbstractVector{T}) where T
        v = sqrtinvdot(J.R, tlm_gop(J.obs, J.xt, dx))
        w = sqrtinvdot(J.B, dx)
        y[1:length(v)] .= v
        y[length(v)+1:end] .= w
    return y
end

Base.size(J::Jacobian4DVar) = (J.obs.n+J.obs.m, J.obs.n)
Base.eltype(J::Jacobian4DVar) = Float64

# Wrapper pour l’adjoint
struct Jacobian4DVarAdj{T}
    J::Jacobian4DVar{T}
end

LinearAlgebra.adjoint(J::Jacobian4DVar{T}) where T = Jacobian4DVarAdj(J)

# Multiplication par l’adjoint : y ← J' * z
function LinearAlgebra.mul!(y::AbstractVector{T}, JT::Jacobian4DVarAdj{T}, z::AbstractVector{T}) where T
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
Base.eltype(::Jacobian4DVarAdj) = T


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
Base.eltype(::Prec) = T


function build_model(obs, R, B, xb, y)

    n = length(xb)
    m = obs.m

    function resid!(r, x)
        gx = gop(obs, x)
        mis =  gx .- y

        r1 = sqrtinvdot(R, mis)        # longueur m
        r2 = sqrtinvdot(B, x .- xb)   # longueur n

        @assert length(r) == m + n
        r[1:m] .= r1
        r[m+1:end] .= r2

        return r
    end


    function jacv!(Jv, x, v)
        j1 = sqrtinvdot(R, tlm_gop(obs, x, v))  # m
        j2 = sqrtinvdot(B, v)                    # n

        @assert length(Jv) == m + n
        Jv[1:m] .= j1
        Jv[m+1:end] .= j2
        return Jv
    end

    function jactv!(Jtv, x, v)
        v1 = v[1:m]
        v2 = v[m+1:end]

        t1 = adj_gop(obs, x, sqrtinvdot(R, v1))  # n
        t2 = sqrtinvdot(B, v2)                    # n

        @assert length(Jtv) == n
        Jtv .= t1 .+ t2
        return Jtv
    end


    x0 = copy(xb)

    model = FirstOrderNLSModel(
    resid!,
    jacv!,
    jactv!,
    n+m,
    x0
)

    return model
end