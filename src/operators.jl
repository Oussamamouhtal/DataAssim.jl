using LinearAlgebra, Random, Statistics

export ObsOperator

export Hessian4DVar
struct ObsOperator
    sigmaR::Float64
    space_inds::Vector{Int}
    n::Int
    time_inds::Vector{Int}
    nt::Int
    model::Lorenz95Model
end

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

function hop(op::ObsOperator, x::Vector{Float64})
    return x[op.space_inds]
end

function tlm_hop(op::ObsOperator, dx::Vector{Float64})
    return dx[op.space_inds]
end

function adj_hop(op::ObsOperator, ay::Vector{Float64})
    ax = zeros(op.n)
    ax[op.space_inds] .= ay
    return ax
end

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



# R Matrix
struct RMatrix
    sigmaR::Float64
end

function invdot(R::RMatrix, d::Vector{Float64})
    return d ./ (R.sigmaR^2)
end

mutable struct BMatrix
    sigmaB::Float64
    n::Int
end

function invdot(B::BMatrix, x::Vector{Float64})
    return x ./ (B.sigmaB^2)
end

function Bdot(B::BMatrix, x::Vector{Float64})
    return x .* (B.sigmaB^2)
end


# Hessienne 3DVAR
struct Hessian3DVar
    obs::ObsOperator
    R::RMatrix
    B::BMatrix
end

function Amul!(H::Hessian3DVar, dx::Vector{Float64})
    w = invdot(H.R, tlm_hop(H.obs, dx))
    htrinvh_dx = adj_hop(H.obs, w)
    binv_dx = invdot(H.B, dx)
    return binv_dx .+ htrinvh_dx
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

