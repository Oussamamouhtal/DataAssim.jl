

using LinearAlgebra

export Lorenz95Model

struct Lorenz95Model
    F
    dt
end

function l95(model::Lorenz95Model, xx::AbstractVector)
    n = length(xx)
    dxdt = zeros(eltype(xx), n)
    for i in 1:n
        im2 = i - 2
        im1 = i - 1
        ip1 = i + 1

        if im2 < 1
            im2 += n
        end
        if im1 < 1
            im1 += n
        end
        if ip1 > n
            ip1 -= n
        end
        dxdt[i] = (xx[ip1] - xx[im2]) * xx[im1] - xx[i] + model.F
    end
    return dxdt
end

function tlm_l95(model::Lorenz95Model, x::AbstractVector, dx::AbstractVector) 
    n = length(x)
    ddxdt = zeros(eltype(x), n)

    for i in 1:n
        im2 = i - 2
        im1 = i - 1
        ip1 = i + 1

        if im2 < 1
            im2 += n
        end
        if im1 < 1
            im1 += n
        end
        if ip1 > n
            ip1 -= n
        end

        ddxdt[i] = (dx[ip1] - dx[im2]) * x[im1] + (x[ip1] - x[im2]) * dx[im1] - dx[i]
    end

    return ddxdt
end


function ad_l95(model::Lorenz95Model, x::AbstractVector, ax::AbstractVector)
    n = length(x)
    adxdt = zeros(eltype(x), n)

    for i in 1:n
        im2 = i - 2
        im1 = i - 1
        ip1 = i + 1

        if im2 < 1
            im2 += n
        end
        if im1 < 1
            im1 += n
        end
        if ip1 > n
            ip1 -= n
        end

        adxdt[im2] -= x[im1] * ax[i]
        adxdt[im1] += x[ip1] * ax[i]
        adxdt[im1] -= x[im2] * ax[i]
        adxdt[ip1] += x[im1] * ax[i]
        adxdt[i]    -= ax[i]
    end

    return adxdt
end


function RKstep(model::Lorenz95Model, xx::AbstractVector) 
    dt = model.dt
    k1 = l95(model, xx)
    k2 = l95(model, xx .+ (dt/2) .* k1)
    k3 = l95(model, xx .+ (dt/2) .* k2)
    k4 = l95(model, xx .+ dt .* k3)
    return xx .+ (dt/6) .* (k1 .+ 2*k2 .+ 2*k3 .+ k4)
end

function dRKstep(model::Lorenz95Model, xx::AbstractVector, dx::AbstractVector) 
    dt = model.dt
    k1 = l95(model, xx)
    dk1 = tlm_l95(model, xx, dx)
    k2 = l95(model, xx .+ (dt/2) .* k1)
    dk2 = tlm_l95(model, xx .+ (dt/2) .* k1, dx .+ (dt/2) .* dk1)
    k3 = l95(model, xx .+ (dt/2) .* k2)
    dk3 = tlm_l95(model, xx .+ (dt/2) .* k2, dx .+ (dt/2) .* dk2)
    dk4 = tlm_l95(model, xx .+ dt .* k3, dx .+ dt .* dk3)
    return dx .+ (dt/6) .* (dk1 .+ 2*dk2 .+ 2*dk3 .+ dk4)
end

function aRKstep(model::Lorenz95Model, xx::AbstractVector, axp::AbstractVector)
    dt = model.dt
    x0 = copy(xx)
    k1 = l95(model, x0)
    x1 = x0 .+ (dt/2) .* k1
    k2 = l95(model, x1)
    x2 = x0 .+ (dt/2) .* k2
    k3 = l95(model, x2)
    x3 = x0 .+ dt .* k3

    ak1 = zeros(eltype(xx), length(xx))
    ak2 = zeros(eltype(xx), length(xx))
    ak3 = zeros(eltype(xx), length(xx))
    ak4 = zeros(eltype(xx), length(xx))
    ax  = zeros(eltype(xx), length(xx))

    jak2 = zeros(eltype(xx), length(xx))
    jak3 = zeros(eltype(xx), length(xx))
    jak4 = zeros(eltype(xx), length(xx))


    ak4 = ak4 .+ (dt/6) .* axp
    ak3 = ak3 .+ (dt/3) .* axp
    ak2 = ak2 .+ (dt/3) .* axp
    ak1 = ak1 .+ (dt/6) .* axp
    ax  = ax  .+ axp

    jak4 = ad_l95(model, x3, ak4)
    ak3 = ak3 .+ dt .* jak4
    ax  = ax  .+ jak4

    jak3 = ad_l95(model, x2, ak3)
    ak2 = ak2 .+ (dt/2) .* jak3
    ax  = ax  .+ jak3

    jak2 = ad_l95(model, x1, ak2)
    ak1 = ak1 .+ (dt/2) .* jak2
    ax  = ax  .+ jak2

    ax = ax .+ ad_l95(model, x0, ak1)

    return ax
end

function traj(model::Lorenz95Model, x::AbstractVector, nt::Int) 
    xx = copy(x)
    for _ in 1:abs(nt)
        xx = RKstep(model, xx)
    end
    return xx
end

function tlm_traj(model::Lorenz95Model, x::AbstractVector, dx::AbstractVector, nt::Int) 
    xx = copy(x)
    dxx = copy(dx)
    for _ in 1:abs(nt)
        dxx = dRKstep(model, xx, dxx)
        xx = RKstep(model, xx)
    end
    return dxx
end

function ad_traj(model::Lorenz95Model, x::AbstractVector, ax::AbstractVector, nt::Int) 
    axx = copy(ax)
    xx = copy(x)
    traj_xx = Vector{Vector{eltype(x)}}()
    for _ in 1:abs(nt)
        push!(traj_xx, xx)
        xx = RKstep(model, xx)
    end
    for i in 1:abs(nt)
        axx = aRKstep(model, traj_xx[nt - i + 1], axx)
    end
    return axx
end