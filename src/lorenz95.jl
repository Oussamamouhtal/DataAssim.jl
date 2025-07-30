

using LinearAlgebra

export Lorenz95Model

struct Lorenz95Model
    F::Float64
    dt::Float64
end

function l95(model::Lorenz95Model, xx::Vector{Float64})
    n = length(xx)
    dxdt = zeros(Float64, n)
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

function tlm_l95(model::Lorenz95Model, x::Vector{Float64}, dx::Vector{Float64})
    n = length(x)
    ddxdt = zeros(Float64, n)

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


function ad_l95(model::Lorenz95Model, x::Vector{Float64}, ax::Vector{Float64})
    n = length(x)
    adxdt = zeros(Float64, n)

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


function RKstep(model::Lorenz95Model, xx::Vector{Float64})
    dt = model.dt
    k1 = l95(model, xx)
    k2 = l95(model, xx .+ (dt/2) .* k1)
    k3 = l95(model, xx .+ (dt/2) .* k2)
    k4 = l95(model, xx .+ dt .* k3)
    return xx .+ (dt/6) .* (k1 .+ 2*k2 .+ 2*k3 .+ k4)
end

function dRKstep(model::Lorenz95Model, xx::Vector{Float64}, dx::Vector{Float64})
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

function aRKstep(model::Lorenz95Model, xx::Vector{Float64}, axp::Vector{Float64})
    dt = model.dt
    x0 = copy(xx)
    k1 = l95(model, x0)
    x1 = x0 .+ (dt/2) .* k1
    k2 = l95(model, x1)
    x2 = x0 .+ (dt/2) .* k2
    k3 = l95(model, x2)
    x3 = x0 .+ dt .* k3

    ak1 = zeros(length(xx))
    ak2 = zeros(length(xx))
    ak3 = zeros(length(xx))
    ak4 = zeros(length(xx))
    ax  = zeros(length(xx))

    jak2 = zeros(length(xx))
    jak3 = zeros(length(xx))
    jak4 = zeros(length(xx))

    ak4 .+= (dt/6) .* axp
    ak3 .+= (dt/3) .* axp
    ak2 .+= (dt/3) .* axp
    ak1 .+= (dt/6) .* axp
    ax  .+= axp

    jak4 .= ad_l95(model, x3, ak4)
    ak3 .+= dt .* jak4
    ax  .+= jak4

    jak3 .= ad_l95(model, x2, ak3)
    ak2 .+= (dt/2) .* jak3
    ax  .+= jak3

    jak2 .= ad_l95(model, x1, ak2)
    ak1 .+= (dt/2) .* jak2
    ax  .+= jak2

    ax .+= ad_l95(model, x0, ak1)

    return ax
end

function traj(model::Lorenz95Model, x::Vector{Float64}, nt::Int)
    xx = copy(x)
    for _ in 1:abs(nt)
        xx = RKstep(model, xx)
    end
    return xx
end

function tlm_traj(model::Lorenz95Model, x::Vector{Float64}, dx::Vector{Float64}, nt::Int)
    xx = copy(x)
    dxx = copy(dx)
    for _ in 1:abs(nt)
        dxx = dRKstep(model, xx, dxx)
        xx = RKstep(model, xx)
    end
    return dxx
end

function ad_traj(model::Lorenz95Model, x::Vector{Float64}, ax::Vector{Float64}, nt::Int)
    axx = copy(ax)
    xx = copy(x)
    traj_xx = Vector{Vector{Float64}}()
    for _ in 1:abs(nt)
        push!(traj_xx, xx)
        xx = RKstep(model, xx)
    end
    for i in 1:abs(nt)
        axx = aRKstep(model, traj_xx[nt - i + 1], axx)
    end
    return axx
end