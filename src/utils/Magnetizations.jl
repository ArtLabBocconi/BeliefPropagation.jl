module Magnetizations

primitive type Mag64 64 end

include("Magnetizations_Common.jl")

magformat() = :plain

convert(::Type{Mag64}, y::Float64) = f2m(y)
convert(::Type{Float64}, y::Mag64) = m2f(y)

forcedmag(y::Float64) = Mag64(y)

mtanh(x::Float64) = f2m(tanh(x))
atanh(x::Mag64) = atanh(m2f(x))

Mag64(pp::Real, pm::Real) = Mag64((pp - pm) / (pp + pm))

isfinite(a::Mag64) = isfinite(m2f(a))

function ⊗(a::Mag64, b::Mag64)
    xa = m2f(a)
    xb = m2f(b)
    return f2m(clamp((xa + xb) / (1 + xa * xb), -1, 1))
end
function ⊘(a::Mag64, b::Mag64)
    xa = m2f(a)
    xb = m2f(b)
    return f2m(xa == xb ? 0.0 : clamp((xa - xb) / (1 - xa * xb), -1, 1))
end

reinforce(m0::Mag64, γ::Float64) = Mag64(tanh(atanh(m2f(m0)) * γ))

damp(newx::Mag64, oldx::Mag64, λ::Float64) = f2m(m2f(newx) * (1 - λ) + m2f(oldx) * λ)
#damp(newx::Mag64, oldx::Mag64, λ::Float64) = Mag64(Float64(newx) * (1 - λ) + Float64(oldx) * λ)

(*)(x::Mag64, y::Mag64) = Mag64(Float64(x) * Float64(y))

merf(x::Float64) = f2m(erf(x))

function exactmix(H::Mag64, p₊::Mag64, p₋::Mag64)
    vH = m2f(H)
    pd = (m2f(p₊) + m2f(p₋)) / 2
    pz = (m2f(p₊) - m2f(p₋)) / 2

    return f2m(pz * vH / (1 + pd * vH))
end

erfmix(H::Mag64, m₊::Float64, m₋::Float64) = Mag64(erfmix(Float64(H), m₊, m₋))

log1pxy(x::Mag64, y::Mag64) = log((1 + Float64(x) * Float64(y)) / 2)

# cross entropy with magnetizations:
#
# -(1 + x) / 2 * log((1 + y) / 2) + -(1 - x) / 2 * log((1 - y) / 2)
#
# == -x * atanh(y) - log(1 - y^2) / 2 + log(2)
function mcrossentropy(x::Mag64, y::Mag64)
    fx = m2f(x)
    fy = m2f(y)
    return -fx * atanh(fy) - log(1 - fy^2) / 2 + log(2)
end

logmag2pp(x::Mag64) = log((1 + m2f(x)) / 2)
logmag2pm(x::Mag64) = log((1 - m2f(x)) / 2)

function logZ(u0::Mag64, u::Vector{Mag64})
    zkip = logmag2pp(u0)
    zkim = logmag2pm(u0)
    for ui in u
        zkip += logmag2pp(ui)
        zkim += logmag2pm(ui)
    end
    zki = exp(zkip) + exp(zkim)
    return log(zki)
end

end
