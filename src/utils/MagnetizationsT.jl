module MagnetizationsT

using StatsFuns

primitive type Mag64 64 end
    
include("Magnetizations_Common.jl")

include("AtanhErf.jl")
using .AtanhErf

const mInf = 30.0

magformat() = :tanh

convert(::Type{Mag64}, y::Float64) = f2m(clamp(atanh(y), -mInf, mInf))
convert(::Type{Float64}, y::Mag64) = tanh(m2f(y))

forcedmag(y::Float64) = f2m(atanh(y))

mtanh(x::Float64) = f2m(x)
atanh(x::Mag64) = m2f(x)

Mag64(pp::Real, pm::Real) = f2m(clamp((log(pp) - log(pm)) / 2, -mInf, mInf))

isfinite(a::Mag64) = !isnan(m2f(a))

⊗(a::Mag64, b::Mag64) = f2m(m2f(a) + m2f(b))
function ⊘(a::Mag64, b::Mag64)
    xa = m2f(a)
    xb = m2f(b)
    return f2m(ifelse(xa == xb, 0.0, xa - xb))
end

reinforce(m0::Mag64, γ::Float64) = f2m(m2f(m0) * γ)

damp(newx::Mag64, oldx::Mag64, λ::Float64) = f2m(m2f(newx) * (1 - λ) + m2f(oldx) * λ)

function (*)(x::Mag64, y::Mag64)
    ax = m2f(x)
    ay = m2f(y)

    if ax ≥ ay && ax ≥ -ay
        t1 = 2ay
    elseif ax ≥ ay && ax < -ay
        t1 = -2ax
    elseif ax < ay && ax ≥ -ay
        t1 = 2ax
    else # ax < ay && ax < -ay
        t1 = -2ay
    end

    t2 = isinf(ax) || isinf(ay) ?
         0.0 : lr(ax + ay) - lr(ax - ay)

    #if !isinf(ax) && !isinf(ay)
    #    t1 = abs(ax + ay) - abs(ax - ay)
    #    t2 = lr(ax + ay) - lr(ax - ay)
    #elseif isinf(ax) && !isinf(ay)
    #    t1 = 2 * sign(ax) * ay
    #    t2 = 0.0
    #elseif !isinf(ax) && isinf(ay)
    #    t1 = 2 * sign(ay) * ax
    #    t2 = 0.0
    #else # isinf(ax) && isinf(ay)
    #    t1 = 2 * sign(ax) * sign(ay) * mInf
    #    t2 = 0.0
    #end

    return f2m((t1 + t2) / 2)
end

#=function _atanherf_largex(x::Float64)
    x² = x^2
    t = 1/x²

    return sign(x) * (2log(abs(x)) + log4π + 2x² +
                      t * @evalpoly(t, 1, -1.25, 3.0833333333333335, -11.03125, 51.0125,
                                   -287.5260416666667, 1906.689732142857, -14527.3759765625, 125008.12543402778, -1.1990066259765625e6)) / 4
end

function atanherf(x::Float64)
    ax = abs(x)
    ax ≤ 4 && return atanh(erf(x))
    if 4 < ax ≤ 4.5 # maximum error with this is about 1.5e-8
        v1 = atanh(erf(x))
        v2 = _atanherf_largex(x)
        return v1 * (4.5 - x) / 0.5 + v2 * (x - 4) / 0.5
    else
        return _atanherf_largex(x)
    end

end=#

merf(x::Float64) = f2m(atanherf(x))

function auxmix(H::Mag64, a₊::Float64, a₋::Float64)
    aH = m2f(H)

    aH == 0.0 && return f2m(0.0)

    xH₊ = aH + a₊
    xH₋ = aH + a₋

    # we need to compute
    #   t1 = abs(xH₊) - abs(a₊) - abs(xH₋) + abs(a₋)
    #   t2 = lr(xH₊) - lr(a₊) - lr(xH₋) + lr(a₋)
    # but we also need to take into account infinities
    if isinf(aH)
        if !isinf(a₊) && !isinf(a₋)
            t1 = sign(aH) * (a₊ - a₋) - abs(a₊) + abs(a₋)
            t2 = -lr(a₊) + lr(a₋)
        elseif isinf(a₊) && !isinf(a₋)
            if sign(a₊) == sign(aH)
                t1 = -sign(aH) * (a₋) + abs(a₋)
                t2 = lr(a₋)
            else
                t1 = -2mInf
                t2 = 0.0
            end
        elseif !isinf(a₊) && isinf(a₋)
            if sign(a₋) == sign(aH)
                t1 = sign(aH) * (a₊) - abs(a₊)
                t2 = -lr(a₊)
            else
                t1 = 2mInf
                t2 = 0.0
            end
        else # isinf(a₊) && isinf(a₋)
            if (sign(a₊) == sign(aH) && sign(a₊) == sign(aH)) || (sign(a₊) ≠ sign(aH) && sign(a₊) ≠ sign(aH))
                t1 = 0.0
                t2 = 0.0
            elseif sign(a₊) == sign(aH) # && sign(a₋) ≠ sign(aH)
                t1 = 2mInf
                t2 = 0.0
            else # sign(a₋) == sign(aH) && sign(a₊) ≠ sign(aH)
                t1 = -2mInf
                t2 = 0.0
            end
        end
    else # !isinf(aH)
        t1 = 0.0
        t1 += isinf(a₊) ? 0.0 : abs(xH₊) - abs(a₊)
        t1 -= isinf(a₋) ? 0.0 : abs(xH₋) - abs(a₋)
        t2 = lr(xH₊) - lr(a₊) - lr(xH₋) + lr(a₋)
    end

    return f2m((t1 + t2) / 2)
end

exactmix(H::Mag64, p₊::Mag64, p₋::Mag64) = auxmix(H, m2f(p₊), m2f(p₋))

function erfmix(H::Mag64, m₊::Float64, m₋::Float64)
    aerf₊ = atanherf(m₊)
    aerf₋ = atanherf(m₋)
    return auxmix(H, aerf₊, aerf₋)
#
    #xH₊ = aH + aerf₊
    #xH₋ = aH + aerf₋
#
    #t1 = abs(xH₊) - abs(aerf₊) - abs(xH₋) + abs(aerf₋)
    #t2 = lr(xH₊) - lr(aerf₊) - lr(xH₋) + lr(aerf₋)
    #return f2m((t1 + t2) / 2)
end

# log((1 + x * y) / 2)
# == log2cosh(atanh(x) + atanh(y)) - log2cosh(atanh(x)) - log2cosh(atanh(y))
function log1pxy(x::Mag64, y::Mag64)
    ax = m2f(x)
    ay = m2f(y)

    return !isinf(ax) && !isinf(ay) ? abs(ax + ay) - abs(ax) - abs(ay) + lr(ax + ay) - lr(ax) - lr(ay) :
            isinf(ax) && !isinf(ay) ? sign(ax) * ay - abs(ay) - lr(ay) :
           !isinf(ax) &&  isinf(ay) ? sign(ay) * ax - abs(ax) - lr(ax) :
           sign(ax) == sign(ay)     ? 0.0 : -Inf # isinf(ax) && isinf(ay)
end

# cross entropy with magnetizations:
#
# -(1 + x) / 2 * log((1 + y) / 2) - (1 - x) / 2 * log((1 - y) / 2)
#
# == -x * atanh(y) - log(1 - y^2) / 2 + log(2)
#
# with atanh's:
#
# == -ay * tanh(ax) + log(2cosh(ay))
function mcrossentropy(x::Mag64, y::Mag64)
    tx = tanh(m2f(x))
    ay = m2f(y)
    return !isinf(ay)          ? -abs(ay) * (sign0(ay) * tx - 1) + lr(ay) :
           sign(tx) ≠ sign(ay) ? Inf : 0.0
end

function logZ(u0::Mag64, u::Vector{Mag64}, y::Float64=1.)
    a0 = m2f(u0)
    if !isinf(a0)
        s1 = y*a0
        s2 = y*log2cosh(a0)
        hasinf = 0.
    else
        s1 = s2 = 0.0
        hasinf = sign(a0)
    end
    for ui in u
        ai = m2f(ui)
        if !isinf(ai)
            s1 += ai
            s2 += log2cosh(ai)
        elseif hasinf == 0
            hasinf = sign(ai)
        elseif hasinf ≠ sign(ai)
            return -Inf
        end
    end
    return log2cosh(s1) - s2
end

end
