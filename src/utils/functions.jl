G(x) = exp(-(x^2)/2) / √(convert(typeof(x),2) * π)
H(x) = erfc(x / √convert(typeof(x),2)) / 2

lg2 = log(2)

# logcosh(x) = abs(x) > 30 ? abs(x) : log(cosh(x))
∞atanh = 25.

function myatanh(x)
    y = atanh(x)
    return isfinite(y) ? y : sign(x)*∞atanh
end

myatanh(p,m) = _myatanh(p/(p+m), m/(p+m))
function _myatanh(p,m)
    # @assert p >= 0 "p >= 0 p=$p"
    # @assert m >= 0
    p == 0 && return -∞atanh
    m == 0 && return ∞atanh
    y = 0.
    if m < 1e-10
        y = 0.5*(lg2 - log(2m))
    elseif p < 1e-10
        y = -0.5*(lg2 - log(2p))
    else
        y = atanh(p-m)
    end
    @assert isfinite(y) "y=$y p=$p m=$m"
    return y
end

logcosh(x) = abs(x) > 600 ? abs(x) - lg2 : log(cosh(x))
logsinhabs(x) = abs(x) > 600 ? abs(x) - lg2 : log(sinh(abs(x)))

atanh2Hm1(x) = abs(x) > 6 ? -sign(x)*0.25*(log(2π) + x^2 + 2log(abs(x))) :
                atanh(2H(x)-1)

function GHapp(x)
    y = 1/x
    y2 = y^2
    x + y * (1 - 2y2 * (1 - 5y2 * (1 - 7.4y2)))
end

GH(x) = x > 30.0 ? GHapp(x) : G(x) / H(x)

function GHnaive(uσ, x)
    Hp = H(x); Hm = 1-Hp
    Gp = G(x);
    p = (tanh(uσ)+1)/2
    Gp*(2p-1) / (p*Hp + (1-p)*Hm)
end

function GH(uσ, x)
    uσ == 0. && return 0.
    uσ == Inf && return GH(x)
    uσ == -Inf && return -GH(-x)
    abs(x) < 5 && return GHnaive(uσ, x)
    uh = atanh2Hm1(x)
    ex = (logsinhabs(uσ) + logcosh(uh)) - (logcosh(uσ+uh) + x^2/2)
    # @show x mσ
    # ex = (logsinhabs(uσ) + logcosh(uh)) - (logcosh(uσ+uh) + x^2/2)
    if abs(ex) > 600
        ex = sign(ex)*600.
    end
    res = sign(uσ)* exp(ex) * √(2/π)
    # if !isfinite(res)
    #     @show p up ug uh ex log(abs(mp)) logcosh(up)  logcosh(uh) logcosh(up+uh)
    # end
    # @assert isfinite(res)
    return res
end

# TODO approx
function DH(σu, x, y, C)
    p = (1+tanh(σu)) /2
    Hpp = H(-(x+y)/C)
    Hpm = H(-(x-y)/C)
    Hmp = 1 - Hpp
    Hmm = 1 - Hpm
    (p*(Hpp - Hpm) + (1-p)*(Hmp - Hmm)) / (p*(Hpp + Hpm) + (1-p)*(Hmp + Hmm))
end
