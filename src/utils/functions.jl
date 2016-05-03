
G(x) = e^(-(x^2)/2) / √(convert(typeof(x),2) * π)
H(x) = erfc(x / √convert(typeof(x),2)) / 2


# logcosh(x) = abs(x) > 30 ? abs(x) : log(cosh(x))
myatanh(x) = atanh(x)
logcosh(x) = abs(x) > 600 ? abs(x) : log(cosh(x))
atanh2Hm1(x) = abs(x) > 8 ? -sign(x)*0.25*(log(2π) + x^2 + 2log(abs(x))) :
                atanh(2H(x)-1)

function GHapp(x)
    y = 1/x
    y2 = y^2
    x + y * (1 - 2y2 * (1 - 5y2 * (1 - 7.4y2)))
end

GH(x) = x > 30.0 ? GHapp(x) : G(x) / H(x)

function GHnaive(p, x)
    Hp = H(x); Hm = 1-Hp
    Gp = G(x); Gm = Gp
    (p*Gp - (1-p)*Gm) / (p*Hp + (1-p)*Hm)
end

function GH(p, x)
    p == 1 && return GH(x)
    p == 0 && return -GH(-x)
    abs(x) < 5 && return GHnaive(p, x)
    up = atanh(2p-1)
    mp = 2p - 1
    mp == 0. && return 0.
    uh = atanh2Hm1(x)
    ex = (log(abs(mp)) + logcosh(up) + logcosh(uh)) - (logcosh(up+uh) + x^2/2)
    if abs(ex) > 600
        ex = sign(ex)*600.
    end
    res = sign(mp)* exp(ex) / √(2π)
    # if !isfinite(res)
    #     @show p up ug uh ex log(abs(mp)) logcosh(up)  logcosh(uh) logcosh(up+uh)
    # end
    # @assert isfinite(res)
    return sign(mp)* exp(ex) / √(2π)
end

# GH(1,x) =GH(x)
# function GH2(p, x)
#     Hp = H(x); Hm = 1-Hp
#     Gp = G(x); Gm = Gp
#     Gp / (p*Hp + (1-p)*Hm)
# end

function DH(p, x, y, C)
    Hpp = H(-(x+y)/C)
    Hpm = H(-(x-y)/C)
    Hmp = 1 - Hpp
    Hmm = 1 - Hpm
    (p*(Hpp - Hpm) + (1-p)*(Hmp - Hmm)) / (p*(Hpp + Hpm) + (1-p)*(Hmp + Hmm))
end
#
# function DH(p, x, C)
#     Hp = H(-x/C)
#     Hm = 1 - Hm
#     (p*Hp - (1-p)*Hm) / (p*Hp  + (1-p)*Hm)
# end

# function DH(p, x, y, C)
#     Hpp = H(-(x+y)/C)
#     Hpm = H(-(x-y)/C)
#     Hmp = 1 - Hpp
#     Hmm = 1 - Hpm
#     0.5 * log((p*Hpp+ (1-p)*Hmp) / (p*Hpm + (1-p)*Hmm))
# end
