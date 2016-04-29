"""
Input from bottom.allpu and top.allpd
and modifies its allpu and allpd
"""

G(x) = e^(-(x^2)/2) / √(convert(typeof(x),2) * π)
H(x) = erfc(x / √convert(typeof(x),2)) / 2
#GH(x) = ifelse(x > 30.0, x+(1-2/x^2)/x, G(x) / H(x))
function GHapp(x)
    y = 1/x
    y2 = y^2
    x + y * (1 - 2y2 * (1 - 5y2 * (1 - 7.4y2)))
end
GH(x) = x > 30.0 ? GHapp(x) : G(x) / H(x)

# GH(1,x) =GH(x)
function GH(p, x)
    Hp = H(x); Hm = 1-Hp
    Gp = G(x); Gm = Gp
    (p*Gp - (1-p)*Gm) / (p*Hp + (1-p)*Hm)
end
# GH(1,x) =GH(x)
function GH2(p, x)
    Hp = H(x); Hm = 1-Hp
    Gp = G(x); Gm = Gp
    Gp / (p*Hp + (1-p)*Hm)
end

# function DH(p, x, y, C)
#     Hpp = H(-(x+y)/C)
#     Hpm = H(-(x-y)/C)
#     Hmp = 1 - Hpp
#     Hmm = 1 - Hpm
#     (p*(Hpp - Hpm) + (1-p)*(Hmp - Hmm)) / (p*(Hpp + Hpm) + (1-p)*(Hmp + Hmm))
# end
#
# function DH(p, x, C)
#     Hp = H(-x/C)
#     Hm = 1 - Hm
#     (p*Hp - (1-p)*Hm) / (p*Hp  + (1-p)*Hm)
# end

function DH(p, x, y, C)
    Hpp = H(-(x+y)/C)
    Hpm = H(-(x-y)/C)
    Hmp = 1 - Hpp
    Hmm = 1 - Hpm
    0.5 * log((p*Hpp+ (1-p)*Hmp) / (p*Hpm + (1-p)*Hmm))
end

myatanh(x) = atanh(x)



abstract AbstractLayer

myatanh(x::Float64) = ifelse(abs(x) > 15, ifelse(x>0,50,-50), atanh(x))

∞ = 10000

type DummyLayer <: AbstractLayer
end

include("layers/input.jl")
include("layers/output.jl")
include("layers/maxsum.jl")
include("layers/bp.jl")
include("layers/tap.jl")
include("layers/parity.jl")
include("layers/bp_real.jl")


istoplayer(layer::AbstractLayer) = (typeof(layer.top_layer) == OutputLayer)
isbottomlayer(layer::AbstractLayer) = (typeof(layer.bottom_layer) == InputLayer)
isonlylayer(layer::AbstractLayer) = istoplayer(layer) && isbottomlayer(layer)

function Base.show{L <: Union{TapExactLayer,TapLayer}}(io::IO, layer::L)
    @extract layer K N M allm allmy allmh allpu allpd
    println(io, "m=$(allm[1])")
    println(io, "my=$(allmy[1])")
end

getW(lay::AbstractLayer) = getWBinary(lay)
getW(lay::BPRealLayer) = getWReal(lay)
getWReal(lay::AbstractLayer) = lay.allm
getWBinary(lay::AbstractLayer) = [Float64[1-2signbit(m) for m in magk]
                        for magk in getWReal(lay)] # TODO return BitArray

function energy(lay::OutputLayer, ξ::Vector, a)
    @extract lay: labels
    @assert length(ξ) == 1
    return all((labels[a] .* ξ) .> 0) ? 0 : 1
end

forward(lay::AbstractLayer, ξ::Vector) = forwardBinary(lay, ξ)
forward(lay::BPRealLayer, ξ::Vector) = forwardReal(lay, ξ)
forward(lay::ParityLayer, ξ::Vector) = forwardParity(lay, ξ)

function forwardBinary(lay::AbstractLayer, ξ::Vector)
    @extract lay: N K
    W = getWBinary(lay)
    stability = map(w->dot(ξ, w), W)
    σks = Int[ifelse(stability[k] > 0, 1, -1) for k=1:K]
    return σks, stability
end

function forwardReal(lay::AbstractLayer, ξ::Vector)
    @extract lay: N K
    W = getWReal(lay)
    stability = map(w->dot(ξ, w), W)
    σks = Int[ifelse(stability[k] > 0, 1, -1) for k=1:K]
    return σks, stability
end

function forwardParity(lay::AbstractLayer, ξ::Vector)
    @extract lay: N K
    @assert N==2
    @assert K==1
    W = getWReal(lay)
    stability = 0.
    σks = [sign(ξ[1]*ξ[2])]

    return σks, stability
end

initYBottom!(lay::AbstractLayer, a::Int) = updateVarY!(lay, a) #TODO define for every layer type

chain!(lay1::InputLayer, lay2::OutputLayer) = error("Cannot chain InputLayer and OutputLayer")

function chain!(lay1::AbstractLayer, lay2::OutputLayer)
    lay1.top_allpd = lay2.allpd
    lay2.l = lay1.l+1
    lay1.top_layer = lay2
end

function chain!{L <: AbstractLayer}(lay1::InputLayer, lay2::L)
    lay2.l = lay1.l+1
    lay2.bottom_allpu = lay1.allpu
    for a=1:lay2.M
        initYBottom!(lay2, a)
    end
    lay2.bottom_layer = lay1
end

function chain!(lay1::AbstractLayer, lay2::AbstractLayer)
    lay2.l = lay1.l+1
    lay1.top_allpd = lay2.allpd
    lay2.bottom_allpu = lay1.allpu
    lay1.top_layer = lay2
    lay2.bottom_layer = lay1
end
