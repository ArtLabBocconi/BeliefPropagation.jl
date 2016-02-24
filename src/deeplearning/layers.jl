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

istoplayer(layer::AbstractLayer) = (typeof(layer.top_layer) == OutputLayer)
isbottomlayer(layer::AbstractLayer) = (typeof(layer.bottom_layer) == InputLayer)

function Base.show{L <: Union{TapExactLayer,TapLayer}}(io::IO, layer::L)
    @extract layer K N M allm allmy allmh allpu allpd
    println(io, "m=$(allm[1])")
    println(io, "my=$(allmy[1])")
end

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
        updateVarY!(lay2, a)
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
