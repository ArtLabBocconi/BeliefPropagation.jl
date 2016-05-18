export Mag64, mfill, mflatp, mrand, damp, reinforce, ⊗, ⊘, ↑, sign0,
       merf, exactmix, erfmix, mtanh, log1pxy, mcrossentropy,
       logZ, forcedmag, showinner, parseinner, magformat, log2cosh

using Base.Intrinsics

import Base: convert, promote_rule, *, /, +, -, sign, signbit, isnan,
             show, showcompact, abs, isfinite, isless, copysign,
             atanh, zero

m2f(a::Mag64) = box(Float64, unbox(Mag64, a))
f2m(a::Float64) = box(Mag64, unbox(Float64, a))

convert{T<:Real}(::Type{T}, y::Mag64) = convert(T, Float64(y))
convert(::Type{Mag64}, y::Real) = convert(Mag64, Float64(y))

promote_rule(::Type{Mag64}, ::Type{Float64}) = Float64

#mag(a::Real) = convert(Mag64, a)
#mag{T<:Real}(v::Vector{T}) = Mag64[Mag64(x) for x in v]

zero(::Type{Mag64}) = f2m(0.0)

isnan(a::Mag64) = isnan(m2f(a))

abs(a::Mag64) = f2m(abs(m2f(a)))
copysign(x::Mag64, y::Float64) = f2m(copysign(m2f(x), y))
copysign(x::Mag64, y::Mag64) = f2m(copysign(m2f(x), m2f(y)))

⊗(a::Mag64, b::Float64) = a ⊗ Mag64(b)
⊘(a::Mag64, b::Float64) = a ⊘ Mag64(b)

⊗(a::Float64, b::Mag64) = b ⊗ a


(*)(a::Mag64, b::Real) = Float64(a) * b
(*)(a::Real, b::Mag64) = b * a

(+)(a::Mag64, b::Real) = Float64(a) + b
(+)(a::Real, b::Mag64) = b + a

(-)(a::Mag64, b::Real) = Float64(a) - b
(-)(a::Real, b::Mag64) = -(b - a)
(-)(a::Mag64) = f2m(-m2f(a))

(-)(a::Mag64, b::Mag64) = Float64(a) - Float64(b)

sign(a::Mag64) = sign(m2f(a))
signbit(a::Mag64) = signbit(m2f(a))
sign0(a::Union{Mag64,Real}) = (1 - 2signbit(a))

show(io::IO, a::Mag64) = show(io, Float64(a))
showcompact(io::IO, a::Mag64) = showcompact(io, Float64(a))
showinner(io::IO, a::Mag64) = show(io, m2f(a))

parseinner(::Type{Val{:tanh}}, s::AbstractString) = mtanh(parse(Float64, s))
parseinner(::Type{Val{:plain}}, s::AbstractString) = Mag64(parse(Float64, s))

mfill(x::Float64, n::Int) = Mag64[Mag64(x) for i = 1:n]
mflatp(n::Int) = mfill(0.0, n)

mrand(x::Float64, n::Int) = Mag64[Mag64(x * (2*rand()-1)) for i = 1:n]

reinforce(m::Mag64, m0::Mag64, γ::Float64) = m ⊗ reinforce(m0, γ)

damp(newx::Float64, oldx::Float64, λ::Float64) = newx * (1 - λ) + oldx * λ

Base.(:(==))(a::Mag64, b::Float64) = (Float64(a) == b)
Base.(:(==))(a::Float64, b::Mag64) = (b == a)

isless(m::Mag64, x::Real) = isless(Float64(m), x)
isless(x::Real, m::Mag64) = isless(x, Float64(m))

merf(x::Float64) = erf(x)

function erfmix(H::Float64, m₊::Float64, m₋::Float64)
    erf₊ = erf(m₊)
    erf₋ = erf(m₋)
    return H * (erf₊ - erf₋) / (2 + H * (erf₊ + erf₋))
end

logZ(u::Vector{Mag64}) = logZ(zero(Mag64), u)

↑(m::Mag64, x::Real) = mtanh(x * atanh(m))

lr(x::Float64) = log1p(exp(-2abs(x)))
log2cosh(x::Float64) = abs(x) + lr(x)
