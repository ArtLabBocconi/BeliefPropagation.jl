mutable struct InputLayer <: AbstractLayer
    l::Int
    allpu::VecVec # p(σ=up) from fact ↑ to y
    isbinary::Bool
    ξ::Matrix
end

function InputLayer(ξ::Matrix)
    isbinary = true
    for x in ξ
        if x != 1 && x != -1
            isbinary = false
            break
        end
    end
    InputLayer(ξ, isbinary)
end

function InputLayer(ξ::Matrix, isbinary::Bool)
    return InputLayer(1, VecVec(), isbinary, ξ)
end

initrand!(layer::InputLayer) = nothing
