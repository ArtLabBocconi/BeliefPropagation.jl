type InputLayer <: AbstractLayer
    l::Int
    allpu::VecVec # p(σ=up) from fact ↑ to y
end

InputLayer(ξ::Matrix) = InputLayer(1,
    [Float64[(1+ξ[i,a])/2 for a=1:size(ξ,2)] for i=1:size(ξ,1)])

initrand!(layer::InputLayer) = nothing
