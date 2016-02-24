type OutputLayer <: AbstractLayer
    l::Int
    allpd::VecVec # p(σ=up) from fact ↑ to y
end

function OutputLayer(σ::Vector)
    allpd = VecVec()
    push!(allpd, Float64[(1+σ[a])/2 for a=1:length(σ)])
    OutputLayer(-1,allpd)
end

initrand!(layer::OutputLayer) = nothing
