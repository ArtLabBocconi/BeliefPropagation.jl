type OutputLayer <: AbstractLayer
    l::Int
    labels::IVec
    allpd::VecVec # p(σ=up) from fact ↑ to y
    β::Float64
end

function OutputLayer(σ::Vector{Int}; β=Inf)
    allpd = VecVec()
    K = maximum(σ)
    if K<=1 #binary classification
        push!(allpd, Float64[(1+tanh(β*σ[a]))/2 for a=1:length(σ)])
        out = OutputLayer(-1,σ,allpd, β)
    elseif K >= 2 # K-ary classification
        for k=1:K
            push!(allpd, Float64[σ[a]==k ? 1 : 0 for a=1:length(σ)])
            out = OutputLayer(-1,σ, allpd, β)
        end
    end

    return out
end

initrand!(layer::OutputLayer) = nothing
