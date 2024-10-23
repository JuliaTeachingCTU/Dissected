function rot_matrix(θ::Vector)
    o = zeros(eltype(θ), 2*length(θ), 2*length(θ))
    for k in 1:length(θ)
        i = 2*k - 1
        o[i,i] = o[i+1,i+1] = cos(θ[k])
        o[i+1,i] =  sin(θ[k])
        o[i,i+1] = -sin(θ[k])
    end
    o
end

function rotary_embedding(x::AbstractMatrix, θ::AbstractVector)
    d = size(x,1)
    2*d == length(θ) && error("θ should be twice of x")
    o = similar(x)
    @inbounds for i in axes(x,2)
        for (kᵢ, θₖ) in enumerate(θ)
            k = 2*kᵢ - 1
            sinᵢ, cosᵢ  = sincos(i * θₖ)
            o[k,i]   = x[k,i]   * cosᵢ  - x[k+1,i] * sinᵢ
            o[k+1,i] = x[k+1,i] * cosᵢ  + x[k,i]   * sinᵢ
        end
    end
    o
end

function ∂rotary_embedding(ȳ, x::AbstractMatrix, θ::AbstractVector)
    x̄ = similar(x)
    θ̄ = similar(θ)
    θ̄ .= 0
    @inbounds for i in axes(x,2)
        for (kᵢ, θₖ) in enumerate(θ)
            k = 2*kᵢ - 1
            sinᵢ, cosᵢ  = sincos(i * θₖ)
            x̄[k,i]   =  ȳ[k,i] * cosᵢ + ȳ[k+1,i] * sinᵢ
            x̄[k+1,i] = -ȳ[k,i] * sinᵢ + ȳ[k+1,i] * cosᵢ

            θ̄[kᵢ] += i* (- ȳ[k,i]   * x[k,i]  * sinᵢ - x[k+1,i] * ȳ[k,i] * cosᵢ 
                   - x[k+1,i] * ȳ[k+1,i] *sinᵢ + x[k,i] * ȳ[k+1,i] * cosᵢ)
        end
    end
    x̄, θ̄
end

function ChainRulesCore.rrule(::typeof(rotary_embedding), x::AbstractMatrix, θ::AbstractVector)
    y = rotary_embedding(x, θ)
    function rotary_pullback(ȳ)
        f̄ = NoTangent()
        x̄, θ̄ = ∂rotary_embedding(ȳ, x, θ)
        return f̄, x̄, θ̄
    end
    return y, rotary_pullback
end
