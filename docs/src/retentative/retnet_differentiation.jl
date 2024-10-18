# # Taking gradients
# 
# So we have a nice implementation of retnets, but it is not yet ready for gradients. 
# The problem is that some functions, like `rot_matrix`, 'rotary_embedding', and the 
# `linear_attention` are not differentiable. The reason why is that most AD frameworks
# do not differentiate through setindex. The exception is Enzyme, with which the author 
# does not have experience. But writing a custom gradient for these functions is not 
# that difficult.

using ChainRulesCore
using FiniteDifferences
using Zygote


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
    function foo_mul_pullback(ȳ)
        f̄ = NoTangent()
        x̄, θ̄ = ∂rotary_embedding(ȳ, x, θ)
        return f̄, x̄, θ̄
    end
    return y, foo_mul_pullback
end

x = randn(4,10)
θ = 2π .* rand(2)

rotary_embedding(x, θ)

gradient(x -> sum(rotary_embedding(x, θ)), x)[1]  ≈ grad(central_fdm(5,1), x -> sum(rotary_embedding(x, θ)), x)[1]
gradient(θ -> sum(rotary_embedding(x, θ)), θ)[1]  ≈ grad(central_fdm(5,1), θ -> sum(rotary_embedding(x, θ)), θ)[1]