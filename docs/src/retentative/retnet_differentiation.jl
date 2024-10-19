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
    function rotary_pullback(ȳ)
        f̄ = NoTangent()
        x̄, θ̄ = ∂rotary_embedding(ȳ, x, θ)
        return f̄, x̄, θ̄
    end
    return y, rotary_pullback
end

x = randn(4,10)
θ = 2π .* rand(2)

rotary_embedding(x, θ)

gradient(x -> sum(rotary_embedding(x, θ)), x)[1]  ≈ grad(central_fdm(5,1), x -> sum(rotary_embedding(x, θ)), x)[1]
gradient(θ -> sum(rotary_embedding(x, θ)), θ)[1]  ≈ grad(central_fdm(5,1), θ -> sum(rotary_embedding(x, θ)), θ)[1]


# The rotary embedding is now differentiable. Let's now turn our attention to the 
# linear attention. We start with the version expressed in terms of matrix multiplication
# and check if Zygote can take a gradient as-is.

struct DMatrix{T} <:AbstractMatrix{T}
    γ::T
    size::Int64
end

Base.getindex(d::DMatrix{T}, n::Integer, m::Integer) where {T} = n ≥ m ? d.γ^(n-m) : zero(T)
Base.size(d::DMatrix,i::Integer) = d.size
Base.size(d::DMatrix) = (d.size, d.size)

function linear_attention(Q, K, V, γ::Real)
    D = DMatrix(γ, size(Q, 2));
    transpose(((Q' * K) .* D) * V')
end

Q = randn(8,17);
K = randn(8,17);
V = randn(6,17);
γ = 0.99;

gradient(Q -> sum(linear_attention(Q, K, V, γ)), Q)[1] ≈ grad(central_fdm(5,1), Q -> sum(linear_attention(Q, K, V, γ)), Q)[1]
# Which fails with an error
# ```julia
# Stacktrace:
#   [1] error(s::String)
#     @ Base ./error.jl:35
#   [2] (::Zygote.Jnew{DMatrix{Float64}, Nothing, false})(Δ::Matrix{Float64})
#     @ Zygote ~/.julia/packages/Zygote/NRp5C/src/lib/lib.jl:330
#   [3] (::Zygote.var"#2210#back#316"{Zygote.Jnew{DMatrix{Float64}, Nothing, false}})(Δ::Matrix{Float64})
#     @ Zygote ~/.julia/packages/ZygoteRules/M4xmc/src/adjoint.jl:72
#   [4] DMatrix
#     @ ~/Work/Julia/Dissected/docs/src/retentative/retnet_differentiation.jl:72 [inlined]
# ```
# Which is caused but the use of our custom matrix `DMatrix`. The problem is that Zygote does not know
# how to construct gradient off `γ` from the gradient with respect to `DMatrix`. We can fix this by
# definiing a custom gradient (called adjoint) for `DMatrix`.


# Let's now move to our implementation of the linear attention with for-cycles. The process of writing
# is similar to the rotaty embedding. We take the forward pass and use it as a template to write 
# the backward pass. As always, we first write make the function correct, without multi-threadding
# and all that stuff, and then make it performant if possible. Adding threads might be tricky, because
# we need to ensure that there will not be any race condition and we want to avoid locks if possible.

function _slices(hidden_dim::Integer, output_dim::Integer, nheads::Integer)
    head_dim = hidden_dim ÷ nheads
    odim = output_dim ÷ nheads
    kvslices = ntuple(i -> (i-1)*head_dim+1:i*head_dim, nheads)
    oslices = ntuple(i -> (i-1)*odim+1:i*odim, nheads)
    return(kvslices, oslices)
end

function _slices(K::AbstractMatrix, V::AbstractMatrix, γs::AbstractVector)
    _slices(size(K,1), size(V,1), length(γs))
end


function linear_attention(Q, K, V, γs::AbstractVector{<:Number})
    kvslices, oslices = _slices(K,V,γs)
    o = zeros(eltype(V), size(V))
    l = size(K,2)
    for n in 1:l
        for (j, (kvslice, oslice)) in enumerate(zip(kvslices, oslices))
            for m in 1:n
                α = zero(eltype(Q))
                for k in kvslice
                    α += Q[k, n] * K[k, m]
                end

                γ = γs[j]^(n-m)
                for k in oslice
                    o[k, n] += γ * α * V[k, m]
                end
            end
        end
    end
    o
end

function ∂linear_attention(ȳ, Q, K, V, γs::AbstractVector{<:Number})
	T = eltype(Q)
	Q̄ = similar(Q)
	K̄ = similar(K)
	V̄ = similar(V)
	γ̄s = similar(γs)
	Q̄ .= 0
	K̄ .= 0
	V̄ .= 0
	γ̄s .= 0
    kvslices, oslices = _slices(K,V,γs)
    l = size(K,2)
    for n in 1:l
        for (j, (kvslice, oslice)) in enumerate(zip(kvslices, oslices))
            for m in 1:n
            	# we need to recompute the alpha for the gradient
                α = zero(T)
                for k in kvslice
                    α += Q[k, n] * K[k, m]
                end

                # then we compute the gradient of the output with respect to α and γ
                γ = γs[j]^(n-m)
            	∂α = zero(T)
            	∂γ = zero(T)
                for k in oslice
                    ∂γ += ȳ[k,n] * α * V[k, m]
                    ∂α += ȳ[k,n] * γ * V[k, m]
                    V̄[k, m] += ȳ[k,n] * γ * α
                end

                # with that we update the gradient of Q and K
                for k in kvslice
                    Q̄[k,n] += ∂α * K[k, m]
                    K̄[k,m] += ∂α * Q[k, n]
                end

                # and finally we updaate the gradient of γ
                γ̄s[j] += n != m ? (n-m)*γs[j]^(n-m-1)*∂γ : zero(T)
            end
        end
    end
    return(Q̄, K̄, V̄, γ̄s)
end

function ChainRulesCore.rrule(::typeof(linear_attention), Q, K, V, γs::AbstractVector{<:Number})
    y = linear_attention(Q,K,V,γs)
    function linatt_pullback(ȳ)
        f̄ = NoTangent()
        Q̄, K̄, V̄, γ̄s = ∂linear_attention(ȳ, Q, K, V, γs)
        return f̄, Q̄, K̄, V̄, γ̄s
    end
    return y, linatt_pullback
end

# By comparing the output of Zygote with the finite difference approximation, we can see that the
# that the implementation is correct. We can now move to the multi-threaded version.

Q = randn(8,17);
K = randn(8,17);
V = randn(6,17);
γs = [0.99, 0.95];

gradient(Q -> sum(linear_attention(Q, K, V, γs)), Q)[1] ≈ grad(central_fdm(5,1), Q -> sum(linear_attention(Q, K, V, γs)), Q)[1]
gradient(K -> sum(linear_attention(Q, K, V, γs)), K)[1] ≈ grad(central_fdm(5,1), K -> sum(linear_attention(Q, K, V, γs)), K)[1]
gradient(V -> sum(linear_attention(Q, K, V, γs)), V)[1] ≈ grad(central_fdm(5,1), V -> sum(linear_attention(Q, K, V, γs)), V)[1]
gradient(γs -> sum(linear_attention(Q, K, V, γs)), γs)[1] ≈ grad(central_fdm(5,1), γs -> sum(linear_attention(Q, K, V, γs)), γs)[1]
