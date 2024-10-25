function _slices(hidden_dim::Integer, output_dim::Integer, nheads::Val{N}) where {N}
    head_dim = hidden_dim ÷ N
    odim = output_dim ÷ N
    kvslices = ntuple(i -> (i-1)*head_dim+1:i*head_dim, nheads)
    oslices = ntuple(i -> (i-1)*odim+1:i*odim, nheads)
    return(kvslices, oslices)
end

function _slices(hidden_dim::Integer, output_dim::Integer, γs)
    _slices(hidden_dim, output_dim, Val(length(γs)))
end

function _slices(K::AbstractMatrix, V::AbstractMatrix, γs)
    _slices(size(K,1), size(V,1), γs)
end

##########
#   Linear attention using Matrix product
##########
struct DMatrix{T} <:AbstractMatrix{T}
    γ::T
    size::Int64
end

Base.getindex(d::DMatrix{T}, n::Integer, m::Integer) where {T} = n ≥ m ? d.γ^(n-m) : zero(T)
Base.size(d::DMatrix,i::Integer) = d.size
Base.size(d::DMatrix) = (d.size, d.size)

function linear_attention_product(Q, K, V, γ::Real)
    D = DMatrix(γ, size(Q, 2));
    transpose(((Q' * K) .* D) * V')
end

function linear_attention_product(Q, K, V, γs::AbstractVector{<:Real})
    kvslices, oslices = _slices(Q, V, γs)
    os = map(enumerate(zip(γs, kvslices, oslices))) do (i, (γ, kvslice, oslice))
        linear_attention_product(Q[kvslice,:], K[kvslice,:], V[oslice,:], γ)
    end
    reduce(vcat, os)
end

##########
#   Linear attention using forloop
##########
function linear_attention_forloop(Q, K, V, γ::Real)
    linear_attention_forloop(Q,K,V,[γ])
end

function linear_attention_forloop(Q, K, V, γs::AbstractVector)
    linear_attention_forloop(Q, K, V, Val(length(γs)), γs)
end

function linear_attention_forloop(Q, K, V, γs, nheads)
    kvslices, oslices = _slices(K, V, nheads)
    o = zeros(eltype(V), size(V))
    l = size(K,2)
    Threads.@threads :static for i in 1:Threads.nthreads()
        for n in i:Threads.nthreads():l
            for (j, (kvslice, oslice)) in enumerate(zip(kvslices, oslices))
                γ = γs[j]
                γₙ = γ ^ n
                @inbounds for m in 1:n
                    α = zero(eltype(Q))
                    for k in kvslice
                        α += Q[k, n] * K[k, m]
                    end

                    γₙ /= γ
                    for k in oslice
                        o[k, n] += γₙ * α * V[k, m]
                    end
                end
            end
        end
    end
    o
end

function ∂linear_attention(ȳ, Q, K, V, γs::AbstractVector{<:Number})
    ∂linear_attention(ȳ, Q, K, V, γs, Val(length(γs)))
end


function ∂linear_attention(ȳ, Q, K, V, γs::AbstractVector{<:Number}, nheads::Val{N}) where {N}
    T = eltype(Q)
    Q̄ = similar(Q)
    K̄ = similar(K)
    V̄ = similar(V)
    γ̄s = similar(γs)
    Q̄ .= 0
    K̄ .= 0
    V̄ .= 0
    γ̄s .= 0
    kvslices, oslices = _slices(K, V, nheads)
    l = size(K,2)

    γsₙ = ones(T, length(γs))
    for n in 1:l
        γsₙ .*= γs
        for (j, (γsₘ, kvslice, oslice)) in enumerate(zip(γsₙ, kvslices, oslices))
            γ = γsₘ
            @inbounds for m in 1:n
                ## we need to recompute the alpha for the gradient
                α = zero(T)
                for k in kvslice
                    α += Q[k, n] * K[k, m]
                end

                ## then we compute the gradient of the output with respect to α and γ
                γ = γ / γs[j] # γ = γs[j]^(n-m)
                ∂α = zero(T)
                ∂γ = zero(T)
                for k in oslice
                    ∂γ += ȳ[k,n] * α * V[k, m]
                    ∂α += ȳ[k,n] * γ * V[k, m]
                    V̄[k, m] += ȳ[k,n] * γ * α
                end

                ## with that we update the gradient of Q and K
                for k in kvslice
                    Q̄[k,n] += ∂α * K[k, m]
                    K̄[k,m] += ∂α * Q[k, n]
                end

                ## and finally we update the gradient of γ
                γ̄s[j] += n != m ? (n-m)*γ*∂γ / γs[j] : zero(T)
                # γ̄s[j] += n != m ? (n-m)*γs[j]^(n-m-1)*∂γ : zero(T)
            end
        end
    end
    return(Q̄, K̄, V̄, γ̄s)
end

function ∂linear_attention(ȳ, Q, K, V, γ::Real)
    Q̄, K̄, V̄, γ̄s = ∂linear_attention(ȳ, Q, K, V, [γ])
    Q̄, K̄, V̄, only(γ̄s)
end

function linear_attention(Q, K, V, γ)
    linear_attention_forloop(Q, K, V, γ)
end

function linear_attention(Q, K, V, nheads, γ)
    linear_attention_forloop(Q, K, V, γ, nheads)
end

function ChainRulesCore.rrule(::typeof(linear_attention), Q, K, V, nheads, γs)
    y = linear_attention_forloop(Q, K, V, nheads, γs)
    function linatt_pullback(ȳ)
        f̄ = NoTangent()
        Q̄, K̄, V̄, γ̄s = ∂linear_attention(ȳ, Q, K, V, nheads, γs)
        return f̄, Q̄, K̄, V̄, γ̄s, NoTangent()
    end
    return y, linatt_pullback
end

function ChainRulesCore.rrule(::typeof(linear_attention), Q, K, V, nheads, γs)
    y = linear_attention_forloop(Q, K, V, γs, nheads)
    function linatt_pullback(ȳ)
        f̄ = NoTangent()
        Q̄, K̄, V̄, γ̄s = ∂linear_attention(ȳ, Q, K, V, γs, nheads)
        return f̄, Q̄, K̄, V̄, γ̄s, NoTangent()
    end
    return y, linatt_pullback
end
