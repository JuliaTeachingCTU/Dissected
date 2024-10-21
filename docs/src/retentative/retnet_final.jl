using ChainRulesCore
using FiniteDifferences
using Zygote

struct DMatrix{T} <:AbstractMatrix{T}
    γ::T
    size::Int64
end

Base.getindex(d::DMatrix{T}, n::Integer, m::Integer) where {T} = n ≥ m ? d.γ^(n-m) : zero(T)
Base.size(d::DMatrix,i::Integer) = d.size
Base.size(d::DMatrix) = (d.size, d.size)


# Define the rotary ebedding and its gradient
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


function linear_attention_single_thread(Q, K, V, nheads, γs)
    kvslices, oslices = _slices(K, V, nheads)
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

function linear_attention_multi_thread(Q, K, V, nheads, γs)
    kvslices, oslices = _slices(K, V, nheads)
    o = zeros(eltype(V), size(V))
    l = size(K,2)
    Threads.@threads :static for i in 1:Threads.nthreads()
        for n in i:Threads.nthreads():l
            for (j, (kvslice, oslice)) in enumerate(zip(kvslices, oslices))
                @inbounds for m in 1:n
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
    end
    o
end

function linear_attention(Q, K, V, nheads, γs)
    if Threads.nthreads() == 1 
        linear_attention_single_thread(Q, K, V, nheads, γs)
    else
        linear_attention_multi_thread(Q, K, V, nheads, γs)
    end
end

function ∂linear_attention(ȳ, Q, K, V, nheads, γs::AbstractVector{<:Number})
    T = eltype(Q)
    Q̄ = similar(Q)
    K̄ = similar(K)
    V̄ = similar(V)
    γ̄s = similar(γs)
    Q̄ .= 0
    K̄ .= 0
    V̄ .= 0
    γ̄s .= 0
    kvslices, oslices = _slices(K,V,nheads)
    l = size(K,2)
    for n in 1:l
        for (j, (kvslice, oslice)) in enumerate(zip(kvslices, oslices))
            for m in 1:n
                ## we need to recompute the alpha for the gradient
                α = zero(T)
                for k in kvslice
                    α += Q[k, n] * K[k, m]
                end

                ## then we compute the gradient of the output with respect to α and γ
                γ = γs[j]^(n-m)
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

                ## and finally we updaate the gradient of γ
                γ̄s[j] += n != m ? (n-m)*γs[j]^(n-m-1)*∂γ : zero(T)
            end
        end
    end
    return(Q̄, K̄, V̄, γ̄s)
end

function ChainRulesCore.rrule(::typeof(linear_attention), Q, K, V, nheads, γs::AbstractVector{<:Number})
    y = linear_attention(Q,K,V,γs)
    function linatt_pullback(ȳ)
        f̄ = NoTangent()
        Q̄, K̄, V̄, γ̄s = ∂linear_attention(ȳ, Q, K, V, nheads, γs)
        return f̄, Q̄, K̄, V̄, γ̄s
    end
    return y, linatt_pullback
end



# Retention Layer
struct RetentionLayer{N, G<:AbstractVector,Θ<:AbstractVector,M<:AbstractMatrix}
    θₐ::Θ
    Wk::M
    Wq::M
    Wv::M
    γ::G
    nheads::Val{N}
end

function RetentionLayer(input_dim, hidden_dim, output_dim; nheads = 1, T = Float32)
   θₐ = T.(2π .* rand(hidden_dim ÷ 2))
   Wk = randn(T, hidden_dim, input_dim)
   Wq = randn(T, hidden_dim, input_dim)
   Wv = randn(T, output_dim, input_dim)

   γs = (nheads > 1) ? 1 .- exp.(range(-log(512),-log(32), length = nheads)) : [1 - exp(-log(512.0))]
   mod(size(Wk, 1), nheads) != 0 && error("The number of heads does not divide the hidden dimension")
   mod(size(Wv, 1), nheads) != 0 && error("The number of heads does not divide the output dimension")
   RetentionLayer(θₐ, Wk, Wq, Wv, T.(γs),Val(nheads))
end

Base.show(io::IO, m::RetentionLayer{N}) where {N} = print(io, "RetentionLayer($(size(m.Wk,2)) => $(size(m.Wk,1))) with $(N) heads")


# The retention layer for multiple heads behave like a stacked single layers. So the first 
# implementaion will behave like that. There will be space for improvements, but let's make the
# implementations correct first and then make them fast. 

function _slices(hidden_dim::Integer, output_dim::Integer, nheads::Val{N}) where {N}
    head_dim = hidden_dim ÷ N
    odim = output_dim ÷ N
    kvslices = ntuple(i -> (i-1)*head_dim+1:i*head_dim, nheads)
    oslices = ntuple(i -> (i-1)*odim+1:i*odim, nheads)
    return(kvslices, oslices)
end

_slices(layer::RetentionLayer) = _slices(size(layer.Wk,1), size(layer.Wv,1), layer.nheads)
_slices(K::AbstractMatrix, V::AbstractMatrix, nheads) = _slices(size(K,1), size(V,1), layer.nheads)

function recursive_forward(layer::RetentionLayer, x)
    Wk, Wq, Wv, θₐ, γs = layer.Wk, layer.Wq, layer.Wv, layer.θₐ, layer.γ
    kvslices, oslices = _slices(Wk, Wv, γs)
    θ_dim = length(first(kvslices)) ÷ 2
    os = map(enumerate(zip(γs, kvslices, oslices))) do (i, (γ, kvslice, oslice))
        θᵢ = θₐ[(i-1)*θ_dim+1:i*θ_dim]
        recursive_forward(Wq[kvslice,:], Wk[kvslice,:], Wv[oslice,:], γ, θᵢ, x)
    end
    reduce(vcat, os)
end

function recursive_forward(Wq, Wk, Wv, γ::Real, θₐ, x)
   T = eltype(x)
   s₀ = zeros(T, size(Wk, 1), size(Wv, 1))
   o₀ = zeros(T, size(Wv, 1))
   A = rot_matrix(θₐ)
   function _step((on, sn), xᵢ)
       Kᵢ = Wk * xᵢ
       Qᵢ = Wq * xᵢ
       vᵢ = Wv * xᵢ
       snn = A * sn + Kᵢ .* vᵢ'
       on = Qᵢ' * snn
       return(on', γ .* snn)
   end
   os = accumulate(_step, eachslice(x, dims = 2), init = (o₀, s₀))
   reduce(hcat, map(first, os))
end



# We implement the `batch_forward` in the similar spirit, but we make a slight abstraction
# that we dynamically switch between batch-mode and chunked-mode based on chunk_size.


function batch_forward(layer::RetentionLayer, x; chunk_size = typemax(Int64))
    Q = rotary_embedding(layer.Wq * x, -layer.θₐ)
    K = rotary_embedding(layer.Wk * x, -layer.θₐ)
    V = layer.Wv * x
    linear_attention(Q, K, V, layer.nheads, layer.γ, chunk_size)
end

function linear_attention(Q, K, V, nheads, γs, chunk_size)
    if chunk_size == typemax(Int64)
        linear_attention(Q, K, V, nheads, γs)
    else
        chunked_linear_attention(Q, K, V, nheads, γs, chunk_size)
    end
end

function chunked_linear_attention(Q, K, V, nheads, γs, chunk_size)
    os = map(1:chunk_size:size(V,2)) do n₀
        n₁ = min(n₀ + chunk_size - 1, size(V,2))
        Qᵢ = Q[:, n₀:n₁]
        Kᵢ = K[:, n₀:n₁]
        Vᵢ = V[:, n₀:n₁]
        oᵢ = linear_attention(Qᵢ, Kᵢ, Vᵢ, nheads, γs) .+ from_previous_chunks(Qᵢ, K, V, nheads, γs, n₀, n₁)
    end
    reduce(hcat, os)
end

function from_previous_chunks(Qᵢ, K, V, nheads, γs::Vector{<:Real}, n₀, n₁)
    kvslices, oslices = _slices(K, V, nheads)
    parts = map(zip(kvslices, oslices, γs)) do (kvslice, oslice, γ)
        R₀ = zeros(eltype(V), length(kvslice), length(oslice))
        R = sum(γ^(-m) * view(K, kvslice, m) * view(V, oslice, m)' for m in 1:n₀-1; init = R₀)
        return(γ .^ (n₀:n₁)'  .* (R' * view(Qᵢ, kvslice, :)))
    end
    reduce(vcat, parts)
end



function batch_forward2(layer::RetentionLayer, x)
    Q = rotary_embedding(layer.Wq * x, -layer.θₐ)
    K = rotary_embedding(layer.Wk * x, -layer.θₐ)
    v = layer.Wv * x
    linear_attention2(Q, K, v, layer.γ)
end

function linear_attention2(Q, K, V, γs::Vector{<:Real})
    kvslices, oslices = _slices(Q, K, γs)
    os = map(enumerate(zip(γs, kvslices, oslices))) do (i, (γ, kvslice, oslice))
        linear_attention2(Q[kvslice,:], K[kvslice,:], V[oslice,:], γ)
    end
    reduce(vcat, os)
end

function linear_attention2(Q, K, V, γ::Real)
    _linear_attention2(Q, K, V, γ)
end

function _linear_attention2(Q, K, V, γ::Real)
    D = DMatrix(γ, size(Q, 2));
    transpose(((Q' * K) .* D) * V')
end


layer = RetentionLayer(32,32,32, nheads = 2)
x = randn(32, 127)

batch_forward(layer, x) ≈ recursive_forward(layer, x)
batch_forward2(layer, x) ≈ recursive_forward(layer, x)
batch_forward(layer, x) ≈ batch_forward2(layer, x)
