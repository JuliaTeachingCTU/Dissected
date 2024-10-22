# # Adding more heads to retnet
# 
# We start by copying the prerequisities from the last time, mainly the `rot_matrix`, `rotary_embedding`
# and decayed masking matrix `DMatrix`.
using LinearAlgebra
using Test
using FiniteDifferences
using ChainRulesCore
using Zygote
using Flux

#
struct DMatrix{T} <:AbstractMatrix{T}
    γ::T
    size::Int64
end

Base.getindex(d::DMatrix{T}, n::Integer, m::Integer) where {T} = n ≥ m ? d.γ^(n-m) : zero(T)
Base.size(d::DMatrix,i::Integer) = d.size
Base.size(d::DMatrix) = (d.size, d.size)

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

# The type defining the Retentanive layer for multiple heads can be left as is. The only 
# difference between single-head and multiple-heads is that the `γ` is a vector 
# instead of a scalar. Thought the constructor is much-more sophisticated, since 
# we need to make sure that the number of heads divides the hidden dimension and dimension
# of the head is even.

struct RetentionLayer{N,G,Θ,M}
    θₐ::Θ
    Wk::M
    Wq::M
    Wv::M
    γ::G
    nheads::Val{N}
end

Flux.@layer RetentionLayer

function RetentionLayer(input_dim, hidden_dim, output_dim; nheads = 1, γ = 0.99, T = Float32)
   θₐ = T.(2π .* rand(hidden_dim ÷ 2))
   Wk = randn(T, hidden_dim, input_dim)
   Wq = randn(T, hidden_dim, input_dim)
   Wv = randn(T, output_dim, input_dim)
   nheads = max(nheads, length(γ))
   if nheads == 1
       RetentionLayer(θₐ, Wk, Wq, Wv, T(only(γ)), Val(1))
   else
       if length(γ) != nheads
           length(γ) > 1 && @info "The length of `γ` is not equal to the number of heads, using default setting `1 .- exp.(range(-log(512),-log(32), length = nheads))`"
           γ = 1 .- exp.(range(-log(512),-log(32), length = nheads))
       end
       head_dim = size(Wk, 1) ÷ length(γ)
       head_dim * length(γ) != size(Wk, 1) && error("The number of heads does not divide the hidden dimension")
       RetentionLayer(θₐ, Wk, Wq, Wv, T.(γ), Val(nheads))
   end
end

Base.show(io::IO, m::RetentionLayer{N}) where {N} = print(io, "RetentionLayer($(size(m.Wk,2)) => $(size(m.Wk,1))) with $(N) heads")

function recursive_forward(layer::RetentionLayer{1,<:Real}, x)
    recursive_forward(layer.Wq, layer.Wk, layer.Wv, layer.γ, layer.θₐ, x)
end


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

function _slices(K::AbstractMatrix, V::AbstractMatrix, γs::AbstractVector)
    _slices(size(K,1), size(V,1), Val(length(γs)))
end

function _slices(layer::RetentionLayer)
    _slices(size(layer.Wk,1), size(layer.Wv,1), layer.nheads)
end

function recursive_forward(layer::RetentionLayer{N,<:AbstractVector}, x) where {N}
    Wk, Wq, Wv, θₐ, γs = layer.Wk, layer.Wq, layer.Wv, layer.θₐ, layer.γ
    kvslices, oslices = _slices(layer)
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
    v = layer.Wv * x
    linear_attention(Q, K, v, layer.γ, chunk_size)
end

function linear_attention(Q, K, V, γ, chunk_size)
    chunk_size < size(V,2) && return(_chunked_linear_attention(Q, K, V, γ, chunk_size))
    linear_attention(Q, K, V, γ)
end

function linear_attention(Q, K, V, γ::Real)
    D = DMatrix(γ, size(Q, 2));
    transpose(((Q' * K) .* D) * V')
end

function linear_attention(Q, K, V, γs::AbstractVector{<:Real})
    kvslices, oslices = _slices(Q, V, γs)
    os = map(enumerate(zip(γs, kvslices, oslices))) do (i, (γ, kvslice, oslice))
        linear_attention(Q[kvslice,:], K[kvslice,:], V[oslice,:], γ)
    end
    reduce(vcat, os)
end

# function _chunked_linear_attention(Q, K, V, γ::Real, chunk_size)
#     R₀ = zeros(eltype(V), size(K, 1), size(V, 1))
#     os = map(1:chunk_size:size(V,2)) do n₀
#         n₁ = min(n₀ + chunk_size - 1, size(V,2))
#         Qᵢ = Q[:, n₀:n₁]
#         Kᵢ = K[:, n₀:n₁]
#         Vᵢ = V[:, n₀:n₁]
#         Rᵢ = sum(γ^(-m) * K[:,m] * V[:,m]' for m in 1:n₀-1; init = R₀)
#         oᵢ = linear_attention(Qᵢ, Kᵢ, Vᵢ, γ) .+ γ .^ (n₀:n₁)'  .* (Rᵢ' * Qᵢ)
#     end
#     reduce(hcat, os)
# end


function _chunked_linear_attention(Q, K, V, γ, chunk_size)
    os = map(1:chunk_size:size(V,2)) do n₀
        n₁ = min(n₀ + chunk_size - 1, size(V,2))
        Qᵢ = Q[:, n₀:n₁]
        Kᵢ = K[:, n₀:n₁]
        Vᵢ = V[:, n₀:n₁]
        CCᵢ = cross_retention(Qᵢ, K, V, γ, n₀, n₁)
        oᵢ = linear_attention(Qᵢ, Kᵢ, Vᵢ, γ) .+ CCᵢ
    end
    reduce(hcat, os)
end

function cross_retention(Qᵢ, K, V, γ::Real, n₀, n₁)
    n₀ ≤ 1 && return(zeros(eltype(V), size(V, 1), n₁ - n₀ + 1))
    Rᵢ = sum(γ^(-m) * K[:,m] * V[:,m]' for m in 1:n₀-1)
    CCᵢ = γ .^ (n₀:n₁)'  .* (Rᵢ' * Qᵢ)
    CCᵢ
end

function cross_retention(Qᵢ, K, V, γₛ::AbstractVector{<:Real}, n₀, n₁)
    n₀ ≤ 1 && return(zeros(eltype(V), size(V, 1), n₁ - n₀ + 1))
    kvslices, oslices = _slices(K, V, γₛ)
    CCᵢ =map(zip(kvslices, oslices, γₛ)) do (kvslice, oslice, γ)
        Rᵢ = sum(γ^(-m) * K[kvslice,m] * V[oslice,m]' for m in 1:n₀-1)
        γ .^ (n₀:n₁)'  .* (Rᵢ' * Qᵢ[kvslice,:])
    end
    reduce(vcat, CCᵢ)
end

function _linear_attention_forloop(Q, K, V, γs::AbstractVector)
    _linear_attention_forloop(Q, K, V, nheads, Val(length(γs)), γs)
end

function _linear_attention_forloop(Q, K, V, nheads, γs)
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

function ∂linear_attention(ȳ, Q, K, V, γs::AbstractVector{<:Number})
    ∂linear_attention(ȳ, Q, K, V, γs, Val(length(γs)))
end

function ∂linear_attention(ȳ, Q, K, V, γs::AbstractVector{<:Number}, nheads::Val)
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

function ∂linear_attention(ȳ, Q, K, V, γ::Real)
    Q̄, K̄, V̄, γ̄s = ∂linear_attention(ȳ, Q, K, V, [γ])
    Q̄, K̄, V̄, only(γ̄s)
end

function ChainRulesCore.rrule(::typeof(linear_attention), Q, K, V, γs, chunk_size)
    y = linear_attention(Q, K, V, γs, chunk_size)
    function linatt_pullback(ȳ)
        f̄ = NoTangent()
        Q̄, K̄, V̄, γ̄s = ∂linear_attention(ȳ, Q, K, V, γs)
        return f̄, Q̄, K̄, V̄, γ̄s, NoTangent()
    end
    return y, linatt_pullback
end

@testset "Gradient of linear attention" begin 
    Q = randn(8,127);
    K = randn(8,127);
    V = randn(6,127);
    @testset "γ = $(γ)" for γ in (0.95, [0.99, 0.95], rand(4))
        γ = 0.95;
        @testset "chunk_size = $(chunk_size)" for chunk_size in [32, 64, 128]
            @test gradient(Q -> sum(linear_attention(Q, K, V, γ, chunk_size)), Q)[1] ≈ grad(central_fdm(5,1), Q -> sum(linear_attention(Q, K, V, γ, chunk_size)), Q)[1]
            @test gradient(K -> sum(linear_attention(Q, K, V, γ, chunk_size)), K)[1] ≈ grad(central_fdm(5,1), K -> sum(linear_attention(Q, K, V, γ, chunk_size)), K)[1]
            @test gradient(V -> sum(linear_attention(Q, K, V, γ, chunk_size)), V)[1] ≈ grad(central_fdm(5,1), V -> sum(linear_attention(Q, K, V, γ, chunk_size)), V)[1]
            @test gradient(γ -> sum(linear_attention(Q, K, V, γ, chunk_size)), γ)[1] ≈ grad(central_fdm(5,1), γ -> sum(linear_attention(Q, K, V, γ, chunk_size)), γ)[1]
        end
    end
end


@testset "RetNet" begin 
    @testset "correctness of forward pass $(nheads)" for nheads in [1,2,3,4]
        head_dim = rand([2,4,8])
        odim = nheads * rand([2,4,8])
        hidden_dim = head_dim*nheads
        layer = RetentionLayer(hidden_dim, hidden_dim, odim; nheads)
        θ_dim = head_dim ÷ 2

        layers = map(zip(1:head_dim:hidden_dim, 1:θ_dim:length(layer.θₐ), 1:(odim÷nheads):odim, layer.γ)) do (i,j,o,γ)
            ii = i:i+head_dim-1
            jj = j:j+θ_dim-1
            oo = o:(o+odim ÷ nheads -1)
            RetentionLayer(layer.θₐ[jj], layer.Wk[ii,:], layer.Wq[ii,:], layer.Wv[oo,:], γ, Val(1))
        end

        x = rand(Float32, hidden_dim, 257)
        ref_o = vcat(map(l -> recursive_forward(l, x), layers)...)

        @test recursive_forward(layer, x) ≈ ref_o
        @test batch_forward(layer, x) ≈ ref_o
        @test batch_forward(layer, x;chunk_size = 16) ≈ ref_o
        @test batch_forward(layer, x;chunk_size = 64) ≈ ref_o
        @test recursive_forward(layer, x) ≈ batch_forward(layer, x)
    end

    @testset "correctness of forward pass $(nheads)" for nheads in [1,2,3,4]
        head_dim = rand([2,4,8])
        odim = nheads * rand([2,4,8])
        hidden_dim = head_dim*nheads
        layer = RetentionLayer(hidden_dim, hidden_dim, odim; nheads)
        x = rand(Float32, hidden_dim, 257)
    
        gradient(layer -> sum(batch_forward(layer, x)), layer)
        gradient(layer -> sum(batch_forward(layer, x;chunk_size = 16)), layer)
    end
end