# # Adding more heads to retnet
# 
# We start by copying the prerequisities from the last time, mainly the `rot_matrix`, `rotary_embedding`
# and decayed masking matrix `DMatrix`.
using LinearAlgebra
using FiniteDifferences
using ChainRulesCore
using Zygote
using Flux

include("rotary_embedding.jl")
include("linear_attention.jl")

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

function _slices(layer::RetentionLayer)
    _slices(size(layer.Wk,1), size(layer.Wv,1), layer.nheads)
end

"""
    recursive_forward(layer::RetentionLayer{1,<:Real}, x)

    Compute the forward pass using the recurrent formulation, which
    is good for inference but bad for training.
"""
function recursive_forward(layer::RetentionLayer{1,<:Real}, x)
    recursive_forward(layer.Wq, layer.Wk, layer.Wv, layer.γ, layer.θₐ, x)
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

"""
    batch_forward(layer::RetentionLayer, x)

"""
function batch_forward(layer::RetentionLayer, x; chunk_size = typemax(Int64))
    Q = rotary_embedding(layer.Wq * x, -layer.θₐ)
    K = rotary_embedding(layer.Wk * x, -layer.θₐ)
    v = layer.Wv * x
    linear_attention(Q, K, v, layer.γ)
end


function chunk_forward(layer::RetentionLayer, x; chunk_size = 64)
    Q = rotary_embedding(layer.Wq * x, -layer.θₐ)
    K = rotary_embedding(layer.Wk * x, -layer.θₐ)
    V = layer.Wv * x
    rep_γs = repeat_gamma(layer)
    Rᵢ = cross_retention(K, V, layer.γ, 0, layer.nheads)
    os = map(1:chunk_size:size(V,2)) do n₀
        Bᵢ = n₀:min(n₀ + chunk_size - 1, size(V,2))
        Bᵢ₁ = max(1, (n₀ - chunk_size)):(n₀-1)
        Qᵢ = Q[:, Bᵢ]
        Kᵢ = K[:, Bᵢ]
        Vᵢ = V[:, Bᵢ]
        Rᵢ += cross_retention(K, V, layer.γ, max(1, (n₀ - chunk_size)):(n₀-1), layer.nheads)
        oᵢ = linear_attention(Qᵢ, Kᵢ, Vᵢ, layer.γ) .+ rep_γs .^ (Bᵢ)'  .* (Rᵢ' * Qᵢ)
    end
    reduce(hcat, os)
end

repeat_gamma(layer::RetentionLayer{1}) = layer.γ
repeat_gamma(layer::RetentionLayer{N,<:AbstractVector}) where {N} = repeat(layer.γ, inner = size(layer.Wv,1) ÷ N)

# function cross_retention(K, V, γ::Real, n₀)
#     sum(γ^(-m) * K[:,m] * V[:,m]' for m in 1:n₀-1)
# end

function cross_retention(K, V, γ::Real, n₀, nheads::Val{1})
    cross_retention(K, V, [γ], n₀, nheads)
end

function cross_retention(K, V, γs::AbstractVector{<:Real}, n₀::Integer, nheads)
    cross_retention(K, V, γs, Base.OneTo(n₀-1), nheads)
end
function cross_retention(K, V, γs::AbstractVector{<:Real}, interval::AbstractUnitRange, nheads::Val{N}) where {N}
    kvslices, oslices = _slices(K, V, nheads)
    Rᵢ = zeros(eltype(V), size(K, 1), size(V, 1))
    for m in interval
        γₘₛ = γs .^ (-m)
        for (γₘ, kvslice, oslice) in zip(γₘₛ, kvslices, oslices)
            for i in kvslice
                for j in oslice
                    Rᵢ[i,j] += γₘ * K[i,m] * V[j,m]
                end
            end
        end
    end
    Rᵢ
end

function _∂cross_retention(ȳ, K, V, γs::AbstractVector{<:Real}, interval::AbstractUnitRange, nheads::Val{N}) where {N}
    kvslices, oslices = _slices(K, V, nheads)
    Rᵢ = zeros(eltype(V), size(K, 1), size(V, 1))
    K̄ = similar(K)
    V̄ = similar(V)
    γ̄s = similar(γs)
    K̄ .= 0
    V̄ .= 0
    γ̄s .= 0
    for m in interval
        for (k, (γᵢ, kvslice, oslice)) in enumerate(zip(γs, kvslices, oslices))
            γₘ = γᵢ .^ (-m)
            ∂γₘ = zero(γₘ)
            for i in kvslice
                for j in oslice
                    K̄[i,m] += γₘ * ȳ[i,j] * V[j,m]
                    V̄[j,m] += γₘ * K[i,m] * ȳ[i,j]
                    ∂γₘ += ȳ[i,j] * K[i,m] * V[j,m]
                end
            end
            γ̄s[k] += (-m)*γᵢ^(-m-1) * ∂γₘ
        end
    end
    return(K̄, V̄, γ̄s)
end

function ChainRulesCore.rrule(::typeof(cross_retention), K, V, γs::AbstractVector{<:Real}, interval::AbstractUnitRange, nheads::Val)
    y = cross_retention(K, V, γs, interval, nheads)
    function linatt_pullback(ȳ)
        f̄ = NoTangent()
        K̄, V̄, γ̄s = _∂cross_retention(ȳ, K, V, γs, interval, nheads) 
        return f̄, K̄, V̄, γ̄s, NoTangent(), NoTangent()
    end
    return y, linatt_pullback
end

function ChainRulesCore.rrule(::typeof(cross_retention), K, V, γ::Real, n₀::AbstractUnitRange, nheads::Val{1})
    y = cross_retention(K, V, γ, n₀, nheads)
    function linatt_pullback(ȳ)
        f̄ = NoTangent()
        K̄, V̄, γ̄s = _∂cross_retention(ȳ, K, V, [γ], n₀::AbstractUnitRange, nheads) 
        return f̄, K̄, V̄, only(γ̄s), NoTangent(), NoTangent()
    end
    return y, linatt_pullback
end


