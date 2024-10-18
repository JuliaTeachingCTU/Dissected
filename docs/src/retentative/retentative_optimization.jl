# Title: Optimizing the implementation of the Retentative Layer
#
# We start by copying the implementation of prerequisities and start by adding support of multiple heads. 
# Prerequisities:
using LinearAlgebra

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

#
# The Retentative Layer with multiple heads differs from the scalar version by the parameter
# `γ`, which will be now `Vector` instead of the scalar. There is one scalar `γ` for each head.
# And for simplicity, we assume that the heads corresponds to continuous ranges, therefore
# the number of heads (and the length of `γ`) determine their dimensions.
struct RetentionLayer{G,Θ,M}
    θₐ::Θ
    Wk::M
    Wq::M
    Wv::M
    γ::G
end

function RetentionLayer(input_dim, hidden_dim, output_dim; nheads = 1, γ = 0.99, T = Float32)
    θₐ = T.(2π .* rand(hidden_dim ÷ 2))
    Wk = randn(T, hidden_dim, input_dim)
    Wq = randn(T, hidden_dim, input_dim)
    Wv = randn(T, output_dim, input_dim)
    nheads = max(nheads, length(γ))
    if nheads == 1
        iseven(hidden_dim) || error("hidden_dim dimension has to be even")
        RetentionLayer(θₐ, Wk, Wq, Wv, T(only(γ)))
    else
        if length(γ) != nheads
            length(γ) > 1 && @info "The length of `γ` is not equal to the number of heads, using default setting `1 .- exp.(range(-log(512),-log(32), length = nheads))`"
            γ = 1 .- exp.(range(-log(512),-log(32), length = nheads))
        end
        head_dim = size(layer.Wk, 1) ÷ length(layer.γ)
        head_dim * length(layer.γ) != size(layer.Wk, 1) && error("The number of heads does not divide the hidden dimension")
        iseven(head_dim) || error("head_dim dimension has to be even")
        RetentionLayer(θₐ, Wk, Wq, Wv, T.(γ))
    end
end

Base.show(io::IO, m::RetentionLayer) = print(io, "RetentionLayer($(size(m.Wk,1)) => $(size(m.Wk,2)))")


# In the first naive implementation, we just run `nheads` parralel instances of the single head version
# and concatenate them on the end.

function recursive_forward(layer::RetentionLayer, x)
    recursive_forward(layer.Wq, layer.Wk, layer.Wv, layer.γ, layer.θₐ, x)
end

function recursive_forward(Wq, Wk, Wv, γs::AbstractVector{<:Real}, θₐ, x)
    head_dim = size(Wk, 1) ÷ length(γs)
    θ_dim = head_dim ÷ 2
    slices = map(enumerate(γs)) do (i, γ)
        slice = (i-1)*head_dim+1:i*head_dim
        θᵢ = θₐ[(i-1)*θ_dim+1:i*θ_dim]
        recursive_forward(Wq[slice,:], Wk[slice,:], Wv[slice,:], γ, θᵢ, x)
    end
    reduce(vcat, slices)
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


function batch_forward(layer::RetentionLayer, x)
    Q = rotary_embedding(layer.Wq * x, -layer.θₐ)
    K = rotary_embedding(layer.Wk * x, -layer.θₐ)
    v = layer.Wv * x
    linear_attention(Q, K, v, γ)
end

function  linear_attention(Q, K, v, γ::Real)
    D = DMatrix(γ, size(Q, 2));
    transpose(((Q' * K) .* D) * v')
end

function  linear_attention(Q, K, v, γs::Vector{<:Real})
    head_dim = size(layer.Wk, 1) ÷ length(γ)
    slices = map(zip(1:head_dim:size(Q, 1), γs)) do (i, γ)
        slice = i:i+head_dim-1
        linear_attention(Q[slice,:], K[slice,:], v[slice,:], γ)
    end
    reduce(vcat, slices)
end

function chunked_forward(layer::RetentionLayer, x; chunk_size = 64)
    T = eltype(layer.Wq)
    Q = rotary_embedding(layer.Wq * x, -layer.θₐ)
    K = rotary_embedding(layer.Wk * x, -layer.θₐ)
    v = layer.Wv * x
    γ = layer.γ
    R₀ = zeros(T, size(layer.Wk, 1), size(layer.Wv, 1))
    os = map(1:chunk_size:size(v,2)) do n₀
        n₁ = min(n₀ + chunk_size - 1, size(v,2))
        Qᵢ = Q[:, n₀:n₁]
        Kᵢ = K[:, n₀:n₁]
        vᵢ = v[:, n₀:n₁]
        Dᵢ = DMatrix(γ, size(Qᵢ, 2));
        Rᵢ = sum(γ^(-m) * K[:,m] * v[:,m]' for m in 1:n₀-1; init = R₀)
        oᵢ = transpose(((Qᵢ' * Kᵢ) .* Dᵢ) * vᵢ') .+ γ .^ (n₀:n₁)'  .* (Rᵢ' * Qᵢ)
    end
    reduce(hcat, os)
end



x = randn(Float32, 32, 512)
layer = RetentionLayer(32, 32, 32; nheads = 2)
layer1 = RetentionLayer(layer.θₐ[1:8], layer.Wk[1:16,:], layer.Wq[1:16,:], layer.Wv[1:16,:], layer.γ[1])
layer2 = RetentionLayer(layer.θₐ[9:end], layer.Wk[17:end,:], layer.Wq[17:end,:], layer.Wv[17:end,:], layer.γ[2])

recursive_forward(layer1, x) ≈ batch_forward(layer1,x)
recursive_forward(layer2, x) ≈ batch_forward(layer2,x)

o = hcat()

recursive_forward(layer, x) ≈ batch_forward(layer,x)
recursive_forward(layer, x) ≈ chunked_forward(layer,x)


# # ### Decresing the memory footprint
# # 
# # The `batch_forward` has quite high memory footprint, because of the outer
# # product `(Q * V ).* D`. We can decrease the memory footprint using the 
# # map
# function batch_forward2(layer::RetentionLayer, x)
#     Q = rotary_embedding(layer.Wq * x, -layer.θₐ)
#     K = rotary_embedding(layer.Wk * x, -layer.θₐ)
#     v = layer.Wv * x
#     γ = layer.γ
#     linear_attention(Q, K, v, γ)
# end

# function linear_attention(Q, K, v, γ)
#     o = map(1:length(v)) do n 
#         @inbounds sum(only( γ ^(n - m) * dot(view(Q, :, n), view(K, :, m)) * v[m]) for m in 1:n)
#     end
#     o'
# end


# layer = RetentionLayer(32, 32, 32)
# x = randn(Float32, 32, 512)

# batch_forward(layer, x) ≈ batch_forward2(layer,x)

# @benchmark batch_forward(layer, x)
# @benchmark batch_forward2(layer,x)

# # The `batch_forward2` is slower, but has much lower memory footprint. 
# # This is because the `batch_forward` uses multi-threadding, while 
# # `batch_forward2` uses only one thread. Let's try multi-threadding

# function linear_attention(Q, K, v, γ)
#     o = zeros(eltype(v), 1, size(x,2))
#     Threads.@threads :dynamic for n in axes(v,2)
#         @inbounds for m in 1:n
#             o[n] += only( γ ^(n - m) * dot(view(Q, :, n), view(K, :, m)) * v[m])
#         end
#     end
#     o
# end

# @benchmark batch_forward3(layer,x)

# # With the implementation is equally fast as the `batch_forward`, 
# # but with much lower memory footprint.


