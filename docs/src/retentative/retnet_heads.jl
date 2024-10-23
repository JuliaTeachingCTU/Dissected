# # Adding more heads to retnet
# 
# We start by copying the prerequisities from the last time, mainly the `rot_matrix`, `rotary_embedding`
# and decayed masking matrix `DMatrix`.
using LinearAlgebra

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


# The type defining the Retentanive layer for multiple heads can be left as is. The only 
# difference between single-head and multiple-heads is that the `γ` is a vector 
# instead of a scalar. Thought the constructor is much-more sophisticated, since 
# we need to make sure that the number of heads divides the hidden dimension and dimension
# of the head is even.

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
       RetentionLayer(θₐ, Wk, Wq, Wv, T(only(γ)))
   else
       if length(γ) != nheads
           length(γ) > 1 && @info "The length of `γ` is not equal to the number of heads, using default setting `1 .- exp.(range(-log(512),-log(32), length = nheads))`"
           γ = 1 .- exp.(range(-log(512),-log(32), length = nheads))
       end
       head_dim = size(Wk, 1) ÷ length(γ)
       head_dim * length(γ) != size(Wk, 1) && error("The number of heads does not divide the hidden dimension")
       RetentionLayer(θₐ, Wk, Wq, Wv, T.(γ))
   end
end

Base.show(io::IO, m::RetentionLayer) = print(io, "RetentionLayer($(size(m.Wk,2)) => $(size(m.Wk,1))) with $(length(m.γ)) heads")

function recursive_forward(layer::RetentionLayer{<:Real}, x)
    recursive_forward(layer.Wq, layer.Wk, layer.Wv, layer.γ, layer.θₐ, x)
end


# The retention layer for multiple heads behave like a stacked single layers. So the first 
# implementaion will behave like that. There will be space for improvements, but let's make the
# implementations correct first and then make them fast. 

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

function recursive_forward(layer::RetentionLayer{<:Vector}, x)
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
    v = layer.Wv * x
    linear_attention(Q, K, v, layer.γ, chunk_size)
end

function linear_attention(Q, K, V, γs::Vector{<:Real}, chunk_size)
    kvslices, oslices = _slices(Q, K, γs)
    os = map(enumerate(zip(γs, kvslices, oslices))) do (i, (γ, kvslice, oslice))
        linear_attention(Q[kvslice,:], K[kvslice,:], V[oslice,:], γ, chunk_size)
    end
    reduce(vcat, os)
end

function linear_attention(Q, K, V, γ::Real, chunk_size)
    chunk_size < size(V,2) && return(_chunked_linear_attention(Q, K, V, γ, chunk_size))
    _linear_attention(Q, K, V, γ)
end

function _linear_attention(Q, K, V, γ::Real)
    D = DMatrix(γ, size(Q, 2));
    transpose(((Q' * K) .* D) * V')
end

function _chunked_linear_attention(Q, K, V, γ::Real, chunk_size)
    R₀ = zeros(eltype(V), size(K, 1), size(V, 1))
    os = map(1:chunk_size:size(V,2)) do n₀
        n₁ = min(n₀ + chunk_size - 1, size(V,2))
        Qᵢ = Q[:, n₀:n₁]
        Kᵢ = K[:, n₀:n₁]
        Vᵢ = V[:, n₀:n₁]
        Rᵢ = sum(γ^(-m) * K[:,m] * V[:,m]' for m in 1:n₀-1; init = R₀)
        oᵢ = _linear_attention(Qᵢ, Kᵢ, Vᵢ, γ) .+ γ .^ (n₀:n₁)'  .* (Rᵢ' * Qᵢ)
    end
    reduce(hcat, os)
end



#
# Now we test that all implementations matches the naive approach where layers are run 
# completely in parallel.
#

x = randn(Float32, 8, 257)
layer = RetentionLayer(8, 8, 8; nheads = 2)
head_dim = size(layer.Wk,1) ÷ length(layer.γ)
odim = size(layer.Wv, 1) ÷ length(layer.γ)
θ_dim = head_dim ÷ 2
layer1 = RetentionLayer(layer.θₐ[1:θ_dim], layer.Wk[1:head_dim,:], layer.Wq[1:head_dim,:], layer.Wv[1:odim,:], layer.γ[1])
layer2 = RetentionLayer(layer.θₐ[θ_dim+1:end], layer.Wk[head_dim+1:end,:], layer.Wq[head_dim+1:end,:], layer.Wv[odim+1:end,:], layer.γ[2])

ref_o = vcat(recursive_forward(layer1, x), recursive_forward(layer2, x))
recursive_forward(layer, x) ≈ ref_o
batch_forward(layer, x) ≈ ref_o
batch_forward(layer, x;chunk_size = 16) ≈ ref_o
batch_forward(layer, x;chunk_size = 64) ≈ ref_o
recursive_forward(layer, x) ≈ batch_forward(layer, x)


nheads = 4
head_dim = 32
hidden_dim = head_dim*nheads
layer = RetentionLayer(hidden_dim, hidden_dim, hidden_dim; nheads)
θ_dim = head_dim ÷ 2

layers = map(zip(1:head_dim:hidden_dim, 1:θ_dim:length(layer.θₐ), layer.γ)) do (i,j,γ)
    ii = i:i+head_dim-1
    jj = j:j+θ_dim-1
    RetentionLayer(layer.θₐ[jj], layer.Wk[ii,:], layer.Wq[ii,:], layer.Wv[ii,:], γ)
end

x = rand(Float32, hidden_dim, 257)
ref_o = vcat(map(l -> recursive_forward(l, x), layers)...)

recursive_forward(layer, x) ≈ ref_o
batch_forward(layer, x) ≈ ref_o
batch_forward(layer, x;chunk_size = 16) ≈ ref_o
batch_forward(layer, x;chunk_size = 64) ≈ ref_o
recursive_forward(layer, x) ≈ batch_forward(layer, x)




layer = RetentionLayer(32, 32, 32; nheads = 4)
x = randn(32, 128)
println("maxdiff: ", maximum(abs.(batch_forward(layer, x; chunk_size = 16) .- batch_forward(layer, x))))
batch_forward(layer, x; chunk_size = 16) ≈ batch_forward(layer, x)

