# This will implement Retentative neural networks, which was published in
# a paper with bold title: "Retentive Network: A Successor to Transformer for Large Language Models",
# at https://arxiv.org/pdf/2307.08621. The main advantage of the architecture
# is that it allows parallel training and efficient (recursive) inference.
# 
# The implementation will be intentionally simple to demonstrate the key concepts
# and especially verify author's understanding of the math.

using LinearAlgebra
using DataFrames

# The explanation of the model stats with state-space model assuming 
# the input `v` to be (projected to) 1d signal.
# The state of the model is updated as `sₙ₊₁ = A*sₙ + K*uₙ` and
# the output is as `oₙ = Q*sₙ`. 
# The model is threfore defined by matrices `A`, `K`, and `Q`.

struct Model{MA,MK,MQ}
    A::MA
    K::MK
    Q::MQ
end

v = sin.(-2π:0.1:2π)
hidden_dim = 8
A = 0.05f0 * randn(hidden_dim, hidden_dim)
K = 0.05f0 * randn(hidden_dim)    
Q = 0.05f0 * randn(1, hidden_dim)    
m = Model(A,K,Q)

# To make the prediction, the natural apprach is to just rewrite the above
# equations as:

function recursive_forward(m::Model, v)
    T = eltype(v)
    s = zeros(T, size(m.A,2))
    o = zero(T)
    os = T[]
    for vᵢ in v
        s = m.A*s + m.K .* vᵢ
        push!(os, only(m.Q * s))
    end
    os
end

# Or in a more functional style as

function recursive_forward2(m::Model, v)
    T = eltype(v)
    s₀ = zeros(size(m.A,2))
    o₀ = zero(T)
    function _step((sn, on), vn)
        snn = m.A*sn + m.K .* vn
        on = only(m.Q * snn)
        return(snn, on)
    end
    os = accumulate(_step, v, init = (s₀, o₀))
    [o[2] for o in os]
end

# For further derivation, the important is the method, 
# which does not use the inner state, but recomputes it from scratch,
# which is super inefficient.

function sum_method(m::Model, v)
    Q, A, K = m.Q, m.A, m.K
    map(1:length(v)) do n 
        sum(only(Q * A^(n-m) * K * v[m]) for m in 1:n)
    end
end

recursive_forward(m, v) ≈ sum_method(m, v)
recursive_forward(m, v) ≈ recursive_forward2(m, v)

# The sum method as-is very inefficient, but it can be optimized
# if the matrix `A` is diagonal. With that, the expensive matrix exponentiation
# `A^(n-m)` with be just exponentiation of scalar numbers. The paper defines the 
#  A to be diagonal with complex numbers on diagonal, which performs rotation. 
# Technically, the matrix `A` is said to be diagonalized, but the left and right matrices
# are absorbed by `Q` and `K`, hence having `A` to be diagonal. Furthermore, it is assumed
# that the diagonal elements have the absolute value 1 and there is a scalar rotation factor `γ`, i.e.
# A = γ * diag(exp(iθ₁), exp(iθ₂), ..., exp(iθ_d)), where θᵢ ∈ [-π, π]. The rotation is then performed by 

# With that, the sum method can be rewritten as
function sum_method2(m::Model, v, γ = one(eltype(v)))
    Q, A, K = m.Q, m.A, m.K
    o = zero(eltype(v))
    Qs = [Q * A^n for n in 1:length(v)]
    Ks = [A^(-m) * K for m in 1:length(v)]
    map(1:length(v)) do n 
        sum(only(γ ^(n -m) * Qs[n] * Ks[m] * v[m]) for m in 1:n)
    end
end

# True to the paper, there is one more tweak to make the model more expressive,
# but that is not that important for now, where we focus on having the recurrent
# part right. The paper points out with A being a rotation in complex plane, 
# the above uses rotation position embedding with damping factor γ as proposed in 
# A length-extrapolatable transformer, https://arxiv.org/pdf/2212.10554.

# A minor trick I do not understand how authors deal with complex numbers and they are not so 
# vocal about that. Anyway the important property of the matrix `A` are such that `A^(-1) = A'` 
# and the exponents can be computed efficiently, which is a property of real rotation matrices 
# used in the implementation. The real rotation matrix is defined as

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

# We can see that the rotation matrix with real properties has has the 
# properties we desire.
θ = randn(2)
rot_matrix(θ)*rot_matrix(θ) ≈ rot_matrix(2θ)
rot_matrix(θ)*rot_matrix(θ)' ≈ I
rot_matrix(θ)*rot_matrix(-θ) ≈ I

#Using the rotation matrix, we define a new model and verify that all identities hold.
hidden_dim = 8
A = rot_matrix(2π .* rand(hidden_dim ÷ 2))
K = 0.05 * randn(hidden_dim)    
Q = 0.05 * randn(1, hidden_dim)    
m = Model(A,K,Q)

sum_method(m, v) ≈ sum_method2(m, v)
recursive_forward(m, v) ≈ sum_method2(m, v)

# We can also try the equivalency with scaling factor, but since our tooling 
# is not exactly designed for this, it will be a bit pesky, as we use two versions
# of the model.
γ = 0.95
mγ = Model(γ .* A,K,Q)
maximum(sum_method(mγ, v) ≈ sum_method2(m, v, γ))
recursive_forward(mγ, v) ≈ sum_method2(m, v, γ)

# To make the model more expressive, Matrices `Q` and `K` can be made input-aware.
# That means assuming the input `x` is actually a sequence of vectors, we can define
# `Q` and `K` as functions of `x`. The model is then defined

x = reduce(vcat, [v' .+ randn(1, length(v)) for _ in 1:8])

input_dim = size(x,1)

m = (
    A = rot_matrix(2π .* rand(hidden_dim ÷ 2)),
    Wk = randn(hidden_dim, input_dim),
    Wq = randn(hidden_dim, input_dim),
    γ = 0.95,
    )

Qs = [(m.Wq * x[:, n])' * m.A^n for n in 1:length(v)]
Ks = [m.A^(-n) * m.Wk * x[:, n] for n in 1:length(v)]

o = map(1:length(v)) do n 
    sum(only(γ ^(n - m) * Qs[n] * Ks[m] * v[m]) for m in 1:n)
end

# The above is nice, as the `map` can be already run in parallel. There is no expensive
# matrix exponentiation, as we just exponentiate scalar.
# The paper shows a transformer-like formulation. For it, we define a mask matrix
# ``D`` with ``D_{ij} = γ^{(|i-j|)}`` if ``i ≥ j``. Then, the inference can be written in matrix
# forms as

struct DMatrix{T} <:AbstractMatrix{T}
    γ::T
    size::Int64
end

Base.getindex(d::DMatrix{T}, n::Integer, m::Integer) where {T} = n ≥ m ? d.γ^(n-m) : zero(T)
Base.size(d::DMatrix,i::Integer) = d.size
Base.size(d::DMatrix) = (d.size, d.size)

matQs = reduce(vcat, Qs);
matKs = reduce(hcat, Ks);
D = DMatrix(γ, size(matQs, 1));

vec(((matQs * matKs) .* DMatrix(γ, size(matQs,1))) * v) ≈ o


# The above demonstrates the basic properties of the Retentative network. Let's now make a nicer
# implementation, which we will make more general and faster.

struct RetentionLayer{Θ,M,T}
    θₐ::Θ
    Wk::M
    Wq::M
    Wv::M
    γ::T
end

function RetentionLayer(input_dim, hidden_dim; γ = 0.99f0, T = Float32)
    θₐ = T.(2π .* rand(hidden_dim ÷ 2))
    Wk = randn(T, hidden_dim, input_dim)
    Wq = randn(T, hidden_dim, input_dim)
    Wv = randn(T, 1, input_dim)
    γ = T(0.95)
    RetentionLayer(θₐ, Wk, Wq, Wv, γ)
end

Base.show(io::IO, m::RetentionLayer) = print(io, "RetentionLayer($(size(m.Wk,1)) => $(size(m.Wk,2)))")

function recursive_forward(layer::RetentionLayer, x)
    T = eltype(v)
    s₀ = zeros(T, size(layer.Wk, 1), size(layer.Wv, 1))
    o₀ = zeros(T, size(layer.Wv, 1))
    A = rot_matrix(layer.θₐ)
    function _step((on, sn), xᵢ)
        Kᵢ = layer.Wk * xᵢ
        Qᵢ = layer.Wq * xᵢ
        vᵢ = layer.Wv * xᵢ
        snn = A * sn + Kᵢ .* vᵢ'
        on = Qᵢ' * snn
        return(on', layer.γ .* snn)
    end
    os = accumulate(_step, eachslice(x, dims = 2), init = (o₀, s₀))
    reduce(hcat, map(first, os))
end


# For the batch version of the forward, we will use a faster version of the multiplication with
# the rotation. The advantage is that we do not need to expciltly construct the rotation matrix,
# but we would need to write our own gradient function, which is shown below.

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

# With that, the batch forward function will look like, which is neet and fast (but allocates a lot with which we will deal later)
function batch_forward(layer::RetentionLayer, x)
    Q = rotary_embedding(layer.Wq * x, -layer.θₐ)
    K = rotary_embedding(layer.Wk * x, -layer.θₐ)
    v = layer.Wv * x
    D = DMatrix(layer.γ, size(Q, 2));
    transpose(((Q' * K) .* D) * v')
end

# Let's test the equivalence of `batch_forward` and `recursive_forward`.

layer = RetentionLayer(6, 8)
x = randn(Float32, 6, 33)

recursive_forward(layer, x) ≈ batch_forward(layer, x)

# The `batch_forward` has quadratic memory footprint and complexity as the original transformer. But
# since the attention is almost linear, we can decrease the memory footprint by "chunking".
# In chunking, the input sequence is broken into continous sub-sequences, on each sub-sequence
# the output is computed in parralel, and then corrected for the previous sub_sequences.
#
# The math behind is simple and therefore not carried in the paper. The chunking is derived below
# to match it to the implemention. The `n`-ith output is computed as 
# ```math
# \begin{aligned}
# o_n & = \sum_{m=1}^n Q_n \gamma^ne^{in\theta} (K_m \gamma^{-m} e^{in\theta})^{\mathrm{T}} v_m \\ 
# o_n & = Q_n \gamma^ne^{in\theta} \sum_{m=1}^n  (K_m \gamma^{-m} e^{in\theta})^{\mathrm{T}} v_m
# \end{aligned}
# ```
# now imagine that we compute only items of the output higher than some ``n_0.`` We can therefore precompute
# part of the sum ``R_{n_0} = \sum_{m=1}^{n_0-1}  (K_m \gamma^{-m} e^{in\theta})^{\mathrm{T}} v_m`` with which 
# the items ``o_n, n \geq n_0`` can be computed as
# ```math
# o_n =  Q_n \gamma^ne^{in\theta} \sum_{m=n_0}^n  (K_m \gamma^{-m} e^{in\theta})^{\mathrm{T}} v_m  + Q_n \gamma^ne^{in\theta} R_{n_0}\\
# ```
# And this is great, because now the complexity is linear with respect to number of chunks and qudratic with respect to length of chunks.
# Interestingly, ``R_{n_0}`` can be computed sequentially utilizing outputs on previous chunks, or in parallel if one 
# wishes to compute chunks in paralel.

function chunked_forward(layer::RetentionLayer, x; chunk_size = 64)
    T = eltype(layer.Wq)
    Q = rotary_embedding(layer.Wq * x, -layer.θₐ)
    K = rotary_embedding(layer.Wk * x, -layer.θₐ)
    v = layer.Wv * x
    R₀ = zeros(T, size(layer.Wk, 1), size(layer.Wv, 1))
    os = map(1:chunk_size:size(v,2)) do n₀
        n₁ = min(n₀ + chunk_size - 1, size(v,2))
        Qᵢ = Q[:, n₀:n₁]
        Kᵢ = K[:, n₀:n₁]
        vᵢ = v[:, n₀:n₁]
        Dᵢ = DMatrix(layer.γ, size(Qᵢ, 2));
        Rᵢ = sum(layer.γ^(-m) * K[:,m] * v[:,m]' for m in 1:n₀-1; init = R₀)
        oᵢ = transpose(((Qᵢ' * Kᵢ) .* Dᵢ) * vᵢ') .+ γ .^ (n₀:n₁)'  .* (Rᵢ' * Qᵢ)
    end
    reduce(hcat, os)
end

chunked_forward(layer, x; chunk_size = 4) ≈ batch_forward(layer, x)

# Let's now test if the value does not have to be scalar, but it can be a vector. It should be
# just a matter of changing the size of matrix `Wv`. We add one more constructor, and 
# fix few bugs to make sure that everything works well (already applied fixed). 

function RetentionLayer(input_dim, hidden_dim, output_dim; γ = 0.99f0, T = Float32)
    θₐ = T.(2π .* rand(hidden_dim ÷ 2))
    Wk = randn(T, hidden_dim, input_dim)
    Wq = randn(T, hidden_dim, input_dim)
    Wv = randn(T, output_dim, input_dim)
    γ = T(0.95)
    RetentionLayer(θₐ, Wk, Wq, Wv, γ)
end

layer = RetentionLayer(8, 8, 8)
x = randn(Float32, 8, 256)
chunked_forward(layer, x; chunk_size = 64) ≈ batch_forward(layer, x)
recursive_forward(layer, x) ≈ batch_forward(layer, x)
 

# Let's now compare the speed of different implementations

layer = RetentionLayer(8, 8)
map(4:14) do i
    x = randn(Float32, 8, 2^i)
    (;
    sequence_length = 2^i,
    recurrent = (@elapsed recursive_forward(layer, x)),
    batch = (@elapsed batch_forward(layer, x)),
    chunked_16 = (@elapsed chunked_forward(layer, x; chunk_size = 16)),
    chunked_64 = (@elapsed chunked_forward(layer, x; chunk_size = 64)),
    chunked_256 = (@elapsed chunked_forward(layer, x; chunk_size = 256)),
    )
end |> DataFrame

# Which gives the following timing
# ```
#  Row │ sequence_length  recurrent    batch        chunked_16   chunked_64   chunked_256
#      │ Int64            Float64      Float64      Float64      Float64      Float64
# ─────┼──────────────────────────────────────────────────────────────────────────────────
#    1 │              16  6.25e-5      1.275e-5     1.8708e-5    5.417e-6     5.292e-6
#    2 │              32  2.8708e-5    1.5583e-5    1.8708e-5    1.1334e-5    1.0375e-5
#    3 │              64  4.4875e-5    3.1041e-5    4.6875e-5    3.2458e-5    3.0292e-5
#    4 │             128  8.2584e-5    0.000107583  0.000167292  8.1e-5       0.00010925
#    5 │             256  0.00015075   0.00046225   0.000606042  0.00485371   0.000414292
#    6 │             512  0.000283459  0.00168833   0.00205875   0.000634333  0.000844542
#    7 │            1024  0.000480084  0.00844817   0.00821817   0.00192354   0.00163133
#    8 │            2048  0.000851167  0.0250193    0.0316693    0.00868054   0.00414017
#    9 │            4096  0.00272846   0.125775     0.128583     0.0318552    0.0134293
#   10 │            8192  0.00445004   0.740209     0.47801      0.11457      0.0372682
#   11 │           16384  0.00708088   3.60559      1.87253      0.480278     0.147014
# ```
#
# The full version of the code can be found at [retnet_1.jl](retnet_1.jl)
#

