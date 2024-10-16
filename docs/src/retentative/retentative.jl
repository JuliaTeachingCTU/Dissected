# This will implement Retentative neural networks, which was published in
# a paper with bold title: "Retentive Network: A Successor to Transformer for Large Language Models",
# at https://arxiv.org/pdf/2307.08621. The main advantage of the architecture
# is that it allows parallel training and efficient (recursive) inference.
# 
# The implementation will be intentionally simple to demonstrate the key concepts
# and especially verify author's understanding of the math.

using LinearAlgebra

# The explanation of the model stats with state-space model assuming 
# the input `v` to be (projected to) 1d signal.
# The state of the model is updated as `s\[n+1\] = A*s\[n\] + K*u\[n\]` and
# the output is as `o\[n\] = Q*s\[n\]`. 
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
# vocal about that. Anyway the important property of the matrix `A` are such that `A^(i+j)` A^i*A^j` 
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
# `D` with `D_{ij} = γ^(|i-j|) if i ≥ j`. Then, the inference can be written in matrix
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

function RetentionLayer(input_dim, hidden_dim, γ = 0.99f0, T = Float32)
    θₐ = T.(2π .* rand(hidden_dim ÷ 2))
    Wk = randn(T, hidden_dim, input_dim)
    Wq = randn(T, hidden_dim, input_dim)
    Wv = randn(T, 1, input_dim)
    γ = T(0.95)
    RetentionLayer(θₐ, Wk, Wq, Wv, γ)
end

Base.show(io::IO, m::RetentionLayer) = print(io, "RetentionLayer($(size(m.Wk,1)) => $(size(m.Wk,2)))")

function recursive_forward(m::RetentionLayer, x)
    T = eltype(v)
    s₀ = zeros(size(m.Wk, 1))
    o₀ = zero(T)
    A = rot_matrix(m.θₐ)
    function _step((on, sn), xᵢ)
        K = m.Wk * xᵢ
        Q = m.Wq * xᵢ
        vᵢ = m.Wv * xᵢ
        snn = A * sn + K .* vᵢ
        on = dot(Q, snn)
        return(on, m.γ .* snn)
    end
    os = accumulate(_step, eachslice(x, dims = 2), init = (o₀, s₀))
    map(first, os)'
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
function batch_forward(m::RetentionLayer, x)
    Qs = rotary_embedding(m.Wq * x, -m.θₐ)
    Ks = rotary_embedding(m.Wk * x, -m.θₐ)
    v = m.Wv * x
    D = DMatrix(m.γ, size(Qs, 2));
    transpose(((Qs' * Ks) .* D) * v')
end

# Does it work? Let's test the equivalence of `batch_forward` and `recursive_forward`.

m = RetentionLayer(6, 8)
x = randn(Float32, 6, 33)

recursive_forward(m, x) ≈ batch_forward(m, x)
