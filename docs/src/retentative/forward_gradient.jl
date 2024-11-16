# While the batch version or chunked version is great for highly parallel systems, 
# like GPU, for a sequentional system, like CPU, the recursive version might be still better. 
# The question about its advantage hinges around the speed for gradient, which is questionable.
# Let's test. We take the definition of the retention layer from the final version and start to
# hack it.

using LinearAlgebra
using FiniteDifferences
using ChainRulesCore
using Zygote
using Flux

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

# Let's now replace the recursion through `accumulate` with a loop, where the output
# is preallocated.

function recursive_forward2(layer::RetentionLayer{N,<:AbstractVector}, x) where {N}
    Wk, Wq, Wv, θₐ, γs = layer.Wk, layer.Wq, layer.Wv, layer.θₐ, layer.γ
    kvslices, oslices = _slices(layer)
    θ_dim = length(first(kvslices)) ÷ 2
    os = map(enumerate(zip(γs, kvslices, oslices))) do (i, (γ, kvslice, oslice))
        θᵢ = θₐ[(i-1)*θ_dim+1:i*θ_dim]
        recursive_forward2(Wq[kvslice,:], Wk[kvslice,:], Wv[oslice,:], γ, θᵢ, x)
    end
    reduce(vcat, os)
end

function recursive_forward2(Wq, Wk, Wv, γ::Real, θₐ, x)
    T = eltype(x)
    s = zeros(T, size(Wk, 1), size(Wv, 1))
    A = rot_matrix(θₐ)
    K = Wk * x
    Q = Wq * x
    V = Wv * x
    inner_recursion2(A, K, V, Q, γ, s)
end

function inner_recursion2(A, K, V, Q, γ::Real, s)
    T = eltype(K)
    o = zeros(T, size(V, 1), size(K,2))
    @inbounds for i in axes(K ,2) 
        s = A * s + K[:,i] .* V[:,i]'
        o[:,i] .= s' * Q[:,i]
        s .*= γ 
    end
    o
end


nheads = 4
head_dim = 8
hidden_dim = nheads * head_dim
layer = RetentionLayer(hidden_dim, hidden_dim, hidden_dim; nheads)
x = randn(Float32, hidden_dim, 257)
recursive_forward(layer, x) ≈ recursive_forward2(layer, x)

# We verify that the output is approximately equal. When we benchmark both version, we obsereve
# that `recursive_forward2` is faster than `recursive_forward` (342.667 μs vs 664.854 μs). But,
# the new version is not ver friendly to AD and it will require to write our own gradient functions
# which we will do later.

# The next step is that we can replace the multiplications with loop

function recursive_forward3(layer::RetentionLayer{N,<:AbstractVector}, x) where {N}
    Wk, Wq, Wv, θₐ, γs = layer.Wk, layer.Wq, layer.Wv, layer.θₐ, layer.γ
    kvslices, oslices = _slices(layer)
    θ_dim = length(first(kvslices)) ÷ 2
    os = map(enumerate(zip(γs, kvslices, oslices))) do (i, (γ, kvslice, oslice))
        θᵢ = θₐ[(i-1)*θ_dim+1:i*θ_dim]
        recursive_forward3(Wq[kvslice,:], Wk[kvslice,:], Wv[oslice,:], γ, θᵢ, x)
    end
    reduce(vcat, os)
end

function recursive_forward3(Wq, Wk, Wv, γ::Real, θₐ, x)
    T = eltype(x)
    s = zeros(T, size(Wk, 1), size(Wv, 1))
    K = Wk * x
    Q = Wq * x
    V = Wv * x
    inner_recursion3(θₐ, K, V, Q, γ, s)
end

function rot(θ::AbstractVector, x::AbstractMatrix)
    o = similar(x)
    2*length(θ) == size(x,1) || error("θ should be twice of number of rows in x")
    sincosθ = map(sincos, θ)
    @inbounds for col in axes(x,2)
        for (k, (sinθ, cosθ)) in enumerate(sincosθ)
            i = 2*k - 1
            o[i,col] = cosθ * x[i,col] - sinθ * x[i+1,col]
            o[i+1,col] = sinθ * x[i,col] + cosθ * x[i+1,col]
        end
    end
    o
end

function inner_recursion3(θ, K, V, Q, γ::Real, s)
    T = eltype(K)
    o = zeros(T, size(V, 1), size(K,2))
    @inbounds for i in axes(K ,2) 
        sᵢ = rot(θ, s)

        # s += K[:,i] .* V[:,i]'
        for j in axes(K, 1)
            for k in axes(V, 1)
                sᵢ[j,k] += K[j,i] * V[k,i]
            end
        end

        # o[:,i] .= s' * Q[:,i]
        for j in axes(V, 1)
            v = zero(T)
            for k in axes(Q, 1)
                v += Q[k,i] * sᵢ[k, j]
            end
            o[j,i] = v
        end

        # s = sᵢ .* γ
        s = sᵢ .* γ 
    end
    o
end

# Explictly handwriting all the multiplication brings further increases the speed and decreses
# the memory consumption. The benchmark now reads 664.854 μs vs 342.667 μs vs 228.000 μs, which
# is faird deal. Before writing the gradient, we modify the function further to store inner state
# `s`, which might be useful for the gradient computation.


function recursive_forward4(layer::RetentionLayer{N,<:AbstractVector}, x) where {N}
    Wk, Wq, Wv, θₐ, γs = layer.Wk, layer.Wq, layer.Wv, layer.θₐ, layer.γ
    kvslices, oslices = _slices(layer)
    θ_dim = length(first(kvslices)) ÷ 2
    os = map(enumerate(zip(γs, kvslices, oslices))) do (i, (γ, kvslice, oslice))
        θᵢ = θₐ[(i-1)*θ_dim+1:i*θ_dim]
        recursive_forward4(Wq[kvslice,:], Wk[kvslice,:], Wv[oslice,:], γ, θᵢ, x)
    end
    reduce(vcat, os)
end

function recursive_forward4(Wq, Wk, Wv, γ::Real, θₐ, x)
    T = eltype(x)
    s = zeros(T, size(Wk, 1), size(Wv, 1))
    K = Wk * x
    Q = Wq * x
    V = Wv * x
    inner_recursion4(θₐ, K, V, Q, γ, s)
end

function rot!(o, θ::AbstractVector, x::AbstractMatrix)
    2*length(θ) == size(x,1) || error("θ should be twice of number of rows in x")
    size(o) == size(x) || error("o should have the same size as x")
    sincosθ = map(sincos, θ)
    @inbounds for col in axes(x,2)
        for (k, (sinθ, cosθ)) in enumerate(sincosθ)
            i = 2*k - 1
            o[i,col] = cosθ * x[i,col] - sinθ * x[i+1,col]
            o[i+1,col] = sinθ * x[i,col] + cosθ * x[i+1,col]
        end
    end
    o
end

function inner_recursion4(θ, K, V, Q, γ::Real, s₀::AbstractMatrix)
        T = eltype(K)
    o = zeros(T, size(V, 1), size(K,2))
    s = similar(s₀, size(s₀,1), size(s₀,2), size(V,2) + 1)
    s[:,:,1] .= s₀

    @inbounds for i in axes(K ,2)
        sᵢ = view(s, :, :, i+1)
        rot!(sᵢ, θ, view(s, :,:,i))

        # s += K[:,i] .* V[:,i]'
        for j in axes(K, 1)
            for k in axes(V, 1)
                sᵢ[j,k] += K[j,i] * V[k,i]
            end
        end

        # o[:,i] .= s' * Q[:,i]
        for j in axes(V, 1)
            v = zero(T)
            for k in axes(Q, 1)
                v += Q[k,i] * sᵢ[k, j]
            end
            o[j,i] = v
        end

        # s = sᵢ .* γ
        sᵢ .= sᵢ .* γ 
    end
    o
end

# This tweak further decreases the memory consumption, number of allocations, 
# and slightly increases the speed as a hidden bonus. The benchmark now reads
# 664.854 μs vs 342.667 μs vs 228.000 μs vs 187.958 μs.

using Test
@testset "Checking correctness" begin 
    nheads = 4
    head_dim = 8
    hidden_dim = nheads * head_dim
    layer = RetentionLayer(hidden_dim, hidden_dim, hidden_dim; nheads)
    x = randn(Float32, hidden_dim, 257)
    
    A = rot_matrix(layer.θₐ)
    @test rot(layer.θₐ, x) ≈ A * x

    @test recursive_forward(layer, x) ≈ recursive_forward2(layer, x)
    @test recursive_forward(layer, x) ≈ recursive_forward3(layer, x)
    @test recursive_forward(layer, x) ≈ recursive_forward4(layer, x)
end


# Let's now move to the implementation of the gradient. This will be a bit tricky
# and we need to be very careful, but the added benefit might be that it will be 
# blazingly fast comparing to the AD of the original version. We start by vriting
# and AD of the rotation, because it is nicely isolated part.

# In line with a convention of the ChainRulesCore, we will denote gradients by bar. 
# We assume that the AD gives us the gradient with respect to the output `ō`, 
# and we update the gradient with respect to the angles θ̄ and the input `x̄`. 


function ∂rot!(θ̄, x̄, ō, θ::AbstractVector, x::AbstractMatrix)
    2*length(θ) == size(x,1) || error("θ should be twice of number of rows in x")
    size(ō) == size(x) || error("o should have the same size as x")
    sincosθ = map(sincos, θ)
    @inbounds for col in axes(x,2)
        for (k, (sinθ, cosθ)) in enumerate(sincosθ)
            i = 2*k - 1
            # o[i,col] = cosθ * x[i,col] - sinθ * x[i+1,col]
            # o[i+1,col] = sinθ * x[i,col] + cosθ * x[i+1,col]

            θ̄[k] += ō[i,col]*(- sinθ * x[i,col] - cosθ * x[i+1,col])
            θ̄[k] += ō[i+1,col]*( cosθ * x[i,col] - sinθ * x[i+1,col])

            x̄[i,col]   = ō[i,col] * cosθ + ō[i+1,col] * sinθ
            x̄[i+1,col] = - sinθ * ō[i,col] + ō[i+1,col] * cosθ
        end
    end
    return(θ̄, x̄)
end

function ∂rot(ō, θ::AbstractVector, x::AbstractMatrix)
    θ̄  = zeros(eltype(θ), length(θ))
    x̄  = zeros(eltype(x), size(x))
    ∂rot!(θ̄, x̄, ō, θ, x)
end

# Always check the gradients with respect to forward diff, otherwise you are doomed.
using FiniteDifferences
using Test

@testset "Checking gradient of rot" begin 
    θ = randn(8)
    x = randn(16, 257)

    ō = ones(size(rot(θ, x)))
    @test ∂rot(ō, θ, x)[1] ≈ grad(central_fdm(5,1), θ -> sum(rot(θ, x)), θ)[1]
    @test ∂rot(ō, θ, x)[2] ≈ grad(central_fdm(5,1), x -> sum(rot(θ, x)), x)[1]
end


# The rotation was easy. Now the formidable oponent is the gradient of `inner_recursion4.jl`,
# as we need to be very careful of what we are overwriting and when.
# Recall that the gradient needs to go the other way around.


function ∂inner_recursion4(ō, θ, K, V, Q, γ::Real, s)
    T = eltype(K)
    Q̄ = zeros(T, size(Q))
    K̄ = zeros(T, size(K))
    V̄ = zeros(T, size(V))
    θ̄ = zeros(T, length(θ)) 
    γ̄ = zero(T)

    s̄ᵢ = zeros(T, size(s,1), size(s,2))
    s̄ᵢ₋₁ = zeros(T, size(s,1), size(s,2))

    @inbounds for i in reverse(axes(K ,2))
        sᵢ = view(s, :, :, i+1)

        # let's undo `sᵢ = sᵢ .* γ`, but update the gradient of γ
        γ̄ += sum(s̄ᵢ .* sᵢ)
        sᵢ .= sᵢ ./ γ

        # them, we update the gradient of θ
        for j in axes(V, 1)
            for k in axes(Q, 1)
                # forward part: o[j,i] += Q[k,i] * sᵢ[k,j]
                Q̄[k,i]  += ō[j,i] * sᵢ[k,j]
                s̄ᵢ[k,j] += Q[k,i] * ō[j,i]
            end
        end


        for j in axes(K, 1)
            for k in axes(V, 1)
                # forward part sᵢ[j,k] += K[j,i] * V[k,i]
                K̄[j,i] += s̄ᵢ[j,k] * V[k,i]
                V̄[k,i] += s̄ᵢ[j,k] * K[j,i]
            end
        end

        # rot!(view(s, :, :, i+1), θ, view(s, :,:,i))
        ∂rot!(θ̄, s̄ᵢ₋₁, s̄ᵢ, θ, view(s, :,:,i))
        s̄ᵢ, s̄ᵢ₋₁ = s̄ᵢ₋₁, s̄ᵢ

    end

    return(θ̄, K̄, V̄, Q̄, γ̄, s̄ᵢ)
end

# Since we need the `s` from the recursive_forward4 pass, we will modify a bit the calculation of 
# the recursive_forward4 to have a version which will output the state `s`.

function _inner_recursion4(θ, K, V, Q, γ::Real, s₀::AbstractMatrix)
        T = eltype(K)
    o = zeros(T, size(V, 1), size(K,2))
    s = similar(s₀, size(s₀,1), size(s₀,2), size(V,2) + 1)
    s[:,:,1] .= s₀

    @inbounds for i in axes(K ,2)
        sᵢ = view(s, :, :, i+1)
        rot!(sᵢ, θ, view(s, :,:,i))

        # s += K[:,i] .* V[:,i]'
        for j in axes(K, 1)
            for k in axes(V, 1)
                sᵢ[j,k] += K[j,i] * V[k,i]
            end
        end

        # o[:,i] .= s' * Q[:,i]
        for j in axes(V, 1)
            v = zero(T)
            for k in axes(Q, 1)
                v += Q[k,i] * sᵢ[k, j]
            end
            o[j,i] = v
        end

        # s = sᵢ .* γ
        sᵢ .= sᵢ .* γ 
    end
    o, s
end

@testset "Checking gradient of ∂inner_recursion4" begin 
    s₀ = zeros(16,16)
    θ = randn(8)
    K = randn(16, 13)
    V = randn(16, 13)
    Q = randn(16, 13)
    γ = 1.0

    o, s = _inner_recursion4(θ, K, V, Q, γ, s₀)
    ō = ones(size(o))
    @test ∂inner_recursion4(ō, θ, K, V, Q, γ, s)[1] ≈ grad(central_fdm(5,1), θ -> sum(inner_recursion4(θ, K, V, Q, γ, s₀)), θ)[1]
    @test ∂inner_recursion4(ō, θ, K, V, Q, γ, s)[2] ≈ grad(central_fdm(5,1), K -> sum(inner_recursion4(θ, K, V, Q, γ, s₀)), K)[1]
    @test ∂inner_recursion4(ō, θ, K, V, Q, γ, s)[3] ≈ grad(central_fdm(5,1), V -> sum(inner_recursion4(θ, K, V, Q, γ, s₀)), V)[1]
    @test ∂inner_recursion4(ō, θ, K, V, Q, γ, s)[4] ≈ grad(central_fdm(5,1), Q -> sum(inner_recursion4(θ, K, V, Q, γ, s₀)), Q)[1]
    @test ∂inner_recursion4(ō, θ, K, V, Q, γ, s)[5] ≈ grad(central_fdm(5,1), γ -> sum(inner_recursion4(θ, K, V, Q, γ, s₀)), γ)[1]
    @test ∂inner_recursion4(ō, θ, K, V, Q, γ, s)[6] ≈ grad(central_fdm(5,1), s₀ -> sum(inner_recursion4(θ, K, V, Q, γ, s₀)), s₀)[1]
end

# This start to looks great. To finish the stuff, we create a chainrule for  `inner_recursion4`
# to make this compatible with Zygote.

function ChainRulesCore.rrule(::typeof(inner_recursion4), θ, K, V, Q, γ, s₀)
    y, s = _inner_recursion4(θ, K, V, Q, γ, s₀)
    function inner_recursion_pullback(ȳ)
        return(NoTangent(), ∂inner_recursion4(ȳ, θ, K, V, Q, γ, s)...)
    end
    return y, inner_recursion_pullback
end

# Finally, we verify that it works with Zygote as intended.
nheads = 4
head_dim = 8
hidden_dim = nheads * head_dim
layer = RetentionLayer(hidden_dim, hidden_dim, hidden_dim; nheads)
x = randn(Float32, hidden_dim, 257)
gradient(layer -> sum(recursive_forward4(layer, x)), layer) !== nothing

# and how about stress-tests?
using DataFrames
using BenchmarkTools
map(2:20) do n 
    l = 2^n
    x = randn(Float32, hidden_dim, l)
    stats = (;
        length = l, 
        forward  = (@elapsed recursive_forward4(layer, x)),
        gradient = (@elapsed gradient(layer -> sum(recursive_forward4(layer, x)), layer)),
    )
    @show stats
    stats
end |> DataFrame
