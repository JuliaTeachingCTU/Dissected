# # Rotary embedding
# Rotary embedding is a technique frequenly used in transformers to encode 
# the position of the input data. The central idea is to rotate the input 
# data with the angle of the rotation being dependent on the position of the 
# token. A nice feature of the rotary embedding embedding is that it is shift
# invariant, which means that after dot-product (the multiplicaiton of query and keys)
# the output depends on the relative distance of tokens.
# 
# The rotary embedding takes its name from rotation, which multiplies the input. It is
# assumed that the dimension of the input (the length of feature vector) is even.
#
# We start with a pedagogical example, which is obvious. The rotation matrix parametrized
# by `Vector θ` is defined as follows:

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

rot_matrix([π/2, π/4])

# We see that he rotation matrix is a block matrix, where the diagonal blocks 
# are 2x2 rotation matrices.
#
# With that, the rotary embedding can be written as

function slow_rotary_embedding(x, θ)
    d = size(x,1)
    2*d == length(θ) && error("θ should be twice of x")

	o = similar(x)
	for i in axes(x,2)
		o[:,i] .= rot_matrix(i*θ) * x[:,i]
	end
	o	
end

slow_rotary_embedding(rand(4, 5), [π/2, π/4])

# Since the rotation matrix is sparse, we can write a function which
# fuses the computations and construction of the rotation matrix.
# This has the nice property that we do not need to explicitly store
# the rotation matrix

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

# We verify that the implementations are equal

x = rand(4, 5)
θ = [π/2, π/4]

rotary_embedding(x, θ) ≈ slow_rotary_embedding(x, θ)

# but the latter is faster with less alocations, which is what we want.
using BenchmarkTools
@benchmark rotary_embedding(x, θ)
@benchmark slow_rotary_embedding(x, θ)
