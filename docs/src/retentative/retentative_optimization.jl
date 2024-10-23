# #  Optimization
# 
# ### Decresing the memory footprint
# 
# The complexity of the batched (and chunked) version completely depends on the complexity of 
# the linear attention. So we will now focus on optimization of this layer. Since the implementation
# is getting longer, reader is kindly asked to copy it from [retnet_heads.jl](retnet_heads.jl)

using BenchmarkTools
include("retnet_heads.jl")
layer = RetentionLayer(128, 128, 128; nheads = 1);
x = randn(Float32, 128, 1024);
Q = rotary_embedding(layer.Wq * x, -layer.θₐ);
K = rotary_embedding(layer.Wk * x, -layer.θₐ);
V = layer.Wv * x;
γ = layer.γ;

function linear_attention(Q, K, V, γ::Real)
    D = DMatrix(γ, size(Q, 2));
    transpose(((Q' * K) .* D) * V')
end

function linear_attention2(Q, K, V, γ)
    o = similar(V)
    for n in axes(Q, 2)
        o[:,n] .= 0
        @inbounds for m in 1:n
            α = zero(eltype(Q))
            for k in axes(Q, 1)
                α += Q[k, n] * K[k, m]
            end

            γₙₘ = γ^(n-m)
            for k in axes(Q, 1)
                o[k,n] += γₙₘ * α * V[k, m]
            end
        end
    end
    o
end

linear_attention(Q, K, V, γ) ≈ linear_attention2(Q, K, V, γ)
@benchmark linear_attention(Q, K, V, γ)
@benchmark linear_attention2(Q, K, V, γ)

# The `linear_attention2` is slower than `linear_attention`, but it allocates less memory.
# The solution is to use multi-threadding, because the `linear_attention` heavily relies on 
# optimized matrix multiplication.

function linear_attention3(Q, K, V, γ)
    o = similar(V)
    Threads.@threads :static for n in axes(Q, 2)
        o[:,n] .= 0
        @inbounds for m in 1:n
            α = zero(eltype(Q))
            for k in axes(Q, 1)
                α += Q[k, n] * K[k, m]
            end

            γₙₘ = γ^(n-m)
            for k in axes(Q, 1)
                o[k, n] += γₙₘ * α * V[k, m]
            end
        end
    end
    o
end


linear_attention(Q, K, V, γ) ≈ linear_attention2(Q, K, V, γ)
linear_attention(Q, K, V, γ) ≈ linear_attention3(Q, K, V, γ)
@benchmark linear_attention(Q, K, V, γ)
@benchmark linear_attention2(Q, K, V, γ)
@benchmark linear_attention3(Q, K, V, γ)

# `linear_attention3` is much better than `linear_attention2`, but pays the problem is that
# the workload of threads is not equal. This is caused by the fact that the inner loop
# grows linearly with `n`. We therefore change the items over which each thread operates.
# The idea is to interleave the indices, so the first thread will work on indices `l:Threads.nthreads():l`,
# second on `2:Threads.nthreads():l`, third on `3:Threads.nthreads():l` etc... This significantly improves
# the performance. Yet it is still slower than the original implementation. 

function linear_attention4(Q, K, V, γ::Number)
    o = zeros(eltype(V), size(V))
    l = size(K,2)
    Threads.@threads :static for i in 1:Threads.nthreads()
        for n in i:Threads.nthreads():l
            @inbounds for m in 1:n
                α = zero(eltype(Q))
                for k in axes(Q, 1)
                    α += Q[k, n] * K[k, m]
                end

                γₙₘ = γ^(n-m)
                for k in axes(Q, 1)
                    o[k,n] += γₙₘ * α * V[k, m]
                end
            end
        end
    end
    o
end

linear_attention(Q, K, V, γ) ≈ linear_attention2(Q, K, V, γ)
linear_attention(Q, K, V, γ) ≈ linear_attention3(Q, K, V, γ)
linear_attention(Q, K, V, γ) ≈ linear_attention4(Q, K, V, γ)

@benchmark linear_attention(Q, K, V, γ)
@benchmark linear_attention2(Q, K, V, γ)
@benchmark linear_attention3(Q, K, V, γ)
@benchmark linear_attention4(Q, K, V, γ)


# The nice part of our implementation is that we can easily add support over multiple heads
# without the pesky splicing that is needed for versions relying on fast blas operations.
# Let's modify the nice multi-threaded version to support multiple heads.

function _slices(hidden_dim::Integer, output_dim::Integer, nheads::Integer)
    head_dim = hidden_dim ÷ nheads
    odim = output_dim ÷ nheads
    kvslices = ntuple(i -> (i-1)*head_dim+1:i*head_dim, nheads)
    oslices = ntuple(i -> (i-1)*odim+1:i*odim, nheads)
    return(kvslices, oslices)
end

_slices(K::AbstractMatrix, V::AbstractVector, γs) = _slices(size(K,1), size(V,1), length(γs))

function linear_attention(Q, K, V, γs::AbstractVector{<:Number})
    kvslices, oslices = _slices(K,V,γs)
    vs = map(kvslices, oslices, γs) do kvslice, oslice, γ
        linear_attention(Q[kvslice,:], K[kvslice,:], V[oslice,:], γ)
    end
    reduce(vcat, vs)
end


function linear_attention4(Q, K, V, γs::AbstractVector{<:Number})
    kvslices, oslices = _slices(K,V,γs)
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

γs = Float32[0.85, 0.9, 0.95, 0.99]
linear_attention(Q, K, V, γs) ≈ linear_attention4(Q, K, V, γs)

@benchmark linear_attention(Q, K, V, γs)
@benchmark linear_attention4(Q, K, V, γs)

# We see that the hard work pays off for the multi-head version, which is more than 3 times faster
# than the version that relies on regular matrix multiplication (6.6ms vs 22ms). For this setting, it alocates less
# memory (518Kb vs 34Mb), which is sweet.


# and we should get rid of power in a thread-safe way
function linear_attention_forloop(Q, K, V, nheads, γs)
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
