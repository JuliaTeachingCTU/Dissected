# #  Optimization
# 
# ### Decresing the memory footprint
# 
# The complexity of the batched (and chunked) version completely depends on the complexity of 
# the linear attention. So we will now focus on optimization of this layer. Since the implementation
# is getting longer, reader is kindly asked to copy it from [retnet_2.jl](retnet_2.jl)

using BenchmarkTools
include("retnet_heads.jl")
layer = RetentionLayer(128, 128, 128; nheads = 1);
x = randn(Float32, 128, 1024);
Q = rotary_embedding(layer.Wq * x, -layer.θₐ);
K = rotary_embedding(layer.Wk * x, -layer.θₐ);
V = layer.Wv * x;
γ = layer.γ;

function linear_attention(Q, K, V, γ)
    D = DMatrix(γ, size(Q, 2));
    transpose(((Q' * K) .* D) * V')
end

function linear_attention2(Q, K, V, γ)
    o = similar(V)
    for n in axes(Q, 2)
        o[:,n] .= 0
        γₙₘ = γ ^n
        @inbounds for m in 1:n
            γₙₘ /= γ
            α = zero(eltype(Q))
            for k in axes(Q, 1)
                α += Q[k, n] * K[k, m]
            end

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
        γₙₘ = γ ^n
        @inbounds for m in 1:n
            γₙₘ /= γ
            α = zero(eltype(Q))
            for k in axes(Q, 1)
                α += Q[k, n] * K[k, m]
            end

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

function linear_attention4(Q, K, V, γ)
    o = zeros(eltype(V), size(V))
    l = size(K,2)
    Threads.@threads :static for i in 1:Threads.nthreads()
        for n in i:Threads.nthreads():l
            γₙₘ = γ^n
            @inbounds for m in 1:n
                γₙₘ /= γ
                α = zero(eltype(Q))
                for k in axes(Q, 1)
                    α += Q[k, n] * K[k, m]
                end

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
