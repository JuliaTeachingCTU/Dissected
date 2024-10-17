
# ### Decresing the memory footprint
# 
# The `batch_forward` has quite high memory footprint, because of the outer
# product `(Q * V ).* D`. We can decrease the memory footprint using the 
# map
function batch_forward2(layer::RetentionLayer, x)
    Q = rotary_embedding(layer.Wq * x, -layer.θₐ)
    K = rotary_embedding(layer.Wk * x, -layer.θₐ)
    v = layer.Wv * x
    γ = layer.γ
    linear_attention(Q, K, v, γ)
end

function linear_attention(Q, K, v, γ)
    o = map(1:length(v)) do n 
        @inbounds sum(only( γ ^(n - m) * dot(view(Q, :, n), view(K, :, m)) * v[m]) for m in 1:n)
    end
    o'
end


layer = RetentionLayer(128, 128)
x = randn(Float32, 128, 512)

batch_forward(layer, x) ≈ batch_forward2(layer,x)

@benchmark batch_forward(layer, x)
@benchmark batch_forward2(layer,x)

# The `batch_forward2` is slower, but has much lower memory footprint. 
# This is because the `batch_forward` uses multi-threadding, while 
# `batch_forward2` uses only one thread. Let's try multi-threadding

function linear_attention(Q, K, v, γ)
    o = zeros(eltype(v), 1, size(x,2))
    Threads.@threads :dynamic for n in axes(v,2)
        @inbounds for m in 1:n
            o[n] += only( γ ^(n - m) * dot(view(Q, :, n), view(K, :, m)) * v[m])
        end
    end
    o
end

@benchmark batch_forward3(layer,x)

# With the implementation is equally fast as the `batch_forward`, 
# but with much lower memory footprint.


