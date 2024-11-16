using SmellyAV
using FiniteDifferences
using Zygote
using Test
using SmellyAV: linear_attention

@testset "Gradient of linear attention" begin
   Q = randn(8,127);
   K = randn(8,127);
   V = randn(6,127);
   @testset "γ = $(γ)" for γ in (0.95, [0.99, 0.95], rand(4))
       γ = 0.95;
       @test gradient(Q -> sum(linear_attention(Q, K, V, γ)), Q)[1] ≈ grad(central_fdm(5,1), Q -> sum(linear_attention(Q, K, V, γ)), Q)[1]
       @test gradient(K -> sum(linear_attention(Q, K, V, γ)), K)[1] ≈ grad(central_fdm(5,1), K -> sum(linear_attention(Q, K, V, γ)), K)[1]
       @test gradient(V -> sum(linear_attention(Q, K, V, γ)), V)[1] ≈ grad(central_fdm(5,1), V -> sum(linear_attention(Q, K, V, γ)), V)[1]
       @test gradient(γ -> sum(linear_attention(Q, K, V, γ)), γ)[1] ≈ grad(central_fdm(5,1), γ -> sum(linear_attention(Q, K, V, γ)), γ)[1]
   end

    # Testing type stability
    Q = randn(Float32, 8,127);
    K = randn(Float32, 8,127);
    V = randn(Float32, 6,127);
    γs = rand(Float32, 4)
    nheads = Val(4)
    ȳ = Zygote.FillArrays.Fill(1f0, (Base.OneTo(size(V,1)), Base.OneTo(size(Q,2))))
    ∂linear_attention(ȳ, Q, K, V, γs, nheads)
end

@testset "gradient of cross_retention" begin 
    @testset "nheads $(nheads)" for nheads in [1,2,4]
        K = randn(8,127);
        V = randn(6,127);
        γs = rand([0.9, 0.95, 0.99], nheads);
        n₀ = rand(1:8)  # do not try larger, since due to numerical instability the error 
        ȳ = ones(size(cross_retention(K, V, γs, n₀, Val(nheads))))

        @test _∂cross_retention(ȳ, K, V, γs, 1:(n₀-1), Val(nheads))[1] ≈ grad(central_fdm(5,1), K -> sum(cross_retention(K, V, γs, n₀, Val(nheads))), K)[1]
        @test _∂cross_retention(ȳ, K, V, γs, 1:(n₀-1), Val(nheads))[2] ≈ grad(central_fdm(5,1), V -> sum(cross_retention(K, V, γs, n₀, Val(nheads))), V)[1]
        @test _∂cross_retention(ȳ, K, V, γs, 1:(n₀-1), Val(nheads))[3] ≈ grad(central_fdm(5,1), γs -> sum(cross_retention(K, V, γs, n₀, Val(nheads))), γs)[1]
    end
end

@testset "Checking gradient of rot" begin 
    θ = randn(8)
    x = randn(16, 257)

    ō = ones(size(rot(θ, x)))
    @test ∂rot(ō, θ, x)[1] ≈ grad(central_fdm(5,1), θ -> sum(rot(θ, x)), θ)[1]
    @test ∂rot(ō, θ, x)[2] ≈ grad(central_fdm(5,1), x -> sum(rot(θ, x)), x)[1]
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

@testset "RetNet" begin
    @testset "RetNet forward $(nheads)" for nheads in [1,2,3,4]
        for i in 1:10
            head_dim = rand([2,4,8])
            odim = nheads * rand([2,4,8])
            hidden_dim = head_dim*nheads
            layer = RetentionLayer(hidden_dim, hidden_dim, odim; nheads)
            θ_dim = head_dim ÷ 2

            layers = map(zip(1:head_dim:hidden_dim, 1:θ_dim:length(layer.θₐ), 1:(odim÷nheads):odim, layer.γ)) do (i,j,o,γ)
               ii = i:i+head_dim-1
               jj = j:j+θ_dim-1
               oo = o:(o+odim ÷ nheads -1)
               RetentionLayer(layer.θₐ[jj], layer.Wk[ii,:], layer.Wq[ii,:], layer.Wv[oo,:], γ, Val(1))
            end

            x = randn(Float32, hidden_dim, 257)
            ref_o = vcat(map(l -> recursive_forward(l, x), layers)...)

            @test recursive_forward(layer, x) ≈ ref_o
            @test batch_forward(layer, x) ≈ ref_o
            @test chunk_forward(layer, x;chunk_size = 16) ≈ ref_o
            @test chunk_forward(layer, x;chunk_size = 64) ≈ ref_o
            @test recursive_forward(layer, x) ≈ batch_forward(layer, x)
        end
   end

   @testset "RetNet gradient $(nheads)" for nheads in [1,2,3,4]
        for i in 1:10
            head_dim = rand([2,4,8])
            hidden_dim = head_dim*nheads
            odim = nheads * rand([2,4,8])
            layer = RetentionLayer(hidden_dim, hidden_dim, odim; nheads)
            x = randn(Float32, hidden_dim, 257)
            @test gradient(layer -> sum(batch_forward(layer, x)), layer) !== nothing
            @test gradient(layer -> sum(chunk_forward(layer, x;chunk_size = 16)), layer) !== nothing
            gradient(layer -> sum(recursive_forward(layer, x)), layer) !== nothing
        end
    end
end


function benchmark(hidden_dim, nheads; max_length = 2^18)
    nheads = 4
    head_dim = 8
    hidden_dim = nheads * head_dim
    layer = RetentionLayer(hidden_dim, hidden_dim, hidden_dim; nheads)

    map(2:14) do n 
        x = randn(Float32, hidden_dim, 2^n)
        (;
        length = 2^n,    
        recursive = (@elapsed recursive_forward(layer, x)),
        batch = (@elapsed batch_forward(layer, x)),
        chunk_16 = (@elapsed chunk_forward(layer, x;chunk_size = 16)),
        chunk_64 = (@elapsed chunk_forward(layer, x;chunk_size = 64)),
        chunk_256 = (@elapsed chunk_forward(layer, x;chunk_size = 256)),
        )
    end |> DataFrame

    #  Row │ length  recursive    batch         chunk_16     chunk_64     chunk_256
    #      │ Int64   Float64      Float64       Float64      Float64      Float64
    # ─────┼──────────────────────────────────────────────────────────────────────────
    #    1 │      4  5.2917e-5     4.8541e-5    4.8334e-5    2.0416e-5    1.9584e-5
    #    2 │      8  3.8291e-5     2.0417e-5    2.6167e-5    2.325e-5     2.2333e-5
    #    3 │     16  7.0959e-5     3.075e-5     3.425e-5     3.4416e-5    3.425e-5
    #    4 │     32  0.000134208   6.4459e-5    7.5e-5       7.1e-5       7.1625e-5
    #    5 │     64  0.000263083   0.000186875  0.000157792  0.000204833  0.0002015
    #    6 │    128  0.000515959   0.000677833  0.000332875  0.00043975   0.000708708
    #    7 │    256  0.00102221    0.00252546   0.000641333  0.000795667  0.00226912
    #    8 │    512  0.00179071    0.00823471   0.00126683   0.00140733   0.00377296
    #    9 │   1024  0.00302042    0.0298707    0.00236908   0.00278829   0.00759108
    #   10 │   2048  0.0111392     0.127241     0.00495408   0.00575862   0.0154631
    #   11 │   4096  0.0159487     0.553578     0.0104537    0.0142831    0.0314418
    #   12 │   8192  0.0288055     2.90426      0.0222641    0.0251438    0.0670638
    #   13 │  16384  0.0641441    14.7794       0.137337     0.0514778    0.128426


    map(2:14) do n 
        l = 2^n
        x = randn(Float32, hidden_dim, l)
        (;
        length = l,    
        batch = (l > 2^11) ? missing : (@elapsed gradient(layer -> sum(batch_forward(layer, x)), layer)),
        chunk_16 = (@elapsed gradient(layer -> sum(chunk_forward(layer, x;chunk_size = 16)), layer)),
        chunk_64 = (@elapsed gradient(layer -> sum(chunk_forward(layer, x;chunk_size = 64)), layer)),
        chunk_256 = (@elapsed gradient(layer -> sum(chunk_forward(layer, x;chunk_size = 256)), layer)),
        )
    end |> DataFrame

    #  Row │ length  batch              chunk_16     chunk_64     chunk_256
    #      │ Int64   Float64?           Float64      Float64      Float64
    # ─────┼──────────────────────────────────────────────────────────────────
    #    1 │      4        0.0533746    0.0385143    0.0370291    0.0367485
    #    2 │      8        6.2834e-5    0.000209209  0.000141792  0.000118
    #    3 │     16        4.6084e-5    0.000124459  0.000122458  0.000121958
    #    4 │     32        9.95e-5      0.000249625  0.000187417  0.000182541
    #    5 │     64        0.000296375  0.00043625   0.000395083  0.000392958
    #    6 │    128        0.00108533   0.000938375  0.000841416  0.00124121
    #    7 │    256        0.00440333   0.00196908   0.00173488   0.00465504
    #    8 │    512        0.019027     0.0172583    0.00407242   0.0102465
    #    9 │   1024        0.0858177    0.0165314    0.00819775   0.0217295
    #   10 │   2048        0.360544     0.0347675    0.143459     0.0439564
    #   11 │   4096  missing            0.296723     0.0481539    0.0925153
    #   12 │   8192  missing            0.686332     0.111078     0.31607
    #   13 │  16384  missing            2.83961      0.69763      0.520861



    nheads = 4
    head_dim = 8
    hidden_dim = nheads * head_dim
    layer = RetentionLayer(hidden_dim, hidden_dim, hidden_dim; nheads)
    map(2:20) do n 
        l = 2^n
        x = randn(Float32, hidden_dim, l)
        stats = (;
        length = l,    
        # chunk_64 = (@elapsed gradient(layer -> sum(chunk_forward2(layer, x;chunk_size = 64)), layer)),
        # chunk_256 = (@elapsed gradient(layer -> sum(chunk_forward(layer, x;chunk_size = 256)), layer)),
        check_1024 = (@elapsed gradient(layer -> sum(chunk_forward2(layer, x;chunk_size = 1024)), layer)),
        # chunk_1024 = (@elapsed gradient(layer -> sum(chunk_forward2(layer, x;chunk_size = 1024)), layer)),
        )
        @show stats
        stats
    end |> DataFrame
end
