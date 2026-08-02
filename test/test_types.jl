@testset "AffineQuadratic point case (K=0) matches plain squared distance" begin
    for N in (2, 3, 4)
        p = SVector{N,Float64}(randn(N))
        q = AffineQuadratic(p)
        for _ in 1:10
            x = SVector{N,Float64}(randn(N))
            @test sqdist(q, x) ≈ sum(abs2, x - p)
        end
    end
end

@testset "AffineQuadratic hyperplane case (K=N-1) matches (n·x-d)²" begin
    for N in (2, 3, 4)
        n = normalize(SVector{N,Float64}(randn(N)))
        d = randn()
        q = hyperplane_quadratic(n, d)
        for _ in 1:10
            x = SVector{N,Float64}(randn(N))
            @test sqdist(q, x) ≈ (dot(n, x) - d)^2
        end
    end
end

@testset "AffineQuadratic general K matches direct projection formula" begin
    for N in (3, 4), K in 0:(N-1)
        p = SVector{N,Float64}(randn(N))
        raw = randn(N, K)
        basis = K == 0 ? SMatrix{N,0,Float64}() : SMatrix{N,K,Float64}(Matrix(qr(raw).Q))
        q = AffineQuadratic{N,K,Float64}(p, basis)
        for _ in 1:10
            x = SVector{N,Float64}(randn(N))
            d = x - p
            expected = sum(abs2, d) - sum(abs2, basis' * d)
            @test sqdist(q, x) ≈ expected
            @test sqdist(q, x) >= -1e-9   # always nonnegative (it's a squared distance)
        end
    end
end

@testset "to_quadric matches sqdist" begin
    for N in (2, 3, 4), K in 0:(N-1)
        p = SVector{N,Float64}(randn(N))
        raw = randn(N, K)
        basis = K == 0 ? SMatrix{N,0,Float64}() : SMatrix{N,K,Float64}(Matrix(qr(raw).Q))
        q = AffineQuadratic{N,K,Float64}(p, basis)
        quad = to_quadric(q)
        for _ in 1:10
            x = SVector{N,Float64}(randn(N))
            @test evaluate(quad, x) ≈ sqdist(q, x)
        end
    end
end

@testset "bisector is equidistant and correctly signed" begin
    for N in (2, 3, 4)
        A = AffineQuadratic(SVector{N,Float64}(randn(N)))
        B = AffineQuadratic(SVector{N,Float64}(randn(N)))
        bis = bisector(A, B)
        # the midpoint between two points always lies on their bisector
        mid = (A.p + B.p) / 2
        @test abs(evaluate(bis, mid)) < 1e-9
        # sqdist(A,x) - sqdist(B,x) matches evaluate(bis,x) exactly (it's the same quadric)
        for _ in 1:10
            x = SVector{N,Float64}(randn(N))
            @test evaluate(bis, x) ≈ sqdist(A, x) - sqdist(B, x)
        end
    end
end
