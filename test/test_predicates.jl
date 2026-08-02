"""
The one predicate every later milestone needs: which of two `AffineQuadratic`s
is closer at `x` (equivalently, which side of their bisector `x` is on).
Wrapped once here so the tests below read as scenarios, not boilerplate.
"""
closer_sign(A, B, x) = exact_sign((a, b, y) -> evaluate(bisector(a, b), y), A, B, x)

@testset "exact_sign: clearly separated case, fast path suffices" begin
    A = AffineQuadratic(SVector(0.0, 0.0))
    B = AffineQuadratic(SVector(2.0, 0.0))
    # bisector is exactly {x1 = 1}; a point far to one side is unambiguous
    @test closer_sign(A, B, SVector(5.0, 0.3)) == 1    # farther from B's side of the algebra: sqdist(A)-sqdist(B) > 0 means B is closer
    @test closer_sign(A, B, SVector(-5.0, 0.3)) == -1
end

@testset "exact_sign: exactly-degenerate configuration returns 0" begin
    A = AffineQuadratic(SVector(0.0, 0.0))
    B = AffineQuadratic(SVector(2.0, 0.0))
    # (1, y) is exactly equidistant from A and B for any y -- a genuine
    # tie, not floating-point noise, and exact_sign must say so exactly.
    for y in (0.0, 0.37, -12.5, 1000.125)
        @test closer_sign(A, B, SVector(1.0, y)) == 0
    end
end

@testset "exact_sign: near-degenerate (not exact) still resolves correctly" begin
    A = AffineQuadratic(SVector(0.0, 0.0))
    B = AffineQuadratic(SVector(2.0, 0.0))
    # nudged by a tiny amount off the true bisector x1=1 -- small enough
    # that the interval fast path may be inconclusive, but the true sign
    # is still well-defined and the exact fallback must get it right.
    @test closer_sign(A, B, SVector(1.0 + 1e-13, 0.0)) == 1
    @test closer_sign(A, B, SVector(1.0 - 1e-13, 0.0)) == -1
end

@testset "exact_sign: identical points tie everywhere (no crash, no NaN)" begin
    # this is the 2D prototype's actual "shared vertex from two different
    # simplices" bug in miniature: the bisector of a point against itself
    # degenerates to the zero quadric, tied at every point in space, not
    # just a curve -- exact_sign must report 0 everywhere, never divide by
    # a zero norm or otherwise misbehave.
    A = AffineQuadratic(SVector(3.0, -1.0))
    Asame = AffineQuadratic(SVector(3.0, -1.0))
    for x in (SVector(0.0, 0.0), SVector(100.0, -50.0), SVector(3.0, -1.0))
        @test closer_sign(A, Asame, x) == 0
    end
end

@testset "exact_sign: works the same way at higher N" begin
    for N in (2, 3, 4)
        A = AffineQuadratic(SVector{N,Float64}(zeros(N)))
        b = zeros(N)
        b[1] = 2.0
        B = AffineQuadratic(SVector{N,Float64}(b))
        far = zeros(N)
        far[1] = 5.0
        @test closer_sign(A, B, SVector{N,Float64}(far)) == 1
    end
end

@testset "symbolic_tiebreak: deterministic, antisymmetric, consistent" begin
    a, b, c = [1, 2], [1, 3], [1, 2]
    @test symbolic_tiebreak(a, b) == -symbolic_tiebreak(b, a)
    @test symbolic_tiebreak(a, c) == 0   # identical tags -- not meant to occur for distinct sub-simplices, but must not crash
    # repeated calls give the same answer (determinism, not just correctness once)
    results = [symbolic_tiebreak(a, b) for _ in 1:5]
    @test all(==(results[1]), results)
end
