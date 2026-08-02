# Containment check: does `Interval` arithmetic actually bound the true
# (exact-rational-computed) result of the same expression? This is the
# property the whole exact-predicate scheme depends on -- if it doesn't
# hold, the fast path could confidently report a wrong sign.
@testset "Interval arithmetic contains the exact result" begin
    for _ in 1:2000
        xs = [rand(-10.0:0.0001:10.0) for _ in 1:6]
        ivs = Interval.(xs)
        exs = Exact.(xs)

        # a representative small expression chain: (a+b)*(c-d) + e*f
        iv_result = (ivs[1] + ivs[2]) * (ivs[3] - ivs[4]) + ivs[5] * ivs[6]
        ex_result = (exs[1] + exs[2]) * (exs[3] - exs[4]) + exs[5] * exs[6]

        @test interval_contains(iv_result, ex_result)
    end
end

@testset "Interval straddles_zero" begin
    @test straddles_zero(Interval(-1.0, 1.0))
    @test straddles_zero(Interval(0.0, 0.0))
    @test straddles_zero(Interval(-1e-20, 1e-20))
    @test !straddles_zero(Interval(1.0, 2.0))
    @test !straddles_zero(Interval(-2.0, -1.0))
end

@testset "Interval arithmetic is not overly loose" begin
    # a chain of well-separated-from-zero operations should not straddle
    # zero just from the safety inflation -- otherwise every predicate
    # would degrade to the slow exact path uselessly often.
    a, b = Interval(3.0), Interval(5.0)
    @test !straddles_zero(a * b - Interval(1.0))   # 3*5-1 = 14, nowhere near 0
end
