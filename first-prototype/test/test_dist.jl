@testset "dist2 / brute_force_label (M0 oracle)" begin
    @testset "two isolated points" begin
        complex = InputComplex(
            SVector{2,Float64}[(0.0, 0.0), (2.0, 0.0)],
            InputSimplex[PointSimplex(1), PointSimplex(2)],
        )
        @test brute_force_label(complex, SVector(0.0, 0.0)) == Set([1])
        @test brute_force_label(complex, SVector(3.0, 0.0)) == Set([2])
        @test brute_force_label(complex, SVector(1.0, 0.0)) == Set([1, 2])   # perpendicular bisector
        @test brute_force_label(complex, SVector(1.0, 5.0)) == Set([1, 2])   # still on the bisector
    end

    @testset "single segment" begin
        complex = InputComplex(
            SVector{2,Float64}[(0.0, 0.0), (2.0, 0.0)],
            InputSimplex[SegmentSimplex(1, 2)],
        )
        @test dist2(complex, 1, SVector(1.0, 1.0)) ≈ 1.0     # perpendicular foot at (1,0)
        @test dist2(complex, 1, SVector(-1.0, 0.0)) ≈ 1.0    # beyond endpoint a
        @test dist2(complex, 1, SVector(3.0, 0.0)) ≈ 1.0     # beyond endpoint b
        @test brute_force_label(complex, SVector(1.0, 1.0)) == Set([1])
    end

    @testset "point vs. segment-interior bisector is a parabola (sanity spot-check)" begin
        # focus p = (0,1); directrix line y = 0 realized as a long segment on the x-axis.
        complex = InputComplex(
            SVector{2,Float64}[(0.0, 1.0), (-100.0, 0.0), (100.0, 0.0)],
            InputSimplex[PointSimplex(1), SegmentSimplex(2, 3)],
        )
        # parabola y = (x^2 + 1) / 2 satisfies dist(x,focus) == dist(x,directrix) for the standard
        # focus-directrix parabola with focus (0,1), directrix y=0.
        for x in -3.0:0.5:3.0
            y = (x^2 + 1) / 2
            @test brute_force_label(complex, SVector(x, y)) == Set([1, 2])
        end
        # off the parabola, only one should win
        @test brute_force_label(complex, SVector(0.0, -5.0)) == Set([2])  # far below: segment wins
        @test brute_force_label(complex, SVector(0.0, 5.0)) == Set([1])   # far above: point wins
    end
end
