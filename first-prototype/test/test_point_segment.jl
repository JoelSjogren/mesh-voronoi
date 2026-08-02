@testset "full pipeline: one point + one segment (M3)" begin
    # focus (0,3) directly above the midpoint of a segment from (-2,0) to (2,0).
    # padded bbox works out to roughly x in [-3.2,3.2], y in [-1.2,4.2].
    complex = InputComplex(
        SVector{2,Float64}[(0.0, 3.0), (-2.0, 0.0), (2.0, 0.0)],
        InputSimplex[PointSimplex(1), SegmentSimplex(2, 3)],
    )
    dcel = compute_output_complex(complex)

    @test simplices_of(label_at(dcel, SVector(0.0, 4.0))) == Set([1])      # near the top of the bbox: point wins
    @test simplices_of(label_at(dcel, SVector(0.0, 0.0))) == Set([2])      # on the segment: segment wins
    @test simplices_of(label_at(dcel, SVector(3.0, 0.0))) == Set([2])      # beyond endpoint b: still segment (via the endpoint feature)
    @test simplices_of(label_at(dcel, SVector(-1.0, -1.1))) == Set([2])    # below the segment interior: segment wins

    # beyond endpoint b (vertex 3) is a *point* feature -- no discrete side
    # -- while directly below the interior is a *segment-interior* feature,
    # which does have one.
    @test label_at(dcel, SVector(3.0, 0.0)) == Set([(face=Set([3]), side=nothing, simplices=Set([2]))])
    @test only(label_at(dcel, SVector(-1.0, -1.1))).face == Set([2, 3])

    # two points symmetric about the segment's own line, both comfortably
    # within its interior's territory (not pulled into either endpoint's or
    # the point simplex's territory), get the same face but opposite side.
    below = only(label_at(dcel, SVector(-1.5, -0.3)))
    above = only(label_at(dcel, SVector(-1.5, 0.3)))
    @test below.face == above.face == Set([2, 3])
    @test below.side != above.side

    check_grid_matches_oracle(dcel, complex; xr=range(-3.1, 3.1, length=29), yr=range(-1.1, 4.1, length=29))
end
