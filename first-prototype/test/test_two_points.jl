"""
Cross-validate a computed DCEL against the brute-force oracle over a dense
grid, skipping points too close to a true classification boundary (where
the "correct" answer is an inherently ambiguous tie and a single-sided
point-in-face test may reasonably land on either neighbor). Compares via
`simplices_of`, projecting the richer face/side-aware label down to the
plain winning-simplex set the oracle produces.
"""
function check_grid_matches_oracle(dcel, complex; xr, yr, tie_atol=1e-6, skip=nothing)
    @testset "grid cross-validation" begin
        for x in xr, y in yr
            pt = SVector(x, y)
            skip !== nothing && skip(pt) && continue
            ds = sort([dist2(complex, i, pt) for i in eachindex(complex.simplices)])
            length(ds) > 1 && ds[2] - ds[1] < tie_atol && continue
            computed = label_at(dcel, pt)
            computed === nothing && continue
            @test simplices_of(computed) == brute_force_label(complex, pt)
        end
    end
end

@testset "full pipeline: two isolated points (M2)" begin
    complex = InputComplex(
        SVector{2,Float64}[(0.0, 0.0), (2.0, 0.0)],
        InputSimplex[PointSimplex(1), PointSimplex(2)],
    )
    dcel = compute_output_complex(complex)

    @test length(live_faces(dcel)) == 2
    @test simplices_of(label_at(dcel, SVector(0.0, 0.0))) == Set([1])
    @test simplices_of(label_at(dcel, SVector(2.0, 0.0))) == Set([2])
    @test simplices_of(label_at(dcel, SVector(-0.4, 0.4))) == Set([1])
    @test simplices_of(label_at(dcel, SVector(2.4, -0.4))) == Set([2])

    # each point simplex's own face is the vertex itself, with no discrete
    # side (a point is codimension 2 in the ambient plane -- its orthogonal
    # complement is a whole subspace, not a single normal direction).
    @test label_at(dcel, SVector(0.0, 0.0)) == Set([(face=Set([1]), side=nothing, simplices=Set([1]))])
    @test label_at(dcel, SVector(2.0, 0.0)) == Set([(face=Set([2]), side=nothing, simplices=Set([2]))])

    check_grid_matches_oracle(dcel, complex; xr=range(-0.55, 2.55, length=27), yr=range(-0.55, 0.55, length=15))
end
