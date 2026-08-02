using Random

"""
A little house with an open chimney. The roof's left slope is rerouted
through the chimney (up one side, across, and -- since the chimney is
"open" -- *not* across the top, leaving two free prongs) instead of running
straight to the peak. This makes the whole mesh one connected path with
exactly two free endpoints (the two open prongs at the chimney top):
topologically an arc (homeomorphic to a line segment), matching the
generic-case scope this project targets so far. Closing the missing top
edge would make it homeomorphic to a circle instead -- a floor-plan-style
closed loop -- which is "the opening of the chimney" this mesh is missing.
A separate, disconnected floating chimney (two components) would not be
generic in this sense: real meshes are connected complexes, not two
unrelated pieces that happen to share a drawing.
"""
function house_chimney()
    coords = SVector{2,Float64}[
        (0.0, 0.0),   # 1 floor / bottom-left
        (4.0, 0.0),   # 2 floor / bottom-right
        (4.0, 3.0),   # 3 right eave
        (2.0, 5.0),   # 4 roof peak
        (0.0, 3.0),   # 5 left eave
        (0.7, 3.7),   # 6 roof/chimney junction, left (on the original roof line)
        (0.7, 4.9),   # 7 chimney top-left (free)
        (1.3, 4.3),   # 8 roof/chimney junction, right (on the original roof line)
        (1.3, 5.5),   # 9 chimney top-right (free)
    ]
    simplices = InputSimplex[
        SegmentSimplex(1, 2),  # floor
        SegmentSimplex(2, 3),  # right wall
        SegmentSimplex(3, 4),  # right roof slope
        SegmentSimplex(4, 8),  # left roof slope, upper part (peak to chimney junction)
        SegmentSimplex(8, 9),  # chimney right side
        SegmentSimplex(6, 7),  # chimney left side
        SegmentSimplex(6, 5),  # left roof slope, lower part (chimney junction to eave)
        SegmentSimplex(5, 1),  # left wall
    ]
    return InputComplex(coords, simplices)
end

"""
The same house, with each vertex nudged by a small deterministic random
offset. The exact house has lots of non-generic coincidences by
construction -- right angles, collinear points, parallel walls -- each of
which is a real (if measure-zero) degeneracy outside this project's current
"generic case only" scope. This perturbed version is the one to actually
exercise the generic-case pipeline against; go back to the exact
`house_chimney()` only once the generic case is solid.
"""
function house_chimney_generic(; seed=1, scale=0.05)
    rng = MersenneTwister(seed)
    base = house_chimney()
    coords = [c + scale * SVector(randn(rng), randn(rng)) for c in base.coords]
    return InputComplex(coords, base.simplices)
end
