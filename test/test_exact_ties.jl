# `CellNode.exact_ties` records a genuine exact tie (`exact_sign` returning
# a real 0, not just numerically small) the moment `clip_by_hyperplane!`'s
# own vertex-level loop finds one, instead of discarding it once
# `symbolic_tiebreak` picks a side -- see the field's own docstring
# (`src/complex/cell.jl`) and `project_pairwise_vs_multiway_ties.md` for why
# this is worth keeping as a first-class fact rather than a resolve-and-
# forget detail.
@testset "exact_ties: a square's own center is a genuine 4-way tie" begin
    # Four points at the corners of an axis-aligned square with integer
    # coordinates -- their common center (2,2) is *exactly* equidistant
    # from all four (not just numerically close: with integer input, the
    # BigInt-rational fallback inside `exact_sign` sees a literal 0), the
    # classic degenerate case where a 2D Voronoi vertex is tied among more
    # than the generic minimum of 3 features.
    entries = [
        (:point, SVector(0.0, 0.0), 1),
        (:point, SVector(4.0, 0.0), 2),
        (:point, SVector(4.0, 4.0), 3),
        (:point, SVector(0.0, 4.0), 4),
    ]
    cx, feats = multi_complex(entries, Val(2))

    center_id = findfirst(1:length(cx.nodes)) do id
        haskey(cx.superseded_by, id) && return false
        node = cx.nodes[id]
        node.dim == 0 && node.point !== nothing && isapprox(node.point, SVector(2.0, 2.0); atol=1e-9)
    end
    @test center_id !== nothing

    ties = exact_ties(cx, center_id)
    @test ties !== nothing
    # Every recorded atom must be a real single-point feature this vertex's
    # own (already independently computed) label agrees is tied there --
    # `exact_ties` is additive evidence for the same fact `label` already
    # settled on, never in conflict with it.
    node = cx.nodes[center_id]
    @test node.label == Set([Set([1]), Set([2]), Set([3]), Set([4])])
    @test ties ⊆ node.label
    @test length(ties) >= 2   # at least one genuine pairwise exact coincidence recorded
end

# Sanity check in the other direction, matching the empirical finding
# behind the docstring above: an arbitrary, asymmetric configuration with
# no special structure should essentially never produce a genuine exact
# tie (probability zero for random real input), so `exact_ties` should stay
# `nothing` everywhere -- confirms the field doesn't fire spuriously.
@testset "exact_ties: an asymmetric configuration has none" begin
    entries = [
        (:point, SVector(0.37, -1.21), 1),
        (:point, SVector(2.84, 0.63), 2),
        (:point, SVector(-1.58, 3.02), 3),
    ]
    cx, feats = multi_complex(entries, Val(2))
    @test all(exact_ties(cx, id) === nothing for id in eachindex(cx.nodes) if !haskey(cx.superseded_by, id))
end
