# The first genuinely new 3D milestone beyond points-only (the previous
# one, in test_multi_points.jl, needed no code changes at all): a 3D line
# segment's own interior feature is a rank-1 curved quadric (in general
# ambient dimension, not just N=2 -- see AffineQuadratic's own docstring),
# so inserting one against an existing point is the first case that
# actually needs `clip_top_cell_3d!`/`clip_flat_face_3d!` (walking a 3D
# cell's own 2-skeleton, restricting the bisector to each flat face's own
# local 2D frame, and capping the resulting cut with a new curved face) --
# the 3D analogue of `clip_top_cell_2d!`, one dimension up.
#
# v1 scope, deliberately: every face touched must be flat (an
# already-curved face -- e.g. a cap left by an earlier insertion -- can't
# be split further yet), with no holes, and crossed at exactly 0 or 2
# points total (a single connected cut per face; the 3D analogue of a
# curved bisector "biting" one edge twice, already handled at N=2, isn't
# supported yet at N=3). Every violation errors loudly rather than being
# silently mishandled -- see the last testset below.

@testset "3D: one point + one segment, vertex-level cross-validation" begin
    # Deliberately asymmetric (no axis-aligned coincidences) coordinates,
    # exercising a genuinely oblique bisector against the box's own flat
    # faces.
    ptpos = SVector(4.371, -2.183, 3.902)
    pa, pb = SVector(-5.234, 3.017, -1.845), SVector(-2.109, -4.876, 6.234)

    lo, hi = SVector(-10.0, -10.0, -10.0), SVector(10.0, 10.0, 10.0)
    cx, top = init_bbox_complex(Val(3), lo, hi)
    pf = point_feature(ptpos, 1)
    label1 = Label([pf.face])
    for id in eachindex(cx.nodes)
        set_label!(cx, id, label1)
    end
    feats = GFeature{3}[pf]
    insert_segment!(cx, feats, pa, pb, 2, 3)
    merge_adjacent_same_label_cells!(cx)

    # Every input sub-simplex's own label should exist as a live cell.
    for face in (Set([1]), Set([2]), Set([3]), Set([2, 3]))
        @test haskey(cx.label_index, Label([face])) && !isempty(cx.label_index[Label([face])])
    end

    # Cross-validated against `recompute_feature_label` (the same trusted
    # oracle `weld_near_duplicate_vertices!` itself uses), not a hand-rolled
    # one: a naive "one region per segment" oracle can't see a genuine tie
    # sitting exactly on the segment's own endpoint/interior validity
    # boundary, which is routinely where a vertex actually lands (confirmed
    # a real false-positive source during development, not a hypothetical
    # one -- see the "tie oracle" investigation notes elsewhere in this
    # project's history).
    checked = 0
    mismatches = 0
    for (id, node) in enumerate(cx.nodes)
        haskey(cx.superseded_by, id) && continue
        node.dim == 0 || continue
        expected = recompute_feature_label(node.point, feats)
        checked += 1
        node.label != expected && (mismatches += 1)
    end
    @test checked > 20
    @test mismatches == 0
end

# Regression test for the very first bug found building this: two faces of
# a fresh box sharing a common edge each independently re-split it when
# clipped by the same plane, creating two different vertices at the same
# location instead of one -- so the four collected trace edges never
# shared a vertex pairwise, and stitching them into the new capping face's
# own boundary loop failed outright (a fresh box, split by one plane, is
# about as simple as a 3D clip gets). Fixed by `clip_top_cell_3d!` sharing
# one `edge_cache` across every face of the same cell.
@testset "3D: clip_by_plane_preserving_label! on a fresh box (shared-edge regression)" begin
    lo, hi = SVector(-10.0, -10.0, -10.0), SVector(10.0, 10.0, 10.0)
    cx, top = init_bbox_complex(Val(3), lo, hi)
    set_label!(cx, top, Label([Set([1])]))
    for id in eachindex(cx.nodes)
        set_label!(cx, id, Label([Set([1])]))
    end
    n = SVector(0.6, 0.8, 0.0)
    d = dot(n, SVector(-3.0, 2.0, 0.0))
    MeshVoronoi.clip_by_plane_preserving_label!(cx, top, n, d)
    @test length(top_cell_ids(cx)) == 2
    for id in top_cell_ids(cx)
        @test cx.nodes[id].label == Label([Set([1])])
    end
end

# Regression test for the second bug found: a flat face's own boundary
# edge can legitimately already be curved (shared with a sibling top cell
# whose own independent clip curved it first -- the N=3 analogue of the
# N=2 "a shared boundary is already patched by the time a later cell is
# examined" behavior `insert_features!`'s own docstring describes), and
# that edge's own curve needs restricting to *this* face's local frame too
# (`edge_curve_crossings`, not the straight-only `edge_crossings`) rather
# than erroring outright or silently treating it as straight.
@testset "3D: point + segment, stress cross-validated across many configurations" begin
    configs = [
        (SVector(-3.635, 1.198, -2.844), SVector(4.021, -3.552, 1.667), SVector(-1.203, 4.416, -3.981)),
        (SVector(2.917, 3.664, -4.203), SVector(-4.552, -1.667, 3.019), SVector(3.881, -2.204, -1.556)),
        (SVector(-1.442, -4.671, 2.883), SVector(1.019, 2.556, 4.117), SVector(-3.664, -0.982, -2.219)),
        (SVector(3.108, -0.664, -3.917), SVector(-2.203, 4.552, 1.019), SVector(0.667, -3.881, 3.204)),
    ]
    for (ptpos, pa, pb) in configs
        lo, hi = SVector(-10.0, -10.0, -10.0), SVector(10.0, 10.0, 10.0)
        cx, top = init_bbox_complex(Val(3), lo, hi)
        pf = point_feature(ptpos, 1)
        label1 = Label([pf.face])
        for id in eachindex(cx.nodes)
            set_label!(cx, id, label1)
        end
        feats = GFeature{3}[pf]
        insert_segment!(cx, feats, pa, pb, 2, 3)
        merge_adjacent_same_label_cells!(cx)

        checked = 0
        mismatches = 0
        for (id, node) in enumerate(cx.nodes)
            haskey(cx.superseded_by, id) && continue
            node.dim == 0 || continue
            expected = recompute_feature_label(node.point, feats)
            checked += 1
            node.label != expected && (mismatches += 1)
        end
        @test checked > 20
        @test mismatches == 0
    end
end

# The v1 scope limit itself, confirmed to fail loudly (not silently
# mishandled) rather than assumed: a curved bisector that dips into and
# back out of a single face (the 3D analogue of the N=2 "curved bisector
# crossing one edge twice" case) isn't supported yet.
@testset "3D: multi-crossing within a single face errors clearly (known v1 gap)" begin
    lo, hi = SVector(-10.0, -10.0, -10.0), SVector(10.0, 10.0, 10.0)
    cx, top = init_bbox_complex(Val(3), lo, hi)
    ptpos = SVector(-1.282, 3.674, 4.0)
    pf = point_feature(ptpos, 1)
    label1 = Label([pf.face])
    for id in eachindex(cx.nodes)
        set_label!(cx, id, label1)
    end
    feats = GFeature{3}[pf]
    pa, pb = SVector(-3.263, 1.587, -0.03), SVector(3.611, 2.368, 2.372)
    err = nothing
    try
        insert_segment!(cx, feats, pa, pb, 2, 3)
    catch e
        err = e
    end
    @test err !== nothing
    @test occursin("v1 scope", sprint(showerror, err))
end
