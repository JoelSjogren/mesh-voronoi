# Confirms a structural fact about the construction, not just a spot-check:
# whenever a subcell's label is a genuine two-way tie {f,g}, the top-
# dimensional cells it bounds strictly satisfy f<g (or g<f) in their own
# interior -- a direct consequence of the construction's core correctness
# invariant (every top-dimensional cell has a single winning label, and
# labels only change across a tie boundary), not a fact needing its own
# separate proof. Checked here at runtime against real constructed
# complexes, mixing points and segments.
@testset "tie-boundary invariant: subcell 'f=g' implies interior 'f<g' on each side" begin
    p1a, p1b = SVector(0.1, 0.05), SVector(1.4, -0.2)
    p2a, p2b = SVector(2.3, 1.1), SVector(3.6, 0.7)
    p3 = SVector(-1.0, 2.0)
    entries = [
        (:segment, p1a, p1b, 1, 2),
        (:segment, p2a, p2b, 3, 4),
        (:point, p3, 5),
    ]
    cx, feats = multi_complex(entries, Val(2))

    checked = 0
    for id in eachindex(cx.nodes)
        haskey(cx.superseded_by, id) && continue
        node = cx.nodes[id]
        node.dim == 1 || continue
        length(node.label) == 2 || continue
        fa, fb = collect(node.label)
        feat_a = only(f for f in feats if f.face == fa)
        feat_b = only(f for f in feats if f.face == fb)

        parents = [p for p in get(cx.referenced_by, id, Int[])
                   if !haskey(cx.superseded_by, p) && cx.nodes[p].dim == 2 && length(cx.nodes[p].label) == 1]
        for p in parents
            plabel = only(cx.nodes[p].label)
            plabel in (fa, fb) || continue
            sample = interior_sample(cx, p)
            # A feature's own quadratic is only a genuine distance within
            # its validity region -- a raw comparison beyond that is
            # extrapolation, not geometry, and can go either way. Skip
            # unless the sample is meaningfully comparable against *both*
            # tied features (the same guard the rest of the suite already
            # applies wherever it compares a feature's raw quadratic, e.g.
            # `point_segment_complex: every descendant point correctly
            # sided`).
            (is_valid(feat_a.validity, sample) && is_valid(feat_b.validity, sample)) || continue
            da, db = sqdist(feat_a.quad, sample), sqdist(feat_b.quad, sample)
            checked += 1
            if plabel == fa
                @test da < db
            else
                @test db < da
            end
        end
    end
    @test checked > 0
end
