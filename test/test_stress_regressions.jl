# Minimal, deterministic reproductions found via random-stress testing
# (`examples/benchmark_bvh.jl`'s `robust_multi_complex` used to roll these
# back and silently skip them) for three previously-open construction bugs.
# Kept as literal fixed-point regression tests -- not randomized -- so a
# future change that reopens any of them fails immediately and
# reproducibly, rather than only showing up again after enough stress-test
# runs to re-discover it by luck.

@testset "regression: quadric_quadric_crossings on an unrelated stale edge_curve" begin
    # A cell's own boundary edge's `.curve` field is fixed at creation time
    # (recording whichever two features were being compared then); a later
    # wholesale relabel (`insert_features!`'s `side == :b` branch) can change
    # the cell's *label* without ever touching that edge. A later insertion
    # comparing the cell's new winner against a new feature can then hand
    # `quadric_quadric_crossings` a `curve`/`edge_curve` pair sharing no
    # common feature at all -- this exact 5-entry sequence used to error
    # with "none of curve, edge_curve, or curve±edge_curve is a genuine
    # line/line-pair" on the 5th insertion.
    entries = Any[
        (:point, SVector(4.394413850810366, -4.9337147815194005), 1),
        (:segment, SVector(-2.6084422686508173, -6.185838000809623), SVector(-3.723307026617353, 8.666991937924088), 28, 29),
        (:point, SVector(-1.9999938706353788, -3.5606423343927514), 9),
        (:segment, SVector(-7.658384375309111, 6.511518411511766), SVector(2.0453362814121334, -6.479656584672098), 16, 17),
        (:point, SVector(-9.254352937220432, 6.229706114749245), 15),
    ]
    cx, _ = init_bbox_complex(Val(2), SVector(-10.0, -10.0), SVector(10.0, 10.0))
    feats = GFeature{2}[]
    for e in entries
        insert_entry!(cx, feats, entry_feats(e, Val(2)))
    end
    @test true   # reaching here (no error) is the whole test
end

@testset "regression: merge_adjacent_same_label_cells! on exact-duplicate cells" begin
    # Two live top cells with the *identical* set of subcells (not merely
    # adjacent) used to make `merge_adjacent_same_label_cells!` count every
    # one of their edges as "internal" (shared by 2+), leaving zero
    # `external` edges and crashing `add_cell!`'s own bbox reduction
    # (`ArgumentError: reducing over an empty collection`) on this exact
    # 23-entry sequence's 20th insertion.
    entries = Any[
        (:segment, SVector(-2.3213932320007213, 2.3509279048928384), SVector(-3.918272426881872, 7.015613184649627), 24, 25),
        (:point, SVector(-2.275381256123083, 9.586194948899763), 7),
        (:point, SVector(-6.1137975461893905, 2.715858654581959), 15),
        (:point, SVector(7.847816665560234, 3.4123166251622212), 14),
        (:point, SVector(-9.25387557631549, -2.6292516903680774), 1),
        (:segment, SVector(-8.891970366650153, 5.862484108278938), SVector(-1.876973564359881, 8.806938225729091), 16, 17),
        (:segment, SVector(-1.985162531061082, -3.5430926352899883), SVector(-8.014144222954645, 5.782593509887416), 22, 23),
        (:point, SVector(-5.000660422213395, -5.556186360453765), 11),
        (:point, SVector(-3.436136387312718, -3.454662850976435), 5),
        (:point, SVector(-0.7051788897834239, -9.213353270416626), 8),
        (:segment, SVector(-5.820947857542889, 4.930460877064171), SVector(8.185012934395292, 3.3787764721904274), 18, 19),
        (:point, SVector(3.244598746070551, 2.115559967639985), 2),
        (:segment, SVector(2.0649008766428008, -6.623064920885467), SVector(7.555979944239741, 1.8334467731916284), 26, 27),
        (:point, SVector(-4.52695659564262, 3.8363843844715895), 9),
        (:segment, SVector(7.704307095049565, -7.263097495269775), SVector(4.959472463391494, 9.735804321735952), 28, 29),
        (:point, SVector(5.0061336479772045, 0.7448429083831432), 10),
        (:point, SVector(9.832999946781808, -3.2822256752541374), 6),
        (:point, SVector(-5.813041787070883, 6.963310359321252), 3),
        (:point, SVector(-3.4237230154392844, -6.158972100082762), 13),
        (:segment, SVector(-5.463495245140684, -4.8174129551133005), SVector(4.56846044632162, -2.711155302749095), 30, 31),
    ]
    cx, _ = init_bbox_complex(Val(2), SVector(-10.0, -10.0), SVector(10.0, 10.0))
    feats = GFeature{2}[]
    for e in entries
        insert_entry!(cx, feats, entry_feats(e, Val(2)))
    end
    @test true
end

@testset "regression: cyclic_boundary_walks with a dangling tip" begin
    # A boundary can end up with a "dangling tip" -- a chain of one or more
    # edges hanging off an otherwise-closed cycle at a vertex with no other
    # live incident edge at all -- which used to make `cyclic_boundary_walks`
    # error outright ("boundary edges don't form a union of simple cycles")
    # on this exact 6-entry sequence's 6th insertion. Such a spike is now
    # pruned before walking (it encloses no area, so dropping it is the
    # geometrically correct answer, not a tolerated approximation).
    entries = Any[
        (:point, SVector(4.394413850810366, -4.9337147815194005), 1),
        (:point, SVector(7.9376819568702714, -6.892931086596262), 11),
        (:segment, SVector(0.6144974458228667, -7.515049471249351), SVector(8.168639722702956, 3.5911023204876162), 18, 19),
        (:segment, SVector(-7.658384375309111, 6.511518411511766), SVector(2.0453362814121334, -6.479656584672098), 16, 17),
        (:segment, SVector(4.942059351038123, -3.8634908293226573), SVector(7.689443024565456, 8.345136737502472), 30, 31),
        (:segment, SVector(-6.765049067049446, -9.53026155531721), SVector(2.7272747481172566, 0.3613427549607007), 22, 23),
    ]
    cx, _ = init_bbox_complex(Val(2), SVector(-10.0, -10.0), SVector(10.0, 10.0))
    feats = GFeature{2}[]
    for e in entries
        insert_entry!(cx, feats, entry_feats(e, Val(2)))
    end
    @test true
end

@testset "regression: hole independently cut by the same bisector as its container" begin
    # A hole is a separate live top cell in its own right, so it can be
    # independently cut by the very same bisector splitting its container in
    # the same insertion round -- the "carried through unchanged" assumption
    # `clip_top_cell_2d!` otherwise relies on. This exact 5-entry sequence's
    # 5th insertion used to error ("a hole ... doesn't land inside any of the
    # ... resulting piece(s)"); a downstream leftover from the same root
    # cause could then also crash a *later* insertion's `interior_sample`
    # with a raw indexing error. Both are now handled: the hole falls back to
    # the nearest resulting piece by bounding-box center (with a warning),
    # and any degenerate (zero-area) cell left behind is retired at the end
    # of `insert_entry!` before it can reach a later round.
    entries = Any[
        (:segment, SVector(6.239801963653861, -9.29181057931733), SVector(-0.9579265517486419, -0.927350108615796), 18, 19),
        (:segment, SVector(5.828773689798458, 2.9316388323026725), SVector(-2.8272926818164983, -7.692953291036437), 24, 25),
        (:point, SVector(-0.42545713321786316, -4.017920873348999), 11),
        (:segment, SVector(-1.0270944049173494, -4.380444227787656), SVector(1.3349766049737966, 0.25404508480747445), 26, 27),
        (:segment, SVector(-9.185219078140863, -5.422162381924782), SVector(5.719202827923883, 1.9616655573564188), 22, 23),
    ]
    cx, _ = init_bbox_complex(Val(2), SVector(-10.0, -10.0), SVector(10.0, 10.0))
    feats = GFeature{2}[]
    for e in entries
        insert_entry!(cx, feats, entry_feats(e, Val(2)))
    end
    @test true
end
