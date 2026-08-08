# MeshVoronoi

A dimension-generic algorithm for **generalized Voronoi diagrams**: instead of
partitioning space by nearest *point*, it partitions by nearest *feature* of
an input simplicial complex (vertices, segments, and in general k-faces),
built from one algebraic fact.

This project was entirely vibe coded.

## The one idea

The squared distance from a point `x` to any affine subspace (anchored at `p`,
with orthonormal direction basis `B`) is

```
sqdist(x) = |x - p|^2 - |Bᵀ(x - p)|^2
```

— exactly quadratic in `x` for every feature dimension at once (`K=0`, a
point, recovers plain squared distance; `K=N-1`, a hyperplane, recovers
`(n·x-d)^2`). Two features tie where their two quadratics are equal, which is
always a **quadric**: `xᵀMx + 2bᵀx + c = 0`. The whole diagram is an
arrangement of quadrics, assembled one feature at a time. Nothing about the
construction assumes flat or curved, 2D or 3D — it assumes *quadric*, and asks
which one. The bisector's own `M` is the difference of the two features' `M`s,
and its rank decides the shape: rank 0 is a hyperplane (point vs. point),
rank 1 is a parabola/paraboloid (point vs. line-interior), rank 2 indefinite
factors *exactly* into a line pair (line-interior vs. line-interior).

Cells are built by one recursion, one skeleton dimension at a time: insert a
new bisector by finding where it meets the *existing* skeleton one dimension
down (itself the same problem, one level lower), bottoming out at inserting
an isolated point. A cell's **label** is the set of input sub-simplices tied
for the win there; labels run opposite to containment (a boundary cell's
label is a *superset* of what it bounds), which makes supercells a free
powerset check and needs no separate index — only subcells need indexing,
populated as a byproduct of the same recursion.

## Status

- **2D is implemented** for the generic case (points and segments, `k=1`):
  flat and curved bisectors, multi-crossing edges, quadric-vs-quadric edge
  intersection, and multiply-connected (annular) cells all work.
- **Exact-predicate kernel**: sign decisions route through an adaptive
  predicate (fast float with a rigorous error bound, falling back to exact
  arithmetic only when needed), with a fixed symbolic tiebreak for genuine
  ties, so degenerate input never resolves two different ways.
- **3D points-only works** (`points_complex(points)` at `N=3`, an ordinary
  3D Voronoi diagram): a point-vs-point bisector is always a flat
  hyperplane regardless of dimension, so the initial milestone needed *no*
  code changes — every piece (`init_bbox_complex`, `insert_point!`, the
  merge/weld passes) was already written generically over `N`. That
  "needed nothing" claim later needed one correction (see the topology
  verification tool bullet below): a genuinely N-agnostic bug in
  `supersede!`, only ever *exercised* by a `dim>=2` face (so invisible at
  `N=2`), was found and fixed. Verified with vertex-level cross-validation
  against a brute-force oracle (zero mismatches across several point
  counts, 3 to 25 points), a genuine 8-way tie at a cube's exact center
  resolving to one correctly-labeled vertex, and the topology tool below.
- **3D points + a single line segment works, within a stated v1 scope**
  (`insert_segment!` at `N=3`): a segment's own interior feature is a
  genuinely curved quadric surface even in 3D (not just `N=2`), so this
  needed the actual new machinery — `clip_top_cell_3d!`/
  `clip_flat_face_3d!` walk a 3D cell's own 2-skeleton (faces), restrict
  the bisector to each flat face's own local 2D frame
  (`restrict_to_plane`), reuse the existing, already-proven 2D
  curved-edge machinery there, and cap the resulting cut with a new
  curved face — the 3D analogue of `clip_top_cell_2d!`'s own 2D boundary
  walk, one dimension up. Verified with vertex-level cross-validation
  (zero mismatches) across several point+segment configurations. v1
  scope, deliberately: every face touched must be flat (an
  already-curved face, e.g. a cap left by an earlier insertion, can't be
  split further yet), with no holes, and crossed at exactly 0 or 2 points
  total per face (a single connected cut — the 3D analogue of a curved
  bisector crossing one edge twice, already handled at `N=2`, isn't
  supported yet at `N=3`). Every one of these errors loudly rather than
  being silently mishandled.
- **3D: two arbitrary segments (nothing else) now work, within a stated
  v1 scope** (`multi_complex` at `N=3`) — the follow-up to the bigon fix
  above. `clip_curved_face_3d!` (`clip.jl`) splits an already-curved face
  by a *flat* clipping plane (a strict, easier sub-case of full
  quadric-vs-quadric, since one side is always flat — the confirmed
  dominant real-world blocker, a second segment's own endpoint
  validity-cut planes crossing the first segment's cap, 57/60 in an
  earlier stress run), dispatched from `clip_top_cell_3d!` alongside
  `clip_flat_face_3d!` based on whether the touched face is already
  curved. Two more weld-timing bugs surfaced and got fixed on the way:
  `insert_own_lines!`'s own two endpoint-plane cuts, and the gap between
  `insert_own_lines!` returning and the main comparison loop starting in
  `insert_features!`, can each leave a near-duplicate vertex pair for a
  *later* step to trip over before the usual end-of-call weld gets a
  chance to run — fixed by welding at both of those earlier points too,
  not just once at the very end. Verified via a 150-trial random
  two-segment stress run: 108/150 construction successes (72%), zero
  label mismatches against `recompute_feature_label` in every one, and
  all 42 failures are the *already-documented* "single connected cut" v1
  limit (multi-crossing), not a new gap.
- **3D: curved-vs-curved is also solved now** — the scenario just above
  where the *clipping* bisector is itself curved (not just the face being
  clipped), e.g. a background point plus two segments, or three arbitrary
  segments, where a later comparison routinely needs to cross an
  already-curved cap with a bisector that's curved too. Checked whether
  the `N=2` "line pair" shortcut generalizes (a 2D segment is a
  hyperplane, so its own squared-distance is a perfect square, making
  segment-vs-segment bisectors trivially factor into two lines) — it
  doesn't: a 3D segment's squared-distance is a genuine rank-2 quadratic,
  and a real 3D segment-vs-segment bisector was confirmed (numerically)
  to have full extended-matrix rank, not a degenerate plane pair. Solved
  instead via `ruled_quadric.jl`: every curved quadric this codebase
  produces has `rank(M) <= 2` (point-vs-segment is rank 1, segment-vs-
  segment is rank 2 indefinite — specifically a *hyperbolic paraboloid*,
  a doubly-ruled surface, not a general hyperboloid), which admits an
  *explicit* two-parameter description without a dedicated "quadric
  surface ∩ quadric surface → space curve" primitive in the fully general
  sense. `clip_curved_face_3d!`'s "clipping bisector must be flat"
  restriction was removed (boundary-edge crossings never actually needed
  it flat); the one genuinely new piece is that the resulting trace edge
  can be confined to two curved surfaces at once, with no flat neighbor
  of its own — `CellNode` gained a second `curve2` field for exactly
  that case, located later via `ruled_trace_crossing` (trace the arc
  through one quadric's own ruled structure, turning "does a third
  quadric cross this edge" into an ordinary 1D bisection). Verified
  directly: the exact point+2-segment repro above (which had failed at
  four different points across this whole 3D effort) now succeeds
  outright, cross-validated against the oracle. Re-running the 3-segment
  stress suite: the "clipping bisector is itself curved" error is
  completely gone (was 33/78, now 0) — but overall 3-segment success is
  still ~0/78, now dominated entirely by the *pre-existing*, unrelated
  "multi-crossing" limit instead, not a new gap this work introduced.
- **3D: multi-crossing within a single face is also solved now** —
  `clip_flat_face_3d!`/`clip_curved_face_3d!` generalized to allow any
  number of connected cuts per face, not just one, mirroring
  `clip_top_cell_2d!`'s own "bite" handling one dimension down. Two real
  bugs surfaced getting this working: (1) a face's single cut needs its
  *one* closing trace edge shared by both resulting pieces, not created
  twice (naively porting `clip_top_cell_2d!`'s own per-run pattern, which
  *is* correct one dimension lower, broke every single-cut case at
  first); (2) independent per-face crossing detection can leave near-
  duplicate cut vertices (and, after merging those, near-duplicate edges)
  at a shared corner before the codebase's normal end-of-insertion weld
  ever runs — fixed by a new, locally-scoped `weld_cap_vertices!` right
  before each cell's own cap-loop check. A narrower numerical issue in
  `ruled_trace_all_crossings`'s branch-tracking scan (spurious crossings
  past the Bezout bound of 4, from losing the true branch) was downgraded
  to a warning + single-side fallback rather than blocking construction.
  Verified via a fresh 150-trial random 3-segment stress run: **21/150
  successes (14%, up from the ~0-2/78 ceiling before)**, zero label
  mismatches in any of them.
- **3D: non-manifold caps solved too — 37/150 (25%)** — the dominant
  blocker just above (two "bites" meeting at a shared vertex, giving the
  cap's own edge graph a branch point) turned out to be a modeling bug,
  not a walk-ordering one: `cyclic_boundary_walks` treats its input as a
  2D *cell's* own boundary (genuinely 1D, correctly a union of simple
  cycles — "cheese", one outer loop plus holes), but a
  `clip_top_cell_3d!` cap is a 2D *surface*, and its own boundary
  structure is a graph embedded on that surface — closer to a
  polyhedron's edge skeleton (a cube corner has three faces meeting
  there, completely normal) than to a simple polygon. New
  `trace_cap_faces` decomposes it properly: a rotation system built from
  `quad`'s own gradient (globally consistent — an independent per-vertex
  best-fit plane isn't, and was confirmed empirically to merge everything
  into one degenerate mega-trace instead of separating faces) drives a
  standard combinatorial-map half-edge walk tracing every distinct 2D
  patch the cut's own topology has, with no constraint on vertex degree
  (that only constrains an edge-disjoint *cycle cover*, a different,
  stricter question this graph was never promised to satisfy).
  `clip_top_cell_3d!` now creates one new cap face per traced patch, all
  shared between the same two new 3-cells (the comparison is strictly
  binary, so however many patches the cut needs, they all separate the
  same A-side from the same B-side) — this transparently subsumes the
  older "multiply-connected cap" limit too (genuinely disjoint cut
  patches), not just the self-touching case. Verified via a fresh
  150-trial random 3-segment stress run: **37/150 successes (25%)**, zero
  label mismatches in any of them.
- **3D: single-face holes solved too — 49/150 (33%)** — ported
  `clip_top_cell_2d!`'s own hole-handling ("cheese": an outer loop plus
  any number of holes) down to one 3D *face*'s own boundary. New
  `find_outer_loop_3d`/`point_in_edge_loop_3d`/`loop_interior_point_3d`
  mirror the 2D versions exactly, projected into a face's own local 2D
  frame first — `flat_face_frame_cached!`'s existing *exact* one for a
  flat face (all of a flat face's own vertices, hole included, are
  genuinely coplanar — it only needed to stop requiring exactly one
  boundary loop, not to change which loop it derives the frame from), or
  a new, deliberately *approximate* `curved_face_local_frame` for a
  curved one (the face's own curve's tangent plane at its vertex
  centroid — a curved face has no single exact flat chart the way a flat
  one does, but this only answers "which resulting piece is this hole
  inside," a question this codebase already accepts a best-effort,
  fallback-guarded answer to even in the exact 2D case).
  `clip_flat_face_3d!`/`clip_curved_face_3d!` now attach each hole
  wholesale to whichever resulting piece's own boundary contains it.
  Verified via a fresh 150-trial random 3-segment stress run: **49/150
  successes (33%)**, zero label mismatches in any of them — confirms this
  is producing genuinely correct topology, not just avoiding a crash.
- **3D: within-face side-propagation fix — 50/150** — root-caused (via a
  user-prompted deep dive on one specific repro) an "edges disagree on
  side despite finding no crossing anywhere" error: a genuine near-tie
  vertex (evaluates to ~1e-13 against the new bisector, restricted to one
  face's own plane) has a global label from earlier processing that's
  itself an arbitrary tie-break; one edge touching it locally overrode
  that label (correctly, via the existing "trust the less-ambiguous
  endpoint" fallback) without ever writing the correction back, so a
  *different* edge touching the same vertex kept trusting the stale
  label — two locally-reasonable answers, disagreeing with nothing
  between them to explain why. Fixed by having `clip_flat_face_3d!`/
  `clip_curved_face_3d!` track the side of whichever vertex the boundary
  walk most recently arrived at, and prefer that over a fresh label
  lookup — past the first edge, a genuine crossing is the only thing
  allowed to flip the side (matching what the surrounding code already
  claimed to do). Confirmed as a **partial** fix, honestly: the same kind
  of disagreement can still occur *across* two different top-level cells
  that independently touch the same near-tied vertex — a deeper,
  cross-cell version of the same phenomenon as the already-known
  `sweep_topology_check` gap below, not resolved here. Verified via a
  fresh 150-trial random 3-segment stress run: **50/150 successes
  (33%)**, zero label mismatches maintained.
- **3D: made the "side-propagation" fix above more thorough, and separately
  built first-class exact-tie tracking** — two related but distinct
  follow-ups from a user-prompted question ("is pairwise comparison really
  central in higher dimensions?"). First, instrumented the three ad hoc
  "trust the less-ambiguous endpoint" fallback sites (`clip_flat_face_3d!`,
  2× `clip_curved_face_3d!`) with the library's own exact predicate and
  reran the Stage 8 repro plus the stress population: **every single time**
  that fallback actually fired, `exact_sign` returned a clean, unambiguous,
  *nonzero* sign for both endpoints — never a genuine tie. So the earlier
  "arbitrary tie-break label" framing was itself imprecise: these aren't
  ties at all, they're the floating-point crossing-finder failing to
  *locate* a crossing that provably exists (sign change between two exact,
  disagreeing endpoints). Fixed all three sites to prefer the walk's own
  already-established `cur_side` over the magnitude heuristic whenever
  available, extending the previous fix's own logic to cover this branch
  too. Verified: full suite green, but the 150-trial stress rate is flat
  (48/150) — a real correctness improvement, not a fix for the dominant
  remaining failures, which are evidently a crossing-finder robustness gap,
  not a labeling one.

  Second, separately: `clip_by_hyperplane!`'s own vertex-level loop already
  computed genuine exact ties correctly (`exact_sign` + `symbolic_tiebreak`)
  but discarded the tie itself the instant a side was picked, keeping only
  the resolution. Added `CellNode.exact_ties` (`Union{Nothing,Label}`) plus
  `record_exact_tie!`/`exact_ties` (`cell.jl`) to keep that fact — every
  feature atom ever found exactly, provably tied at a vertex, accumulated
  across every clip that touches it, independent of which side each
  individual clip resolved to. Verified on a classic degenerate case (a
  square's own center, exactly equidistant from all 4 corners by
  construction with integer input): the vertex's `exact_ties` correctly
  records 3 of the 4 tied atoms (the 4th is baked in at the vertex's
  *creation*, not re-derived via this loop, so isn't independently
  re-confirmed) — new tests in `test_exact_ties.jl`. Confirms, empirically,
  that the two follow-ups target genuinely different things: essentially
  none of today's 3-segment stress failures are real exact ties (previous
  paragraph), but real exact ties do occur on symmetric/degenerate input
  and are now tracked as a first-class fact rather than silently thrown
  away, per `project_pairwise_vs_multiway_ties.md`.
- **3D: continuation-based branch tracking for the ruled crossing-finder — 48/150 → 50/150.** User pushed back on the `exact_ties` detour ("real exact ties are non-generic — did you fix the 3 segments case yet?"), then, after confirming `exact_ties` genuinely fires zero times on the actual failure population (wired it into all 4 fallback sites, instrumented with a hit counter, ran the full stress population: 0 hits), gave a sharper instruction: use multi-way-tie representation for *codimension >= 2* cells specifically — i.e. edges in 3D (codim 2, generically the locus where exactly 3 features tie), not vertices — to fix the bug for real. Traced this to a precise, previously-undiagnosed mechanism in `ruled_trace_all_crossings` (`ruled_quadric.jl`): an edge confined to two curved surfaces (`Q1`, `Q2`) is parametrized via `Q2`'s own ruled structure, and at each sampled point along it, `Q1`'s intersection with the ruling line gives *two* candidate roots (a quadratic) — the code picked whichever was closer to a straight-line interpolation between the arc's two far-apart *endpoints*. That guess is only reliable near the endpoints; wherever the true arc curves away from that line (generic, not exceptional, for a real space curve), it silently picks the wrong root, making the traced point jump between the two sheets of `Q1 ∩ ruling-line` — registering as spurious sign changes against the third quadric (exactly the previously-observed, previously-unexplained "18/27/32 candidate crossings" pattern, well past the Bezout bound of 4) — or just as easily *hiding* a genuine crossing by jumping past it. This is the concrete sense in which the fix uses "codimension >= 2 representation": the edge is the exact locus where two already-tied feature pairs coincide (a genuine 3-way tie curve), and the previous code treated each sampled point as an independent query rather than respecting that it's *one* connected 1-manifold. Fixed by tracking the branch via continuity with the *immediately preceding* resolved point (both in the coarse scan and during each bracket's own bisection refinement), not a global guess from the two endpoints — standard numerical continuation, now actually representing the curve as connected rather than a bag of disconnected samples. Full suite green (26701/26701). Stress result: **50/150 (up from 48/150)** — a real, verified, but modest gain, not a full fix: the original Stage 8 repro still fails identically (same face, same two edges still reporting exactly 18 and 27 spurious crossings, unchanged by this fix) — for those two specific edges the true arc apparently has a genuinely different, harder problem (a real near-tangency/rapid-oscillation stretch, which the code's own existing comments already flagged as a case this scan "is not equipped to resolve exactly either way"), not a branch-selection artifact this fix targets.
- **What's still open**: (1) No single failure category dominates the
  remaining 3-segment failures — a long tail of smaller, not-yet-root-
  caused issues (numerical edge cases in the curved-vs-curved ruled-
  surface fallback, a handful of construction-flow bugs, and the cross-
  cell tie-vertex inconsistency above) accounts for the rest. Not chased
  further yet. (2) Topological consistency, even where labels are
  correct, is not solid: among 108 label-correct two-segment successes in
  an earlier stress run, `sweep_topology_check` found real structural
  issues in 101 of them — the same *kind* of "shared boundary represented
  differently on its two sides" defect found (and left open, after an
  unsafe fix attempt was reverted) in the `N=2` mixed-feature
  investigation below, apparently also present at `N=3`. Neither gap
  corrupts vertex labels, but both are real, honestly tracked in
  `test/test_3d_segments.jl` rather than fixed.
- **Topology verification tool**: `sweep_topology_check` (`N=3`) checks
  that the complex actually tiles the box — no gaps, no overlaps —
  everywhere, via a discrete combinatorial sweep (an Euler-characteristic
  argument on the cross-section at each of finitely many
  combinatorially-distinct sweep positions), not floating-point sampling.
  Complementary to the vertex-level `recompute_feature_label`
  cross-validation used elsewhere: that checks *labels* at points already
  in the complex; this checks *topology* everywhere at once. Building it
  immediately found, and led to fixing, a real bug: points-only 3D
  construction (believed fully solid) had faces whose own boundary walk
  repeated the same edge — a genuinely zero-area degenerate face, present
  in every configuration tried. Root cause: `supersede!` (`src/complex/
  cell.jl`) could splice a *merged* id (from `weld_duplicate_edges!`
  collapsing two independently-created copies of the same geometric edge
  — itself an existing, intentional cleanup for two top cells clipped
  independently against different bisectors at a shared tie point) into a
  face that already referenced it separately, producing a duplicate
  instead of a clean collapse. Fixed by deduplicating a patched parent's
  `subcells` after any splice (only for `dim>=2` — a `dim=1` edge's
  `subcells` are positional endpoints, not an unordered set, and must
  never be deduplicated), plus a new `remove_degenerate_faces!`
  (`src/algorithm/multi_points.jl`) that removes any face left unable to
  close into a boundary loop at all — the dim=2 analogue of the dim=1
  "edge collapsed to `[v,v]`" cleanup `weld_near_duplicate_vertices!`
  already had. Verified via a 40-trial stress run (3–10 points) and a
  25-trial run (10–25 points): both vertex-level cross-validation and
  `sweep_topology_check` now come back fully clean, 0 failures. There's
  also an `N=2` method (`sweep_topology_check(cx::CellComplex{2}, ...)`),
  sweeping a *line* rather than a plane — points-only and simple
  points+segments configs are fully clean there too, but richer mixed 2D
  configs (3+ points/segments) surface a **real, not-yet-fixed** defect:
  a shared boundary between two adjacent cells can end up as two separate
  edges (identical `curve`, one shared endpoint, but *overlapping* spans)
  instead of one, in ~40% of random 3–7-entry stress trials — distinct
  from the task #42/#49 gap below (confirmed: that check never warns on
  this repro). Root-caused precisely: when a new feature is clipped
  independently against two related pre-existing cells (the same pattern
  `weld_duplicate_edges!` already exists for), only the cell that *also*
  borders a third cell along the same stretch gets subdivided there — the
  other keeps one coarse edge. A targeted fix (splitting the coarser edge
  to match) worked on the minimal repro but broke down under broader
  stress testing (only fixed a fraction of cases, and introduced a new
  crash on previously-working input), so it was reverted rather than
  shipped unsafe — a real fix likely needs propagating a new subdivision
  point to every edge that geometrically passes through it, not just a
  pairwise match. Doesn't appear to corrupt final labels, just the
  internal edge structure; tracked honestly (a testset that expects and
  asserts this specific failure) rather than fixed, in
  `test/test_sweep_topology.jl`.
- **3D with curved bisector *planes* (triangles) is not built yet** — the
  next real jump in difficulty. A triangle's own interior feature (`k=2`)
  gives a bisector whose intersection with an existing face is no longer
  reducible to the 2D case the same way a line's is; intersecting two
  quadric *surfaces* in full generality (not just one flat face against
  one already-proven-2D-reducible bisector) is a new geometric primitive
  this codebase doesn't have yet. See `triangulation-difficulties.md` for
  related notes.
- **Out of scope for now**: exact coincidences (e.g. exact right-angle
  junctions) — the generic-case assumption is deliberate for v1; see the
  `house_exact` case in the dev dashboard for a live example kept around on
  purpose.
- `first-prototype/` holds the original 2D-only, half-edge/DCEL-based first
  attempt. It's deprecated and kept only for historical reference — all
  active development is in this repository's top-level `src/`.

## Install

```sh
git clone <this repo> mesh-voronoi
cd mesh-voronoi
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Run the tests

```sh
julia --project=. test/runtests.jl
```

## Plotting (optional)

`GLMakie` is a weak dependency — the core package never requires it. Load it
alongside `MeshVoronoi` in the same session to get `plot_cells_2d`:

```julia
using MeshVoronoi, GLMakie
points = [SVector(0.0, 0.0), SVector(2.0, 0.0)]
cx = points_complex(points)
# ... pick live top-cell ids, then:
plot_cells_2d(cx, cell_ids_colors, points)
```

## Interactive demo

Pure OpenGL — a raw GLFW window and a hand-written fragment shader do all the
drawing (no plotting library): the shader evaluates every feature's distance
formula directly, per pixel, so curved bisectors render smoothly at any zoom.

```sh
julia --project=. examples/interactive_demo.jl
```

Click empty space to add a point; click-drag to add a segment; right-click
to clear; scroll to zoom. Hovering tints the cell/edge/vertex under the
cursor and shows the constraints that define it. `R` refines every segment
in half, `S` saves a named preset, `L` loads one (including the built-in
`house` and `benchmark` presets).

## Benchmark

```sh
julia --project=. examples/benchmark_bvh.jl
```

Measures how much a BVH over cached cell bounding boxes speeds up point
location on a large points-only construction.
