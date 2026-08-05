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
  hyperplane regardless of dimension, so this needed *no* code changes at
  all — every piece (`init_bbox_complex`, `insert_point!`, the merge/weld
  passes) was already written generically over `N`. Verified with vertex-
  level cross-validation against a brute-force oracle (zero mismatches
  across several point counts) and a genuine 8-way tie at a cube's exact
  center resolving to one correctly-labeled vertex.
- **3D points + line segments works, within a stated v1 scope**
  (`insert_segment!` at `N=3`): a segment's own interior feature is a
  genuinely curved quadric surface even in 3D (not just `N=2`), so this
  needed the actual new machinery — `clip_top_cell_3d!`/
  `clip_flat_face_3d!` walk a 3D cell's own 2-skeleton (faces), restrict
  the bisector to each flat face's own local 2D frame
  (`restrict_to_plane`), reuse the existing, already-proven 2D
  curved-edge machinery there, and cap the resulting cut with a new
  curved face — the 3D analogue of `clip_top_cell_2d!`'s own 2D boundary
  walk, one dimension up. Verified with vertex-level cross-validation
  (zero mismatches) across several point+segment configurations and a
  60-trial random stress run (0 failures beyond the stated scope limits
  below, out of 56 successful trials). v1 scope, deliberately: every face
  touched must be flat (an already-curved face, e.g. a cap left by an
  earlier insertion, can't be split further yet), with no holes, and
  crossed at exactly 0 or 2 points total per face (a single connected
  cut — the 3D analogue of a curved bisector crossing one edge twice,
  already handled at `N=2`, isn't supported yet at `N=3`). Every one of
  these errors loudly rather than being silently mishandled.
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
