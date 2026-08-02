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
- **Out of scope for now**: exact coincidences (e.g. exact right-angle
  junctions) — the generic-case assumption is deliberate for v1; see the
  `house_exact` case in the dev dashboard for a live example kept around on
  purpose. **3D is not built yet** — the feature/bisector/label machinery is
  already dimension-generic, but inserting a bisector *surface* (`k=2`,
  finding its own trace on the existing 2-skeleton) is the one genuinely new
  step still needed; see `triangulation-difficulties.md` for related notes.
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
