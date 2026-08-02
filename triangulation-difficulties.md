# Triangulation difficulties (historical note)

This documents the difficulties encountered while the interactive demo
(`examples/interactive_demo.jl`) rendered hover highlights by reconstructing
a cell's polygon on the CPU and triangulating it for a `GL_TRIANGLES` fill.
That approach has since been **removed entirely** and replaced with a
shader-based design (the background fragment shader tints pixels directly,
per-pixel, using the same distance-field evaluation it already performs for
the main diagram — see the demo's top-of-file comment and task #44). This
note exists so a future implementer doesn't have to rediscover the same
problems if triangulation is ever reintroduced for some other purpose (an
export format, a static report image, etc.).

## Why triangulation was attempted at all

Before the redesign, hovering a cell/edge/vertex highlighted it by finding
the hovered cell (`find_hover_target`), reconstructing its polygon
(`polygon_vertices_2d`, walking the cell's boundary edges and tessellating
curved ones into polyline samples), and triangulating that polygon
(`ear_clip_simple_polygon`) so it could be filled with `GL_TRIANGLES`. This
seemed natural because the output complex's cells are stored as boundary
graphs, not pre-triangulated meshes — filling one on screen "should" mean
triangulating it. In hindsight this entire code path was solving a problem
the renderer didn't need to solve, since the shader can answer "is this
pixel inside the hovered cell" directly.

## Difficulty 1: cells are not convex

The first bug (bug2 preset) was a point-in-polygon fill test that assumed
convexity. A segment's own Voronoi territory against a nearby point site is
bounded by a parabola; the cell on the *concave* side of that parabola is
genuinely non-convex, and simple centroid/fan-based point-in-polygon checks
gave wrong answers there. Any polygon-fill approach for these cells has to
handle arbitrary simple (non-convex) polygons from the start — there is no
"assume convex, special-case the rest" shortcut available in this domain.

## Difficulty 2: ear-clipping got stuck on near-duplicate vertices

`ear_clip_simple_polygon` implements standard ear-clipping (repeatedly
removing a convex vertex whose triangle contains no other polygon vertex).
It got numerically stuck — leaving part of the polygon untriangulated,
silently, with no error — whenever the vertex list contained duplicate or
near-duplicate points. Two independent sources produced these:

- `polygon_vertices_2d`'s own boundary walk closes back to its start point,
  so `poly[end] == poly[1]` exactly (a literal duplicate coordinate).
- Curved-edge tessellation (sampling a parabolic/quadric boundary into a
  polyline) can place adjacent samples extremely close together near high
  curvature, producing *near*-duplicates that ear-clipping's convexity/
  containment tests (built on exact cross-product signs) treat inconsistently.

This was fixed at the time by deduplicating consecutive near-identical
vertices (`atol=1e-9`, including the wrap-around last/first pair) before
handing the polygon to the ear-clipper. The fix worked, but it is a patch
on a symptom: any future tessellation parameter change, or any construction
routine that produces tighter-than-expected sample spacing, can reintroduce
the same failure mode. There is no principled bound on "how close is too
close" for ear-clipping's cross-product tests short of exact/adaptive
arithmetic, which the demo's rendering path never had.

## Difficulty 3: detecting the bug at all was unreliable

Visual inspection did not reliably reveal a partially-untriangulated
polygon (a missing sliver of a large cell can be a few pixels wide). The
bug was actually confirmed via a shoelace-formula area of the reconstructed
polygon compared against the summed area of the emitted triangles — a
mismatch meant the ear-clipper had dropped a region. This is a good
technique in general but wasn't in place from the start, so the bug shipped
silently for a while before being caught.

A related trap: an early "overlap detector" (`polys_overlap`, checking
whether one polygon's centroid or edge-midpoints fall inside another) threw
many false positives — 70 flagged "overlaps" on one preset — because it is
itself unreliable for non-convex cells. Every flagged case had to be
cross-validated against a brute-force point-sampling oracle
(`brute_force_label_multi`) or the real `find_containing_cell` before being
trusted as a genuine bug, rather than trusting the heuristic's own verdict.

## Difficulty 4: a genuine self-intersecting cell boundary

Even after the deduplication fix, one cell in the bug5 preset (id 685, in a
10-entry construction) produced a polygon with a real geometric
self-intersection — edge (7,8) crossed edge (22,23) — not a numerical
artifact of tessellation. `polygon_vertices_2d`'s boundary walk only checks
vertex *adjacency* (does the graph form a closed cycle), not whether the
resulting polygon is *simple* (non-self-intersecting) once vertex
coordinates are taken into account. No amount of triangulation-side
patching (deduplication, ear-clip robustness) can fix this, because the
input to triangulation is already invalid — the underlying subcells
constructed by `insert_own_lines!`/`clip_by_hyperplane!` do not actually
form a simple cycle for this cell, even though every individual edge
looked locally correct. This is suspected to be related to (but not
conclusively identified as) the multi-crossing case tracked separately
(task #26: a curved bisector crossing a single edge more than once) — it
was not caught by that case's own detection logic, so the relationship is
unconfirmed. This is a construction-level bug, independent of rendering,
and remains open (task #43).

## The broader lesson

Every difficulty above is a consequence of the same root cause: polygon
reconstruction and triangulation is a *second, independent* representation
of "what does this cell look like," built by walking a boundary graph and
re-deriving geometry that the underlying distance-field computation already
gets right by construction. Bugs in that second representation (duplicate
vertices, non-convexity, self-intersection) do not imply bugs in the
diagram itself — but they were indistinguishable from real diagram bugs
without careful oracle cross-validation, which cost significant debugging
time. The shader-based redesign sidesteps the whole class: it asks "which
feature(s) win at this pixel" using the exact same per-pixel distance
comparison that produces the main diagram, with no separate polygon,
no boundary walk, and no triangulation step to get wrong.
