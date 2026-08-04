"""
The convex hull of `points` (2D), returned as its vertices in
counter-clockwise order, each appearing once (collinear points along a hull
edge are dropped, keeping only genuine corners). Andrew's monotone chain:
sort by `(x,y)`, then build the lower and upper chains independently, each
by a simple stack scan that pops the last point whenever it would make a
non-left (clockwise or straight) turn -- the standard `O(n log n)`
construction, no dependency needed for it.

Requires at least 3 affinely-independent points among `points` (the generic
case this project targets, per the project's own scope notes on generic vs
degenerate input) -- errors otherwise, rather than silently returning a
degenerate (0- or 1-dimensional) "hull".
"""
function convex_hull_2d(points::Vector{Pt{2,Float64}})
    pts = sort(unique(points); by=p -> (p[1], p[2]))
    length(pts) >= 3 || error("convex_hull_2d: need at least 3 distinct points")
    cross(o, a, b) = (a[1] - o[1]) * (b[2] - o[2]) - (a[2] - o[2]) * (b[1] - o[1])

    function chain(pts)
        h = Pt{2,Float64}[]
        for p in pts
            while length(h) >= 2 && cross(h[end-1], h[end], p) <= 0
                pop!(h)
            end
            push!(h, p)
        end
        return h
    end

    lower = chain(pts)
    upper = chain(reverse(pts))
    hull = vcat(lower[1:end-1], upper[1:end-1])
    length(hull) >= 3 || error("convex_hull_2d: input is collinear -- doesn't affinely span the plane (need >= 3 affinely independent points)")
    return hull
end

"""
The polygon obtained by pushing every edge of a convex polygon `hull`
(vertices in counter-clockwise order, as `convex_hull_2d` returns) outward
along its own outward normal by a fixed distance `D`, then re-intersecting
each pair of adjacent offset edges for the new vertices -- see the
"layer at infinity" planning report for why this (not scaling the hull by a
factor about a fixed point, which can misplace a vertex relative to its own
normal cone -- confirmed by a concrete counterexample there) is the
construction that's exact for an arbitrary convex hull, unconditionally.

Returns the offset polygon's vertices, in the same cyclic order as `hull`
(`offset[i]` is the new vertex corresponding to the offset lines of edges
`hull[i-1]->hull[i]` and `hull[i]->hull[i+1]` meeting -- i.e. it "belongs
to" `hull[i]`'s own normal cone, the `<hull[i],∞>` label).
"""
function offset_polygon(hull::Vector{Pt{2,Float64}}, D::Float64)
    n = length(hull)
    n >= 3 || error("offset_polygon: need a genuine polygon (>= 3 vertices)")
    outward_normal(a, b) = let d = b - a
        SVector(d[2], -d[1]) / norm(d)
    end
    offset_lines = Vector{Tuple{Pt{2,Float64},Pt{2,Float64}}}(undef, n)
    for i in 1:n
        a, b = hull[i], hull[mod1(i + 1, n)]
        n̂ = outward_normal(a, b)
        offset_lines[i] = (a + D * n̂, b + D * n̂)
    end
    new_verts = Vector{Pt{2,Float64}}(undef, n)
    for i in 1:n
        prev = mod1(i - 1, n)
        (p1, p2), (q1, q2) = offset_lines[prev], offset_lines[i]
        d1, d2 = p2 - p1, q2 - q1
        denom = d1[1] * d2[2] - d1[2] * d2[1]
        abs(denom) < 1e-12 && error("offset_polygon: adjacent edges at hull vertex $i are parallel -- degenerate hull, outside the generic case this targets")
        t = ((q1[1] - p1[1]) * d2[2] - (q1[2] - p1[2]) * d2[1]) / denom
        new_verts[i] = p1 + t * d1
    end
    return new_verts
end
