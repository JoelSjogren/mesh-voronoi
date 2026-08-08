"""
The convex hull of `points` (2D), vertices in counter-clockwise order
(collinear points along an edge dropped). Andrew's monotone chain.
Requires at least 3 affinely-independent points -- errors otherwise rather
than silently returning a degenerate hull.
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
The polygon from pushing every edge of a convex polygon `hull`
(counter-clockwise, as `convex_hull_2d` returns) outward along its own
normal by `D`, then re-intersecting adjacent offset edges for the new
vertices -- exact for an arbitrary convex hull, unlike scaling about a
fixed point (which can misplace a vertex relative to its own normal cone;
confirmed by a concrete counterexample). `offset[i]` belongs to `hull[i]`'s
own normal cone.
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
