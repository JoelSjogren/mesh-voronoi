"""
Point-in-convex-polygon test for cross-validation only (not part of the
library): `verts` must already be in cyclic order (as `polygon_vertices_2d`
returns) -- `x` is inside iff it's on the same rotational side of every
edge.
"""
function point_in_convex_polygon(verts::Vector{<:SVector{2,Float64}}, x::SVector{2,Float64}; atol=1e-9)
    n = length(verts)
    signs = Float64[]
    for i in 1:n
        a, b = verts[i], verts[mod1(i + 1, n)]
        edge = b - a
        push!(signs, edge[1] * (x[2] - a[2]) - edge[2] * (x[1] - a[1]))
    end
    return all(s -> s >= -atol, signs) || all(s -> s <= atol, signs)
end

@testset "two_points_complex: N=2 grid cross-validation against oracle" begin
    pa, pb = SVector(0.0, 0.0), SVector(2.0, 0.5)
    cx, a_id, b_id, cut_id = two_points_complex(pa, pb)
    lo, hi = padded_bbox([pa, pb])

    a_poly = polygon_vertices_2d(cx, a_id)
    b_poly = polygon_vertices_2d(cx, b_id)

    mismatches = 0
    checked = 0
    for x in range(lo[1], hi[1], length=41), y in range(lo[2], hi[2], length=41)
        pt = SVector(x, y)
        expected = brute_force_label_two_points(pa, pb, pt)
        length(expected) > 1 && continue   # skip points on/near the true tie boundary
        in_a = point_in_convex_polygon(a_poly, pt)
        in_b = point_in_convex_polygon(b_poly, pt)
        (in_a && in_b) && continue   # skip points on the shared edge (both report true near it)
        (!in_a && !in_b) && continue  # outside both (shouldn't happen inside the bbox, but be safe)
        checked += 1
        got = in_a ? Set([1]) : Set([2])
        got != expected && (mismatches += 1)
    end
    @test checked > 1000   # sanity: the grid actually exercised plenty of points
    @test mismatches == 0
end

@testset "two_points_complex: labels are correctly assigned" begin
    pa, pb = SVector(-1.0, 0.3), SVector(1.5, -0.7)
    cx, a_id, b_id, cut_id = two_points_complex(pa, pb)
    @test cx.nodes[a_id].label == Set([Set([1])])
    @test cx.nodes[b_id].label == Set([Set([2])])
    @test cx.nodes[cut_id].label == Set([Set([1]), Set([2])])
end

@testset "two_points_complex: works at N=3 too" begin
    pa, pb = SVector(0.0, 0.0, 0.0), SVector(1.0, 1.0, 1.0)
    cx, a_id, b_id, cut_id = two_points_complex(pa, pb)
    for p in descendant_points(cx, a_id)
        @test sqdist(AffineQuadratic(pa), p) <= sqdist(AffineQuadratic(pb), p) + 1e-9
    end
    for p in descendant_points(cx, b_id)
        @test sqdist(AffineQuadratic(pb), p) <= sqdist(AffineQuadratic(pa), p) + 1e-9
    end
    @test cx.nodes[a_id].label == Set([Set([1])])
    @test cx.nodes[b_id].label == Set([Set([2])])
end
