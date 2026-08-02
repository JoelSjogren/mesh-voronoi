struct Vtx
    coord::SVector{2,Float64}
end

mutable struct HalfEdge
    origin::Int
    twin::Int
    next::Int
    prev::Int
    face::Int
    curve::Curve
end

"""
`label` is `nothing` until labeled, then a `Set` of classification atoms
(see `src/algorithm/pipeline.jl` for the atom shape) -- the DCEL kernel
itself doesn't need to know their internal structure, only that they
support `==` (used by `merge_equal_label_faces!`).
"""
mutable struct Face
    halfedge::Int
    label::Union{Nothing,Set}
end

mutable struct DCEL
    vertices::Vector{Vtx}
    halfedges::Vector{HalfEdge}
    faces::Vector{Face}
    outer_face::Int
end

vcoord(dcel::DCEL, i::Int) = dcel.vertices[i].coord

"""
The half-edge ids forming `face_id`'s boundary cycle, in order.
"""
function face_halfedges(dcel::DCEL, face_id::Int)
    start = dcel.faces[face_id].halfedge
    hs = Int[start]
    h = dcel.halfedges[start].next
    while h != start
        push!(hs, h)
        h = dcel.halfedges[h].next
        @assert length(hs) <= length(dcel.halfedges) "face boundary walk did not close -- corrupted DCEL"
    end
    return hs
end

function face_polygon(dcel::DCEL, face_id::Int)
    return [vcoord(dcel, dcel.halfedges[h].origin) for h in face_halfedges(dcel, face_id)]
end

"""
Bounding-box-only DCEL: one bounded rectangular face plus the sentinel
unbounded outer face.
"""
function init_bbox_dcel(xmin, xmax, ymin, ymax)
    v = SVector{2,Float64}[(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
    vertices = Vtx.(v)

    bottom = Line(SVector(0.0, 1.0), ymin)
    right  = Line(SVector(1.0, 0.0), xmax)
    top    = Line(SVector(0.0, 1.0), ymax)
    left   = Line(SVector(1.0, 0.0), xmin)

    halfedges = HalfEdge[
        HalfEdge(1, 5, 2, 4, 1, bottom),  # 1: v1->v2
        HalfEdge(2, 6, 3, 1, 1, right),   # 2: v2->v3
        HalfEdge(3, 7, 4, 2, 1, top),     # 3: v3->v4
        HalfEdge(4, 8, 1, 3, 1, left),    # 4: v4->v1
        HalfEdge(2, 1, 8, 6, 2, bottom),  # 5: v2->v1
        HalfEdge(3, 2, 5, 7, 2, right),   # 6: v3->v2
        HalfEdge(4, 3, 6, 8, 2, top),     # 7: v4->v3
        HalfEdge(1, 4, 7, 5, 2, left),    # 8: v1->v4
    ]
    faces = Face[Face(1, nothing), Face(5, nothing)]
    return DCEL(vertices, halfedges, faces, 2)
end
