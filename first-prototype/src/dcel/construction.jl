"""
Split half-edge `h` (and its twin) at point `x`, which must lie in the
*interior* of that edge. Returns the new vertex id.
"""
function split_edge!(dcel::DCEL, h::Int, x::SVector{2,Float64})
    he = dcel.halfedges[h]
    t = he.twin
    ht = dcel.halfedges[t]

    if he.next == t
        # h/t form an isolated dangling tip (h's destination has nothing
        # else attached -- exactly what a free segment endpoint produces,
        # e.g. this project's open-chimney examples): the face wraps
        # straight back via the twin, so h.next == t and (equivalently)
        # t.prev == h. The general case below assumes old_h_next is some
        # *other*, unrelated edge; here it *is* t, which this same call is
        # also about to re-origin to vnew -- aliasing the two roles and
        # corrupting hnew's implied destination. Build the "there and back"
        # tip explicitly instead of trying to force the general formula.
        old_t_origin = ht.origin   # = B, h's destination
        push!(dcel.vertices, Vtx(x))
        vnew = length(dcel.vertices)
        hnew = length(dcel.halfedges) + 1
        tnew = length(dcel.halfedges) + 2
        push!(dcel.halfedges, HalfEdge(vnew, tnew, tnew, h, he.face, he.curve))         # hnew: vnew->B, next=tnew, prev=h
        push!(dcel.halfedges, HalfEdge(old_t_origin, hnew, t, hnew, ht.face, ht.curve))  # tnew: B->vnew, next=t, prev=hnew
        he.next = hnew
        ht.origin = vnew
        ht.prev = tnew
        return vnew
    end

    old_h_next = he.next
    old_t_prev = ht.prev
    old_t_origin = ht.origin

    push!(dcel.vertices, Vtx(x))
    vnew = length(dcel.vertices)

    hnew = length(dcel.halfedges) + 1
    tnew = length(dcel.halfedges) + 2
    push!(dcel.halfedges, HalfEdge(vnew, tnew, old_h_next, h, he.face, he.curve))
    push!(dcel.halfedges, HalfEdge(old_t_origin, hnew, t, old_t_prev, ht.face, ht.curve))

    he.next = hnew          # h: A -> vnew (twin still t)
    ht.origin = vnew        # t: vnew -> A (twin still h)
    ht.prev = tnew

    dcel.halfedges[old_h_next].prev = hnew
    dcel.halfedges[old_t_prev].next = tnew

    return vnew
end

"""
The vertex id at point `x`, which must lie on half-edge `h`'s curve within
that edge's extent -- reusing an existing endpoint if `x` coincides with one,
otherwise splitting the edge to create a new vertex there. Also returns two
"anchor" half-edges: the specific outgoing edge from that vertex within
`h`'s own face, and the corresponding one within `h`'s twin's face.

`split_face!` needs one of these, not just the vertex id, to know exactly
where in a face's boundary cycle to splice a new chord. A single vertex can
appear at more than one position around one face's cycle -- a dangling
open-segment tip revisits its own vertex on the way back out -- so looking
the vertex up by identity alone can't tell which occurrence a given
crossing actually happened at; splicing at the wrong one silently produces
a corrupted, self-overlapping split.
"""
function ensure_vertex!(dcel::DCEL, h::Int, x::SVector{2,Float64}; atol=GEOM_ATOL)
    he = dcel.halfedges[h]
    t = he.twin
    o = vcoord(dcel, he.origin)
    d = vcoord(dcel, dcel.halfedges[t].origin)
    if norm(x - o) < atol
        return he.origin, h, dcel.halfedges[t].next
    elseif norm(x - d) < atol
        return dcel.halfedges[t].origin, he.next, t
    else
        vnew = split_edge!(dcel, h, x)
        return vnew, dcel.halfedges[h].next, dcel.halfedges[h].twin
    end
end

"""
Resolve an `ensure_vertex!` anchor pair down to the one specific half-edge
that actually lies on `face_id`'s boundary cycle: `same_anchor` if `h`'s own
face is `face_id`, `twin_anchor` if `h`'s twin's face is. Falls back to a
plain vertex-identity search (the pre-anchor behavior, with its same
ambiguity if the vertex repeats within `face_id`'s own cycle) only for the
rarer case of a vertex shared by more than these two faces, e.g. a
higher-degree vertex where 3+ faces meet.
"""
function anchor_for(dcel::DCEL, h::Int, same_anchor::Int, twin_anchor::Int, face_id::Int)
    dcel.halfedges[h].face == face_id && return same_anchor
    dcel.halfedges[dcel.halfedges[h].twin].face == face_id && return twin_anchor
    v = dcel.halfedges[same_anchor].origin
    hs = face_halfedges(dcel, face_id)
    idx = findfirst(hh -> dcel.halfedges[hh].origin == v, hs)
    return idx === nothing ? nothing : hs[idx]
end

face_has_vertex(dcel::DCEL, face_id::Int, v::Int) = any(h -> dcel.halfedges[h].origin == v, face_halfedges(dcel, face_id))

"""
Point-in-face test that respects curved boundaries: ray-casts from `x` along
a fixed non-axis-aligned direction and counts parity of crossings with the
face's boundary curves (not their polygon chords).
"""
function point_in_face(dcel::DCEL, face_id::Int, x::SVector{2,Float64})
    dir = SVector(1.0, 0.6180339887498949)
    n = normalize(SVector(-dir[2], dir[1]))
    ray = Line(n, dot(n, x))
    tx = param_of(ray, x)

    count = 0
    for h in face_halfedges(dcel, face_id)
        he = dcel.halfedges[h]
        o = vcoord(dcel, he.origin)
        d = vcoord(dcel, dcel.halfedges[he.twin].origin)
        lo, hi = minmax(param_of(he.curve, o), param_of(he.curve, d))
        for p in intersect_points(ray, he.curve)
            tp_edge = param_of(he.curve, p)
            (lo - GEOM_ATOL <= tp_edge <= hi + GEOM_ATOL) || continue
            param_of(ray, p) > tx + GEOM_ATOL || continue
            count += 1
        end
    end
    return isodd(count)
end

function locate_face(dcel::DCEL, x::SVector{2,Float64}, candidates)
    for f in candidates
        f == dcel.outer_face && continue
        point_in_face(dcel, f, x) && return f
    end
    return nothing
end

"""
Split `face_id` into two faces by adding a chord along `curve` between
existing boundary vertices, given as anchor half-edges `e1`/`e2` (see
`ensure_vertex!`/`anchor_for`) -- each already known to lie on `face_id`'s
own boundary cycle, at the exact position a chord endpoint should splice
in. Looking these up by anchor rather than plain vertex id matters because
a vertex can appear at more than one position around a single face's cycle
(a dangling open-segment tip revisits its own vertex); only the anchor
identifies which occurrence is meant. Returns `(face_id, new_face_id)`.
"""
function split_face!(dcel::DCEL, face_id::Int, e1::Int, e2::Int, curve::Curve)
    hs = face_halfedges(dcel, face_id)
    i1 = findfirst(==(e1), hs)
    i2 = findfirst(==(e2), hs)
    if i1 === nothing || i2 === nothing
        v1, v2 = dcel.halfedges[e1].origin, dcel.halfedges[e2].origin
        origins = [dcel.halfedges[h].origin for h in hs]
        coords = [vcoord(dcel, o) for o in origins]
        error("split_face!: e1=$e1 (v1=$v1, $(i1===nothing ? "missing" : "found")) or e2=$e2 (v2=$v2, $(i2===nothing ? "missing" : "found")) not on face $face_id's boundary.\n  boundary origins=$origins\n  boundary coords=$coords\n  v1 coord=$(vcoord(dcel,v1))\n  v2 coord=$(vcoord(dcel,v2))\n  curve=$curve")
    end
    e1, e2 = hs[i1], hs[i2]
    v1, v2 = dcel.halfedges[e1].origin, dcel.halfedges[e2].origin
    @assert e1 != e2 "split_face!: e1 and e2 must be distinct boundary half-edges"

    old_e1_prev = dcel.halfedges[e1].prev
    old_e2_prev = dcel.halfedges[e2].prev

    hnew = length(dcel.halfedges) + 1
    htwin = length(dcel.halfedges) + 2
    push!(dcel.halfedges, HalfEdge(v1, htwin, e2, old_e1_prev, face_id, curve))
    push!(dcel.halfedges, HalfEdge(v2, hnew, e1, old_e2_prev, face_id, curve))

    dcel.halfedges[old_e1_prev].next = hnew
    dcel.halfedges[e2].prev = hnew
    dcel.halfedges[old_e2_prev].next = htwin
    dcel.halfedges[e1].prev = htwin

    new_face_id = length(dcel.faces) + 1
    push!(dcel.faces, Face(htwin, nothing))
    dcel.faces[face_id].halfedge = hnew

    h = hnew
    steps = 0
    while true
        dcel.halfedges[h].face = face_id
        h = dcel.halfedges[h].next
        steps += 1
        h == hnew && break
        steps <= length(dcel.halfedges) || error("split_face!: relabel walk from hnew=$hnew did not close after $steps steps -- corrupted DCEL (face_id=$face_id, v1=$v1, v2=$v2)")
    end
    h = htwin
    steps = 0
    while true
        dcel.halfedges[h].face = new_face_id
        h = dcel.halfedges[h].next
        steps += 1
        h == htwin && break
        steps <= length(dcel.halfedges) || error("split_face!: relabel walk from htwin=$htwin did not close after $steps steps -- corrupted DCEL (face_id=$face_id, v1=$v1, v2=$v2)")
    end

    return face_id, new_face_id
end

"""
Insert `curve` into the arrangement, splitting every face (and edge) it
passes through. If `candidate_faces` is given, the curve is only traced
through that set of faces (and their boundary edges) -- used to insert a
bisector curve local to one region of the arrangement without touching the
rest of it. Returns the updated set of face ids now covering the same
region as `candidate_faces` did before the call (old ids that survived plus
any newly created ones); with `candidate_faces=nothing` (the default),
returns the analogous set over the whole (non-outer) arrangement.

If `on_split` is given, it's called as `on_split(f_old, f_new)` immediately
after every `split_face!`, letting a caller track exactly which face each
new one descends from (e.g. to maintain a precise ancestry map) instead of
inferring it after the fact from the returned scope, which can't
distinguish which of several unrelated input faces a new fragment came from
when `candidate_faces` spans more than one.

Works by finding every point where `curve` crosses an in-scope edge, sorting
those crossings along `curve`'s own parameter, and walking consecutive pairs:
each such pair spans an arc of `curve` lying entirely inside one existing
face (since no in-scope edge crosses it in between), which becomes a new
chord splitting that face.
"""
function insert_curve!(dcel::DCEL, curve::Curve; candidate_faces::Union{Nothing,Set{Int}}=nothing, on_split=nothing)
    global_scope = candidate_faces === nothing
    scope = global_scope ? Set(f for f in eachindex(dcel.faces) if f != dcel.outer_face && dcel.faces[f].halfedge != -1) : copy(candidate_faces)

    edge_ids = Int[]
    seen = falses(length(dcel.halfedges))
    for h in eachindex(dcel.halfedges)
        seen[h] && continue
        twin = dcel.halfedges[h].twin
        seen[h] = true
        seen[twin] = true
        f1, f2 = dcel.halfedges[h].face, dcel.halfedges[twin].face
        (f1 == 0 || f2 == 0) && continue   # both sides dead (merged away) -- not part of the live arrangement
        if global_scope || f1 in scope || f2 in scope
            push!(edge_ids, h)
        end
    end

    crossings = Tuple{Float64,Int,SVector{2,Float64}}[]
    for h in edge_ids
        he = dcel.halfedges[h]
        o = vcoord(dcel, he.origin)
        d = vcoord(dcel, dcel.halfedges[he.twin].origin)
        lo, hi = minmax(param_of(he.curve, o), param_of(he.curve, d))
        for x in intersect_points(curve, he.curve)
            tx_edge = param_of(he.curve, x)
            (lo - GEOM_ATOL <= tx_edge <= hi + GEOM_ATOL) || continue
            push!(crossings, (param_of(curve, x), h, x))
        end
    end
    isempty(crossings) && return scope

    sort!(crossings, by=first)
    # Dedup by *world position*, not curve parameter: two crossings against
    # different existing edges that both land on the same shared vertex can
    # accumulate slightly different parameter values (each computed via a
    # separate root-find), even though they're geometrically the same point.
    dedup = [crossings[1]]
    for c in crossings[2:end]
        norm(c[3] - dedup[end][3]) > GEOM_ATOL && push!(dedup, c)
    end
    crossings = dedup
    length(crossings) < 2 && return scope   # tangential touch only -- generic case excludes this

    for i in 1:length(crossings)-1
        t1, h1, x1 = crossings[i]
        t2, h2, x2 = crossings[i+1]
        xmid = point_at(curve, (t1 + t2) / 2)
        face_id = locate_face(dcel, xmid, scope)
        v1, anchor1_same, anchor1_twin = ensure_vertex!(dcel, h1, x1)
        v2, anchor2_same, anchor2_twin = ensure_vertex!(dcel, h2, x2)
        # `locate_face`'s ray-cast is numerical and can occasionally disagree
        # with the actual topology once v1/v2 are pinned down (e.g. very
        # close to another face's boundary). Trust the structural fact --
        # which face in scope actually has both v1 and v2 on its boundary --
        # over the midpoint ray-cast whenever they disagree.
        if !(face_id !== nothing && face_has_vertex(dcel, face_id, v1) && face_has_vertex(dcel, face_id, v2))
            face_id = nothing
            for f in scope
                if face_has_vertex(dcel, f, v1) && face_has_vertex(dcel, f, v2)
                    face_id = f
                    break
                end
            end
        end
        face_id === nothing && continue
        anchor1 = anchor_for(dcel, h1, anchor1_same, anchor1_twin, face_id)
        anchor2 = anchor_for(dcel, h2, anchor2_same, anchor2_twin, face_id)
        f_old, f_new = split_face!(dcel, face_id, anchor1, anchor2, curve)
        on_split !== nothing && on_split(f_old, f_new)
        # A later interval in this same call may need to locate into the
        # piece we just created (e.g. the curve re-enters a face that an
        # earlier interval already split) -- keep `scope` current. Faces the
        # curve never touches at all stay in `scope` unchanged, which is
        # exactly right: they're still valid pieces of the region a caller
        # asked us to refine, just not ones this particular curve affected.
        push!(scope, f_new)
    end
    return scope
end
