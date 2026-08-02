"""
Golden-angle hue rotation: a simple, dependency-free way to get visually
distinct colors for an a-priori-unknown number of labels.
"""
function hue_color(k::Int)
    h = mod(k * 137.508, 360.0)
    s, v = 0.55, 0.95
    c = v * s
    x = c * (1 - abs(mod(h / 60, 2) - 1))
    m = v - c
    r, g, b = if h < 60
        (c, x, 0.0)
    elseif h < 120
        (x, c, 0.0)
    elseif h < 180
        (0.0, c, x)
    elseif h < 240
        (0.0, x, c)
    elseif h < 300
        (x, 0.0, c)
    else
        (c, 0.0, x)
    end
    return RGBf(r + m, g + m, b + m)
end

"""
Lighten (`amount > 0`) or darken (`amount < 0`) an RGB color toward
white/black -- used to distinguish the two sides of a codimension-1 feature
(same base hue, different shade) without needing a whole separate hue.
"""
function shade(c::RGBf, amount::Float64)
    amount >= 0 && return RGBf(c.r + (1 - c.r) * amount, c.g + (1 - c.g) * amount, c.b + (1 - c.b) * amount)
    return RGBf(c.r * (1 + amount), c.g * (1 + amount), c.b * (1 + amount))
end

"""
One deterministic representative atom from a tied label `Set`, for coloring
purposes -- which specific tied atom "wins" the color doesn't matter, only
that every occurrence of the *same* label set consistently picks the same
one.
"""
canonical_atom(label::Set) = first(sort(collect(label); by=a -> (sort(collect(a.face)), a.side === nothing ? -1 : Int(a.side))))

"""
Color for a classification-atom label: hue keyed by the winning sub-simplex
(`face`, via the shared `ids` registry so the same sub-simplex always gets
the same hue across a plot), shaded lighter/darker by `side` so the two
sides of a segment's interior read as distinct shades of one hue, while a
vertex win (`side === nothing`, no discrete side) gets the plain base color.
"""
function label_color(ids::Dict, label::Set)
    a = canonical_atom(label)
    face_key = sort(collect(a.face))
    k = get!(ids, face_key, length(ids) + 1)
    base = hue_color(k)
    a.side === nothing && return base
    return shade(base, a.side ? 0.22 : -0.22)
end

function padded_bbox(coords; pad=0.2)
    xs = [c[1] for c in coords]
    ys = [c[2] for c in coords]
    xmin, xmax = extrema(xs)
    ymin, ymax = extrema(ys)
    w, h = xmax - xmin, ymax - ymin
    scale = max(w, h, 1.0)   # guard against degenerate (collinear / single-point) inputs
    padw, padh = pad * scale, pad * scale
    return (xmin - padw, xmax + padw, ymin - padh, ymax + padh)
end

"""
Grid-sample `complex` and classify every sample point by classification
atom (see `atom_label_at`) -- which sub-simplex and, for a codimension-1
feature, which side, not just which input simplex wins. Returns the sample
ranges and a matrix of labels (`Set` of atoms).
"""
function label_grid(complex::InputComplex; nx=500, ny=500, pad=0.2)
    xmin, xmax, ymin, ymax = padded_bbox(complex.coords; pad=pad)
    xs = range(xmin, xmax, length=nx)
    ys = range(ymin, ymax, length=ny)
    feats = all_features(complex)
    labels = Matrix{Set}(undef, nx, ny)
    for (i, x) in enumerate(xs), (j, y) in enumerate(ys)
        labels[i, j] = atom_label_at(feats, SVector(x, y))
    end
    return xs, ys, labels
end

"""
Render the pointwise-classified raster of `complex` via GLMakie: each
distinct sub-simplex gets its own hue, the two sides of a segment's
interior render as distinct shades of that hue, and the input
segments/points are drawn on top for reference. This is the quick,
approximate first visualization (M0) -- it does not compute or depend on
the exact output complex.
"""
function plot_raster(complex::InputComplex; nx=600, ny=600)
    xs, ys, labels = label_grid(complex; nx=nx, ny=ny)

    ids = Dict{Vector{VertexIdx},Int}()
    img = Matrix{RGBf}(undef, nx, ny)
    for idx in eachindex(labels)
        img[idx] = label_color(ids, labels[idx])
    end

    fig = Figure(size=(900, 900))
    ax = Axis(fig[1, 1], aspect=DataAspect())
    image!(ax, (first(xs), last(xs)), (first(ys), last(ys)), img)

    for s in complex.simplices
        if s isa SegmentSimplex
            a, b = complex.coords[s.a], complex.coords[s.b]
            lines!(ax, [a[1], b[1]], [a[2], b[2]]; color=:black, linewidth=3)
        elseif s isa PointSimplex
            p = complex.coords[s.v]
            scatter!(ax, [p[1]], [p[2]]; color=:black, markersize=12)
        end
    end

    return fig
end

"""
Sample points along half-edge `h`'s own curve between its two endpoints
(not the straight chord between them) -- needed so parabola-typed edges
render as actual arcs.
"""
function tessellate_halfedge(dcel::DCEL, h::Int; n=24)
    he = dcel.halfedges[h]
    o = vcoord(dcel, he.origin)
    d = vcoord(dcel, dcel.halfedges[he.twin].origin)
    to, td = param_of(he.curve, o), param_of(he.curve, d)
    return [point_at(he.curve, t) for t in range(to, td, length=n)]
end

function face_outline(dcel::DCEL, face_id::Int; n=24)
    pts = SVector{2,Float64}[]
    for h in face_halfedges(dcel, face_id)
        seg = tessellate_halfedge(dcel, h; n=n)
        append!(pts, seg[1:end-1])
    end
    return pts
end

"""
Render every live (non-outer, non-merged-away) face of `dcel` as a filled,
curve-tessellated polygon with a distinct color -- for visual sanity-
checking of the arrangement itself, independent of any label semantics.
"""
function plot_dcel(dcel::DCEL; title="")
    fig = Figure(size=(900, 900))
    ax = Axis(fig[1, 1], aspect=DataAspect(), title=title)
    for face_id in live_faces(dcel)
        pts = face_outline(dcel, face_id)
        xs, ys = [p[1] for p in pts], [p[2] for p in pts]
        poly!(ax, xs, ys; color=(hue_color(face_id), 0.65), strokecolor=:black, strokewidth=1.5)
    end
    return fig
end

"""
Render the real, exact output complex: live faces colored by classification
atom -- distinct sub-simplices get distinct hues, the two sides of a
segment's interior render as distinct shades of one hue, and disconnected
faces sharing a label match -- with the original input mesh overlaid in
black.
"""
function plot_output_complex(dcel::DCEL, complex::InputComplex; title="")
    fig = Figure(size=(900, 900))
    ax = Axis(fig[1, 1], aspect=DataAspect(), title=title)

    ids = Dict{Vector{VertexIdx},Int}()
    for face_id in live_faces(dcel)
        pts = face_outline(dcel, face_id)
        xs, ys = [p[1] for p in pts], [p[2] for p in pts]
        poly!(ax, xs, ys; color=(label_color(ids, dcel.faces[face_id].label), 0.85), strokecolor=:black, strokewidth=1.5)
    end

    for s in complex.simplices
        if s isa SegmentSimplex
            a, b = complex.coords[s.a], complex.coords[s.b]
            lines!(ax, [a[1], b[1]], [a[2], b[2]]; color=:black, linewidth=3)
        elseif s isa PointSimplex
            p = complex.coords[s.v]
            scatter!(ax, [p[1]], [p[2]]; color=:black, markersize=12)
        end
    end

    return fig
end
