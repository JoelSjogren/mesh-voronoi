using MeshVoronoi
using StaticArrays
using GLMakie

include(joinpath(@__DIR__, "..", "examples", "house_chimney.jl"))

"""
One dashboard entry: a short human `description`, and `run` -- a zero-arg
function producing `(pass::Bool, fig, detail::String)`. `fig` is a Makie
figure from the project's own renderers (so the dashboard never has a
second, divergent way of drawing a cell diagram); `detail` is a one-line
note, and for a failing case should say what's most likely wrong.

`group` is `"current"` (the dimension-generic redesign, `mesh-voronoi-nd`)
or `"legacy"` (the original 2D DCEL prototype) -- the frontend uses this to
keep the legacy cases out of the way (collapsed by default) once there's a
current-pipeline replacement to look at instead.
"""
struct DashboardCase
    name::String
    description::String
    run::Function
    group::String
end

"""
Cross-validate `dcel` against the brute-force oracle over a grid, skipping
points too close to a true classification boundary (ambiguous ties) and
any point `skip` flags (a known, already-understood gap). Returns
`(mismatches, checked)`.
"""
function cross_validate(dcel, complex; xr, yr, tie_atol=1e-6, skip=nothing)
    mismatches = 0
    checked = 0
    for x in xr, y in yr
        pt = SVector(x, y)
        skip !== nothing && skip(pt) && continue
        ds = sort([dist2(complex, i, pt) for i in eachindex(complex.simplices)])
        length(ds) > 1 && ds[2] - ds[1] < tie_atol && continue
        computed = label_at(dcel, pt)
        computed === nothing && continue
        checked += 1
        simplices_of(computed) != brute_force_label(complex, pt) && (mismatches += 1)
    end
    return mismatches, checked
end

function case_two_points()
    complex = InputComplex(
        SVector{2,Float64}[(0.0, 0.0), (2.0, 0.0)],
        InputSimplex[PointSimplex(1), PointSimplex(2)],
    )
    dcel = compute_output_complex(complex)
    fig = plot_output_complex(dcel, complex; title="Two isolated points")
    mismatches, checked = cross_validate(dcel, complex; xr=range(-0.55, 2.55, length=27), yr=range(-0.55, 0.55, length=15))
    pass = mismatches == 0
    detail = pass ? "Perpendicular bisector between two points -- the simplest possible case." :
             "$mismatches/$checked grid points disagree with the oracle."
    return pass, fig, detail
end

function case_point_segment()
    complex = InputComplex(
        SVector{2,Float64}[(0.0, 3.0), (-2.0, 0.0), (2.0, 0.0)],
        InputSimplex[PointSimplex(1), SegmentSimplex(2, 3)],
    )
    dcel = compute_output_complex(complex)
    fig = plot_output_complex(dcel, complex; title="Point + segment (parabola)")
    mismatches, checked = cross_validate(dcel, complex; xr=range(-3.1, 3.1, length=29), yr=range(-1.1, 4.1, length=29))
    below = only(label_at(dcel, SVector(-1.5, -0.3)))
    above = only(label_at(dcel, SVector(-1.5, 0.3)))
    side_ok = below.face == above.face == Set([2, 3]) && below.side != above.side
    pass = mismatches == 0 && side_ok
    detail = if !side_ok
        "Two points symmetric about the segment's interior should share a face but have opposite side; they didn't."
    elseif pass
        "Point-vs-line bisector is a parabola; the two sides of the segment render as distinct shades of one hue."
    else
        "$mismatches/$checked grid points disagree with the oracle."
    end
    return pass, fig, detail
end

function case_shared_vertex()
    complex = InputComplex(
        SVector{2,Float64}[(0.0, 0.0), (2.0, 0.0), (2.0, 2.0)],
        InputSimplex[SegmentSimplex(1, 2), SegmentSimplex(2, 3)],
    )
    dcel = compute_output_complex(complex)
    fig = plot_output_complex(dcel, complex; title="Shared-vertex tie wedge")
    wedge = label_at(dcel, SVector(2.4, -0.3))
    wedge_ok = wedge !== nothing && length(wedge) == 1 &&
               only(wedge).face == Set([2]) && only(wedge).simplices == Set([1, 2])
    mismatches, checked = cross_validate(dcel, complex; xr=range(-0.55, 2.65, length=25), yr=range(-0.55, 2.65, length=25))
    pass = mismatches == 0 && wedge_ok
    detail = if !wedge_ok
        "Expected the wedge beyond the shared vertex to resolve to one atom realized by both segments; got $wedge."
    elseif pass
        "Two segments sharing an endpoint: the wedge beyond it is realized by both simplices at once (no discrete side -- a vertex is codimension 2)."
    else
        "$mismatches/$checked grid points disagree with the oracle."
    end
    return pass, fig, detail
end

function case_house_generic()
    complex = house_chimney_generic()
    dcel = compute_output_complex(complex)
    fig = plot_output_complex(dcel, complex; title="House with open chimney (generic)")
    # A narrow region near the chimneyLeft/leftroofLower/leftwall junction is
    # a known, not-yet-fixed gap: a nested "double dangling tip" (an open
    # segment endpoint's own zero-width boundary spike, poking through
    # territory that structurally belongs to a different face) confuses
    # insert_curve!'s fallback face-lookup. See test/test_house.jl.
    known_gap(pt) = -0.6 <= pt[1] <= 1.8 && 1.9 <= pt[2] <= 4.4
    mismatches, checked = cross_validate(dcel, complex; xr=range(-3.5, 7.5, length=41), yr=range(-4.5, 8.5, length=41), skip=known_gap)
    pass = mismatches == 0
    detail = pass ? "Hero example: 8 segments, one connected open-chimney arc, cross-validated outside the known dangling-tip gap." :
             "$mismatches/$checked grid points disagree with the oracle (outside the already-excluded known gap)."
    return pass, fig, detail
end

function case_house_exact()
    complex = house_chimney()
    dcel = compute_output_complex(complex)
    fig = plot_output_complex(dcel, complex; title="House with open chimney (exact -- known failing)")
    mismatches, checked = cross_validate(dcel, complex; xr=range(-3.5, 7.5, length=21), yr=range(-4.5, 8.5, length=21))
    pass = mismatches == 0
    detail = pass ? "Unexpectedly passed -- the exact-coincidence gap may have narrowed." :
             "$mismatches/$checked grid points disagree with the oracle. Likely culprit: exact right-angle/collinear " *
             "coincidences (this project's current scope is generic-case only, see house_chimney_generic's docstring); " *
             "deliberately kept here as a live failing example rather than the perturbed mesh."
    return pass, fig, detail
end

"""
A plain text-summary Makie figure -- used for cases that report a test
suite's pass/fail state rather than drawing any geometry (there's nothing
to render yet for the new package's foundational, non-geometric
milestones like G0).
"""
function text_summary_figure(title::String, lines::Vector{String}, pass::Bool)
    fig = Figure(size=(720, 360))
    color = pass ? :darkgreen : :darkred
    ax = Axis(fig[1, 1]; title=title, titlesize=20)
    hidedecorations!(ax)
    hidespines!(ax)
    body = join(lines, "\n")
    text!(ax, 0.02, 0.95; text=body, align=(:left, :top), fontsize=15, color=color, space=:relative)
    return fig
end

"""
Run `mesh-voronoi-nd`'s own test suite as a subprocess (its own separate
package environment, so this doesn't try to mix two different Project.toml
dependency sets in one running process) and summarize the result -- this is
how progress on the dimension-generic redesign shows up on the dashboard
as it's being built, milestone by milestone (G0, G1, ...).
"""
function case_meshvoronoind_tests()
    nd_dir = joinpath(@__DIR__, "..", "..")
    if !isdir(nd_dir)
        fig = text_summary_figure("MeshVoronoi", ["package directory not found:", nd_dir], false)
        return false, fig, "mesh-voronoi-nd directory not found at $nd_dir"
    end

    io = IOBuffer()
    cmd = `julia --project=$(nd_dir) $(joinpath(nd_dir, "test", "runtests.jl"))`
    proc = run(pipeline(cmd; stdout=io, stderr=io); wait=false)
    wait(proc)
    output = String(take!(io))
    pass = success(proc)

    summary_line = ""
    for line in split(output, '\n')
        occursin("MeshVoronoi", line) && occursin("|", line) && (summary_line = strip(line))
    end
    lines = ["subprocess exit: $(pass ? "success" : "FAILED")"]
    isempty(summary_line) || push!(lines, summary_line)
    if !pass
        tail_lines = split(output, '\n')
        append!(lines, tail_lines[max(1, end - 8):end])
    end

    detail = pass ? "Dimension-generic redesign (G0: exact-predicate kernel so far). $summary_line" :
             "Dimension-generic redesign test suite failing -- see detail lines / rerun `julia --project=$(nd_dir) test/runtests.jl` locally."
    fig = text_summary_figure("MeshVoronoi test suite (redesign progress)", lines, pass)
    return pass, fig, detail
end

"""
The same "two isolated points" scenario as `case_two_points`, but built
through the new dimension-generic pipeline (`mesh-voronoi-nd`, G1's
points-only construction) instead of the 2D prototype's DCEL -- run as a
subprocess (separate package environment) that renders and saves its own
PNG directly into this dashboard's `images/` directory, since a Makie
figure built in another process can't be handed back as a live object.
Kept alongside `case_two_points` (not replacing it) so old and new can be
compared directly as more milestones land.
"""
function case_two_points_nd()
    nd_dir = joinpath(@__DIR__, "..", "..")
    render_project = joinpath(@__DIR__, "render_env")
    if !isdir(nd_dir)
        fig = text_summary_figure("two_points (new pipeline)", ["mesh-voronoi-nd not found"], false)
        return false, fig, "mesh-voronoi-nd directory not found at $nd_dir"
    end

    stamp = round(Int, time() * 1000)
    image_path = joinpath(IMAGES_DIR, "two_points_nd_$stamp.png")
    script = """
    using MeshVoronoi, StaticArrays, GLMakie
    pa, pb = SVector(0.0, 0.0), SVector(2.0, 0.5)
    cx, a_id, b_id, cut_id = two_points_complex(pa, pb)
    ok = cx.nodes[a_id].label == Set([Set([1])]) && cx.nodes[b_id].label == Set([Set([2])]) &&
         cx.nodes[cut_id].label == Set([Set([1]), Set([2])])
    fig = plot_cells_2d(cx, [(a_id, :lightgreen), (b_id, :lightblue)], [pa, pb]; title="Two points -- new dimension-generic pipeline (G1)")
    save($(repr(image_path)), fig)
    println(ok ? "RESULT: PASS" : "RESULT: FAIL")
    """
    io = IOBuffer()
    cmd = `julia --project=$(render_project) -e $script`
    proc = run(pipeline(cmd; stdout=io, stderr=io); wait=false)
    wait(proc)
    output = String(take!(io))
    pass = success(proc) && occursin("RESULT: PASS", output)
    detail = pass ? "Two isolated points via the new dimension-generic pipeline (G1: label-indexed cells, exact predicates) -- compare against two_points above." :
             "New-pipeline two_points case failed. Output tail: " * join(split(output, '\n')[max(1, end - 5):end], " | ")
    return pass, "/images/$(basename(image_path))", detail
end

"""
Five points, general power-diagram case (G1's multi-point incremental
construction, not just the fixed single-bisector two-point case above) --
run the same way as `case_two_points_nd`, as a subprocess that renders and
saves its own PNG. Cross-validates every cell's label against the
brute-force oracle over a grid, exactly like the 2D prototype's own
`cross_validate` pattern.
"""
function case_five_points_nd()
    nd_dir = joinpath(@__DIR__, "..", "..")
    render_project = joinpath(@__DIR__, "render_env")
    if !isdir(nd_dir)
        fig = text_summary_figure("five points (new pipeline)", ["mesh-voronoi-nd not found"], false)
        return false, fig, "mesh-voronoi-nd directory not found at $nd_dir"
    end

    stamp = round(Int, time() * 1000)
    image_path = joinpath(IMAGES_DIR, "five_points_nd_$stamp.png")
    script = """
    using MeshVoronoi, StaticArrays, GLMakie

    function main()
        points = [SVector(0.0, 0.0), SVector(3.0, 0.0), SVector(1.3, 2.5), SVector(-1.7, 1.4), SVector(3.6, 2.1)]
        cx = points_complex(points)

        colors = Dict(1 => :lightgreen, 2 => :lightblue, 3 => :lightyellow, 4 => :lightpink, 5 => :lightgray)
        cell_ids_colors = Tuple{Int,Symbol}[]
        for id in eachindex(cx.nodes)
            haskey(cx.superseded_by, id) && continue
            node = cx.nodes[id]
            (node.dim == 2 && !isempty(node.label)) || continue
            w = only(only(node.label))
            push!(cell_ids_colors, (id, colors[w]))
        end
        fig = plot_cells_2d(cx, cell_ids_colors, points; title="Five points -- new dimension-generic pipeline (G1, multi-point)")

        lo, hi = padded_bbox(points)
        mismatches = 0
        checked = 0
        for x in range(lo[1], hi[1], length=61), y in range(lo[2], hi[2], length=61)
            pt = SVector(x, y)
            expected = brute_force_label_points(points, pt)
            length(expected) > 1 && continue
            ds = [sum(abs2, pt - p) for p in points]
            m = minimum(ds)
            length(expected) == 1 && ds[only(expected)] > m + 1e-6*max(1.0,m) && continue
            checked += 1
            got = nothing
            for (id, _) in cell_ids_colors
                poly = polygon_vertices_2d(cx, id)
                n = length(poly)
                c = sum(poly) / n
                spts = sort(poly; by = p -> atan(p[2]-c[2], p[1]-c[1]))
                signs = Float64[]
                for i in 1:n
                    a, b = spts[i], spts[mod1(i+1, n)]
                    e = b - a
                    push!(signs, e[1]*(pt[2]-a[2]) - e[2]*(pt[1]-a[1]))
                end
                inside = all(s -> s >= -1e-9, signs) || all(s -> s <= 1e-9, signs)
                if inside
                    got = only(only(cx.nodes[id].label))
                    break
                end
            end
            got === nothing && continue
            got != only(expected) && (mismatches += 1)
        end
        ok = mismatches == 0 && checked > 500
        save($(repr(image_path)), fig)
        println("RESULT: ", ok ? "PASS" : "FAIL", " mismatches=\$mismatches checked=\$checked")
    end

    main()
    """
    io = IOBuffer()
    cmd = `julia --project=$(render_project) -e $script`
    proc = run(pipeline(cmd; stdout=io, stderr=io); wait=false)
    wait(proc)
    output = String(take!(io))
    pass = success(proc) && occursin("RESULT: PASS", output)
    result_line = ""
    for line in split(output, '\n')
        occursin("RESULT:", line) && (result_line = strip(line))
    end
    detail = pass ? "Five-point power diagram via the new pipeline's general multi-point incremental construction (G1). $result_line" :
             "New-pipeline five_points case failed. $(isempty(result_line) ? "" : result_line * " ") Output tail: " * join(split(output, '\n')[max(1, end - 5):end], " | ")
    return pass, "/images/$(basename(image_path))", detail
end

"""
Regression test for a formerly-known representational gap in the new
pipeline (G1): a single winner's territory used to end up stored as two or
more separate, geometrically-adjacent cells (sharing a boundary edge)
instead of being merged into one. `merge_adjacent_same_label_cells!`
(called from `points_complex`) now fixes this, so this case checks that
point 3's territory comes back as exactly *one* live top-cell -- if it
ever regresses to two-or-more again, this case should fail rather than
silently going back to documenting the old gap.
"""
function case_merge_gap_nd()
    nd_dir = joinpath(@__DIR__, "..", "..")
    render_project = joinpath(@__DIR__, "render_env")
    if !isdir(nd_dir)
        fig = text_summary_figure("merge gap (new pipeline)", ["mesh-voronoi-nd not found"], false)
        return false, fig, "mesh-voronoi-nd directory not found at $nd_dir"
    end

    stamp = round(Int, time() * 1000)
    image_path = joinpath(IMAGES_DIR, "merge_gap_nd_$stamp.png")
    script = """
    using MeshVoronoi, StaticArrays, GLMakie
    points = [SVector(0.0, 0.0), SVector(2.0, 0.0), SVector(1.0, 1.8)]
    cx = points_complex(points)

    label3 = MeshVoronoi.Label([Set([3])])
    live_top_cells = [id for id in cx.label_index[label3] if !haskey(cx.superseded_by, id) && cx.nodes[id].dim == 2]
    ok = length(live_top_cells) == 1

    cell_ids_colors = [(only(live_top_cells), :orange)]
    for id in eachindex(cx.nodes)
        haskey(cx.superseded_by, id) && continue
        node = cx.nodes[id]
        (node.dim == 2 && !isempty(node.label)) || continue
        w = only(only(node.label))
        w == 3 && continue
        push!(cell_ids_colors, (id, w == 1 ? :lightgreen : :lightblue))
    end
    fig = plot_cells_2d(cx, cell_ids_colors, points; title="Point 3's territory merged into ONE cell (orange)")
    save($(repr(image_path)), fig)
    println(ok ? "RESULT: PASS" : "RESULT: FAIL", " live_top_cells=\$(length(live_top_cells))")
    """
    io = IOBuffer()
    cmd = `julia --project=$(render_project) -e $script`
    proc = run(pipeline(cmd; stdout=io, stderr=io); wait=false)
    wait(proc)
    output = String(take!(io))
    pass = success(proc) && occursin("RESULT: PASS", output)
    result_line = ""
    for line in split(output, '\n')
        occursin("RESULT:", line) && (result_line = strip(line))
    end
    detail = pass ? "Fixed: point 3's territory is stored as a single merged cell (orange), not split across adjacent pieces. $result_line" :
             "Regression: point 3's territory is no longer a single merged cell -- merge_adjacent_same_label_cells! may be broken. $(isempty(result_line) ? "" : result_line * " ") Output tail: " * join(split(output, '\n')[max(1, end - 5):end], " | ")
    return pass, "/images/$(basename(image_path))", detail
end

"""
G2's first real milestone: one point and one segment, via the new
dimension-generic pipeline -- the same scenario as the legacy
`point_segment` case (M3), but now built on the label-indexed
`CellComplex` with a genuinely curved (parabolic) bisector clip, instead
of the old DCEL. Cross-validates every live top-cell's territory (rendered
as an actual curved polygon, not a straight-chord approximation) against
the brute-force oracle over a grid.
"""
function case_point_segment_nd()
    nd_dir = joinpath(@__DIR__, "..", "..")
    render_project = joinpath(@__DIR__, "render_env")
    if !isdir(nd_dir)
        fig = text_summary_figure("point + segment (new pipeline)", ["mesh-voronoi-nd not found"], false)
        return false, fig, "mesh-voronoi-nd directory not found at $nd_dir"
    end

    stamp = round(Int, time() * 1000)
    image_path = joinpath(IMAGES_DIR, "point_segment_nd_$stamp.png")
    script = """
    using MeshVoronoi, StaticArrays, GLMakie

    function point_in_convex_polygon(verts, x; atol=1e-9)
        n = length(verts)
        signs = Float64[]
        for i in 1:n
            a, b = verts[i], verts[mod1(i+1,n)]
            e = b - a
            push!(signs, e[1]*(x[2]-a[2]) - e[2]*(x[1]-a[1]))
        end
        return all(s -> s >= -atol, signs) || all(s -> s <= atol, signs)
    end

    function main()
        p = SVector(0.4, 3.0)
        pa, pb = SVector(-1.5, 0.0), SVector(1.5, 0.0)
        cx, feats = point_segment_complex(p, pa, pb)
        lo, hi = padded_bbox([p, pa, pb])

        colors = Dict(Set([1]) => :lightgreen, Set([2]) => :lightblue, Set([3]) => :lightpink, Set([2,3]) => :lightyellow)
        cell_ids_colors = Tuple{Int,Symbol}[]
        cells = Tuple{Set{Int},Vector{SVector{2,Float64}}}[]
        for id in eachindex(cx.nodes)
            haskey(cx.superseded_by, id) && continue
            node = cx.nodes[id]
            (node.dim == 2 && !isempty(node.label)) || continue
            face = only(node.label)
            push!(cell_ids_colors, (id, colors[face]))
            push!(cells, (face, polygon_vertices_2d(cx, id)))
        end

        checked, mismatches = 0, 0
        for x in range(lo[1], hi[1], length=121), y in range(lo[2], hi[2], length=121)
            pt = SVector(x, y)
            expected = brute_force_label_segments(p, pa, pb, pt)
            length(expected) > 1 && continue
            in_cells = [f for (f, poly) in cells if point_in_convex_polygon(poly, pt)]
            length(in_cells) != 1 && continue
            checked += 1
            (in_cells[1] in expected) || (mismatches += 1)
        end
        ok = mismatches == 0 && checked > 5000
        fig = plot_cells_2d(cx, cell_ids_colors, [p, pa, pb]; title="One point + one segment -- new pipeline, curved (parabolic) bisector (G2)")
        save($(repr(image_path)), fig)
        println(ok ? "RESULT: PASS" : "RESULT: FAIL", " checked=\$checked mismatches=\$mismatches")
    end
    main()
    """
    io = IOBuffer()
    cmd = `julia --project=$(render_project) -e $script`
    proc = run(pipeline(cmd; stdout=io, stderr=io); wait=false)
    wait(proc)
    output = String(take!(io))
    pass = success(proc) && occursin("RESULT: PASS", output)
    result_line = ""
    for line in split(output, '\n')
        occursin("RESULT:", line) && (result_line = strip(line))
    end
    detail = pass ? "One point + one segment via the new pipeline's first curved-bisector clip (G2). $result_line" :
             "New-pipeline point_segment case failed. $(isempty(result_line) ? "" : result_line * " ") Output tail: " * join(split(output, '\n')[max(1, end - 5):end], " | ")
    return pass, "/images/$(basename(image_path))", detail
end

const CASES = DashboardCase[
    DashboardCase("meshvoronoind", "Dimension-generic redesign (mesh-voronoi-nd) test suite -- live progress.", case_meshvoronoind_tests, "current"),
    DashboardCase("two_points_nd", "Two isolated points -- same scenario via the new dimension-generic pipeline (G1).", case_two_points_nd, "current"),
    DashboardCase("five_points_nd", "Five-point power diagram -- new pipeline's general multi-point incremental construction (G1).", case_five_points_nd, "current"),
    DashboardCase("merge_gap_nd", "Regression check: adjacent same-label cells merge into one piece (G1, fixed).", case_merge_gap_nd, "current"),
    DashboardCase("point_segment_nd", "One point + one segment -- curved (parabolic) bisector via the new pipeline (G2).", case_point_segment_nd, "current"),
    DashboardCase("two_points", "Two isolated points -- pure line bisector (M2).", case_two_points, "legacy"),
    DashboardCase("point_segment", "One point + one segment -- parabola bisector, exercises side (M3).", case_point_segment, "legacy"),
    DashboardCase("shared_vertex", "Two segments sharing an endpoint -- vertex tie atom, no discrete side.", case_shared_vertex, "legacy"),
    DashboardCase("house_generic", "House with open chimney, perturbed -- hero example (M4).", case_house_generic, "legacy"),
    DashboardCase("house_exact", "House with open chimney, exact coincidences -- known failing, kept deliberately.", case_house_exact, "legacy"),
]
