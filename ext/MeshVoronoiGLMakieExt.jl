module MeshVoronoiGLMakieExt

using MeshVoronoi
using MeshVoronoi: Pt, CellComplex, polygon_vertices_2d
using GLMakie

function MeshVoronoi.plot_cells_2d(cx::CellComplex{2}, cell_ids_colors::Vector{<:Tuple}, input_points::Vector{Pt{2,Float64}}; title="")
    fig = Figure(size=(700, 700))
    ax = Axis(fig[1, 1], aspect=DataAspect(), title=title)
    for (id, color) in cell_ids_colors
        verts = polygon_vertices_2d(cx, id)
        xs, ys = [p[1] for p in verts], [p[2] for p in verts]
        poly!(ax, xs, ys; color=color, strokecolor=:black, strokewidth=1.5)
    end
    for p in input_points
        scatter!(ax, [p[1]], [p[2]]; color=:black, markersize=12)
    end
    return fig
end

end # module MeshVoronoiGLMakieExt
