module MeshVoronoi

using LinearAlgebra
using StaticArrays
using Polynomials: Polynomial, roots, degree, coeffs
using GLMakie

include("geometry/curves.jl")
include("geometry/quadratics.jl")
include("dcel/dcel.jl")
include("dcel/construction.jl")
include("complex/input_complex.jl")
include("complex/features.jl")
include("algorithm/dist.jl")
include("algorithm/pipeline.jl")
include("rendering/makie_glue.jl")

export VertexIdx, InputSimplex, PointSimplex, SegmentSimplex, InputComplex
export dist2, brute_force_label
export label_grid, plot_raster, plot_dcel, plot_output_complex, face_outline, tessellate_halfedge
export Line, Parabola, Curve, evaluate, point_at, param_of, intersect_points
export Vtx, HalfEdge, Face, DCEL, init_bbox_dcel, face_halfedges, face_polygon
export insert_curve!, point_in_face, locate_face
export Quadratic, PointQuadratic, LineQuadratic, sqdist, bisector
export ValidityRegion, WholePlane, HalfPlane, Strip, is_valid, Feature, features, all_features
export label_all_faces!, merge_equal_label_faces!, dissolve_degree2_vertices!
export compute_output_complex, incorporate_and_label!, label_at, live_faces, interior_sample_point
export QuadraticGroup, group_active_features, atom_of, simplices_of, atom_label_at

end # module MeshVoronoi
