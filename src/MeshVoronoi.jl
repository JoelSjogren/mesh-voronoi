module MeshVoronoi

using LinearAlgebra
using StaticArrays

include("geometry/types.jl")
include("geometry/interval.jl")
include("geometry/predicates.jl")
include("geometry/validity.jl")
include("complex/cell.jl")
include("complex/bbox.jl")
include("algorithm/clip.jl")
include("algorithm/two_points.jl")
include("algorithm/multi_points.jl")
include("algorithm/features.jl")
include("algorithm/segments.jl")
include("rendering/plot2d.jl")

export Pt, AffineQuadratic, hyperplane_quadratic, sqdist
export Quadric, evaluate, to_quadric, bisector
export Interval, straddles_zero, interval_contains
export Exact, to_interval, to_exact, exact_sign, symbolic_tiebreak
export VertexIdx, Label, CellNode, CellComplex, add_cell!, supercells, descendant_points
export set_label!, supersede!, resolve
export face_dim, face_specs, immediate_subface_specs, init_bbox_complex
export clip_by_hyperplane!, quadric_crossing_point, edge_crossings, is_curved
export polygon_vertices_2d, point_in_polygon_2d, in_bbox, find_containing_cell, find_hover_target, edge_distance, edge_polyline, plot_cells_2d
export BVH, build_bvh, find_containing_cell_bvh
export padded_bbox, two_points_complex, brute_force_label_two_points
export insert_point!, points_complex, brute_force_label_points
export weld_near_duplicate_vertices!, weld_duplicate_edges!, fix_boundary_labels!, recompute_point_label, recompute_feature_label
export HalfSpace, is_valid
export GFeature, segment_features, point_feature
export insert_segment!, point_segment_complex, brute_force_label_segments
export insert_own_lines!, insert_features!, insert_entry!, entry_feats, multi_complex, brute_force_label_multi, top_cell_ids, interior_sample
export merge_adjacent_same_label_cells!, cell_bbox, boxes_touch_or_overlap, assert_label_bbox_invariant

end # module MeshVoronoi
