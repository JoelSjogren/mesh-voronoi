using Test
using MeshVoronoi
using StaticArrays
using LinearAlgebra

@testset "MeshVoronoi" begin
    include("test_types.jl")
    include("test_interval.jl")
    include("test_predicates.jl")
    include("test_bbox.jl")
    include("test_convex_hull.jl")
    include("test_compactified.jl")
    include("test_clip.jl")
    include("test_two_points.jl")
    include("test_supersede.jl")
    include("test_multi_points.jl")
    include("test_3d_segments.jl")
    include("test_segments.jl")
    include("test_tie_boundary.jl")
    include("test_stress_regressions.jl")
end
