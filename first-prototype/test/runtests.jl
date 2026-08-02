using Test
using MeshVoronoi
using StaticArrays

@testset "MeshVoronoi" begin
    include("test_dist.jl")
    include("test_dcel_basic.jl")
    include("test_two_points.jl")
    include("test_point_segment.jl")
    include("test_house.jl")
end
