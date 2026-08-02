const VertexIdx = Int

abstract type InputSimplex end

struct PointSimplex <: InputSimplex
    v::VertexIdx
end

struct SegmentSimplex <: InputSimplex
    a::VertexIdx
    b::VertexIdx
end

struct InputComplex
    coords::Vector{SVector{2,Float64}}
    simplices::Vector{InputSimplex}
end
