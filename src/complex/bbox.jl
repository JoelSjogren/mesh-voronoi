"""
A face of an `N`-cube: `spec[i]` is `:lo`/`:hi` if coordinate `i` is pinned
to that bound, or `:free` if it ranges freely over that face. Dimension is
the count of `:free` entries.
"""
const FaceSpec{N} = NTuple{N,Symbol}

face_dim(spec::FaceSpec) = count(==(:free), spec)

"""All `N`-cube face specs of dimension `dim`, via direct bitmask enumeration."""
function face_specs(::Val{N}, dim::Int) where {N}
    out = FaceSpec{N}[]
    for free_mask in 0:(2^N-1)
        count_ones(free_mask) == dim || continue
        nfixed = N - dim
        for bit_mask in 0:(2^nfixed-1)
            spec = Vector{Symbol}(undef, N)
            bi = 0
            for i in 1:N
                if (free_mask >> (i - 1)) & 1 == 1
                    spec[i] = :free
                else
                    spec[i] = ((bit_mask >> bi) & 1 == 1) ? :hi : :lo
                    bi += 1
                end
            end
            push!(out, Tuple(spec))
        end
    end
    return out
end

"""
The `2*dim` immediate (dimension `dim-1`) subfaces of a dimension-`dim`
face spec: for each free coordinate, pin it to `:lo` or `:hi` in turn.
"""
function immediate_subface_specs(spec::FaceSpec{N}) where {N}
    out = FaceSpec{N}[]
    for i in 1:N
        spec[i] == :free || continue
        for bound in (:lo, :hi)
            sub = collect(spec)
            sub[i] = bound
            push!(out, Tuple(sub))
        end
    end
    return out
end

"""
Build the full face lattice of the axis-aligned box `[lo,hi]` as a fresh
`CellComplex{N}` (every 0..N-dimensional face, correctly linked), all
initially unlabeled -- labels are assigned once real input features exist
to compete for them (`clip_by_hyperplane!`). Returns the complex and the
id of its single top-dimensional cell.
"""
function init_bbox_complex(::Val{N}, lo::Pt{N,Float64}, hi::Pt{N,Float64}) where {N}
    cx = CellComplex{N}()
    node_of_spec = Dict{FaceSpec{N},Int}()
    for dim in 0:N
        for spec in face_specs(Val(N), dim)
            if dim == 0
                point = SVector{N,Float64}(ntuple(i -> spec[i] == :lo ? lo[i] : hi[i], N))
                id = add_cell!(cx, 0, Label(), Int[], point)
            else
                subids = [node_of_spec[s] for s in immediate_subface_specs(spec)]
                id = add_cell!(cx, dim, Label(), subids, nothing)
            end
            node_of_spec[spec] = id
        end
    end
    top_id = node_of_spec[ntuple(_ -> :free, N)]
    return cx, top_id
end

"""
Build the face lattice of an arbitrary convex polygon `verts` (2D, cyclic
order, as `offset_polygon` returns) as a fresh `CellComplex{2}` -- one
dim=0 cell per vertex, one dim=1 cell per edge, one dim=2 top cell
referencing every edge, all unlabeled like `init_bbox_complex`. Returns
the complex and the id of its top cell.
"""
function init_hull_offset_complex(verts::Vector{Pt{2,Float64}})
    n = length(verts)
    n >= 3 || error("init_hull_offset_complex: need a genuine polygon (>= 3 vertices)")
    cx = CellComplex{2}()
    vertex_ids = [add_cell!(cx, 0, Label(), Int[], v) for v in verts]
    edge_ids = [add_cell!(cx, 1, Label(), [vertex_ids[i], vertex_ids[mod1(i + 1, n)]], nothing) for i in 1:n]
    top_id = add_cell!(cx, 2, Label(), edge_ids, nothing)
    return cx, top_id
end
