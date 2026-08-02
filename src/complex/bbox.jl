"""
A face of an `N`-cube: `spec[i]` is `:lo`/`:hi` if coordinate `i` is pinned
to that bound, or `:free` if coordinate `i` ranges freely over that face.
The face's dimension is the number of `:free` entries. This is the
standard combinatorial description of a hypercube's full face lattice
(vertices, edges, ..., facets, the cube itself), used only to bootstrap
the initial bounded region the incremental construction starts from.
"""
const FaceSpec{N} = NTuple{N,Symbol}

face_dim(spec::FaceSpec) = count(==(:free), spec)

"""
All `N`-cube face specs of dimension `dim`: choose which `dim` coordinates
are free (`binomial(N,dim)` ways), and a `:lo`/`:hi` assignment for the
rest (`2^(N-dim)` ways) -- `binomial(N,dim)*2^(N-dim)` total, the standard
hypercube face-count formula. Implemented via direct bitmask enumeration
(no extra dependency) rather than a combinatorics library, which is fine
at the small `N` this project targets.
"""
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
`CellComplex{N}` (every 0-, 1-, ..., `N`-dimensional face stored, with
correct subcell links), all initially unlabeled (`Label()`, empty --
labels are assigned once real input points exist to compete for them; see
`clip_by_hyperplane!`). Returns the complex and the id of its single
top-dimensional (`dim=N`) cell.
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
