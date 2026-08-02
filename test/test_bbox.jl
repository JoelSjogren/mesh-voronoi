@testset "init_bbox_complex face counts match hypercube formula" begin
    for N in (2, 3, 4)
        lo = SVector{N,Float64}(zeros(N))
        hi = SVector{N,Float64}(ones(N))
        cx, top = init_bbox_complex(Val(N), lo, hi)
        for dim in 0:N
            expected = binomial(N, dim) * 2^(N - dim)
            actual = count(n -> n.dim == dim, cx.nodes)
            @test actual == expected
        end
        @test cx.nodes[top].dim == N
    end
end

@testset "init_bbox_complex vertex coordinates are the true corners" begin
    for N in (2, 3)
        lo = SVector{N,Float64}(zeros(N))
        hi = SVector{N,Float64}(2.0 .* ones(N))
        cx, top = init_bbox_complex(Val(N), lo, hi)
        verts = [n.point for n in cx.nodes if n.dim == 0]
        @test length(verts) == 2^N
        @test all(p -> all(c -> c == 0.0 || c == 2.0, p), verts)
        @test length(unique(verts)) == 2^N   # all distinct
    end
end

@testset "init_bbox_complex subcell structure is consistent" begin
    for N in (2, 3)
        lo = SVector{N,Float64}(zeros(N))
        hi = SVector{N,Float64}(ones(N))
        cx, top = init_bbox_complex(Val(N), lo, hi)
        for node in cx.nodes
            if node.dim == 0
                @test isempty(node.subcells)
                @test node.point !== nothing
            else
                @test length(node.subcells) == 2 * node.dim
                @test all(s -> cx.nodes[s].dim == node.dim - 1, node.subcells)
                @test node.point === nothing
            end
        end
        # every 0-cell descendant of the top cell is one of the 2^N corners
        @test length(descendant_points(cx, top)) == 2^N
    end
end
