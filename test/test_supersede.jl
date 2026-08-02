@testset "supersede!: patches a single parent's subcell list in place" begin
    cx = CellComplex{2}()
    v1 = add_cell!(cx, 0, Label(), Int[], SVector(0.0, 0.0))
    v2 = add_cell!(cx, 0, Label(), Int[], SVector(1.0, 0.0))
    edge = add_cell!(cx, 1, Label(), [v1, v2])

    @test cx.referenced_by[v1] == [edge]
    @test cx.referenced_by[v2] == [edge]

    # split v2's role: pretend it's superseded by two brand-new 0-cells
    newv1 = add_cell!(cx, 0, Label(), Int[], SVector(1.0, 0.5))
    newv2 = add_cell!(cx, 0, Label(), Int[], SVector(1.0, -0.5))
    supersede!(cx, v2, [newv1, newv2])

    @test cx.nodes[edge].subcells == [v1, newv1, newv2]
    @test resolve(cx, v2) == [newv1, newv2]
    @test resolve(cx, v1) == [v1]   # untouched node resolves to itself
    @test !haskey(cx.referenced_by, v2)
    @test edge in cx.referenced_by[newv1]
    @test edge in cx.referenced_by[newv2]
end

@testset "supersede!: patches multiple parents referencing the same shared subcell" begin
    cx = CellComplex{2}()
    v1 = add_cell!(cx, 0, Label(), Int[], SVector(0.0, 0.0))
    v2 = add_cell!(cx, 0, Label(), Int[], SVector(1.0, 0.0))
    shared = add_cell!(cx, 1, Label(), [v1, v2])   # a "boundary" shared by two 2-cells

    v3 = add_cell!(cx, 0, Label(), Int[], SVector(0.0, 1.0))
    v4 = add_cell!(cx, 0, Label(), Int[], SVector(1.0, 1.0))
    cellA = add_cell!(cx, 2, Label([Set([1])]), [shared, v3, v4])

    v5 = add_cell!(cx, 0, Label(), Int[], SVector(0.0, -1.0))
    v6 = add_cell!(cx, 0, Label(), Int[], SVector(1.0, -1.0))
    cellC = add_cell!(cx, 2, Label([Set([3])]), [shared, v5, v6])

    @test Set(cx.referenced_by[shared]) == Set([cellA, cellC])

    # the shared boundary itself gets split (e.g. a new point's bisector
    # crosses it) -- both cellA and cellC should see the update
    new1 = add_cell!(cx, 1, Label(), [v1])
    new2 = add_cell!(cx, 1, Label(), [v2])
    supersede!(cx, shared, [new1, new2])

    @test cx.nodes[cellA].subcells == [new1, new2, v3, v4]
    @test cx.nodes[cellC].subcells == [new1, new2, v5, v6]
    @test Set(cx.referenced_by[new1]) == Set([cellA, cellC])
    @test Set(cx.referenced_by[new2]) == Set([cellA, cellC])
end

@testset "resolve: chained supersession resolves transitively" begin
    cx = CellComplex{2}()
    v = add_cell!(cx, 0, Label(), Int[], SVector(0.0, 0.0))
    a = add_cell!(cx, 0, Label(), Int[], SVector(1.0, 0.0))
    b = add_cell!(cx, 0, Label(), Int[], SVector(2.0, 0.0))
    supersede!(cx, v, [a])       # first supersession
    c = add_cell!(cx, 0, Label(), Int[], SVector(3.0, 0.0))
    d = add_cell!(cx, 0, Label(), Int[], SVector(4.0, 0.0))
    supersede!(cx, a, [c, d])    # a itself later superseded too
    @test resolve(cx, v) == [c, d]
    @test resolve(cx, a) == [c, d]
    @test resolve(cx, c) == [c]
end
