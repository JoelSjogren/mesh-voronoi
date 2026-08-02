@testset "DCEL + insert_curve! (M1)" begin
    @testset "init_bbox_dcel" begin
        dcel = init_bbox_dcel(-5.0, 5.0, -5.0, 5.0)
        @test length(dcel.faces) == 2
        poly = face_polygon(dcel, 1)
        @test length(poly) == 4
        @test point_in_face(dcel, 1, SVector(0.0, 0.0))
        @test !point_in_face(dcel, 1, SVector(10.0, 0.0))
    end

    @testset "insert a single line" begin
        dcel = init_bbox_dcel(-5.0, 5.0, -5.0, 5.0)
        insert_curve!(dcel, Line(SVector(1.0, 0.0), 0.0))   # x = 0
        @test length(dcel.faces) == 3
        @test length(dcel.vertices) == 6

        left_face = locate_face(dcel, SVector(-2.0, 0.0), Set(1:length(dcel.faces)))
        right_face = locate_face(dcel, SVector(2.0, 0.0), Set(1:length(dcel.faces)))
        @test left_face !== nothing
        @test right_face !== nothing
        @test left_face != right_face
        @test point_in_face(dcel, left_face, SVector(-4.9, 4.9))
        @test point_in_face(dcel, right_face, SVector(4.9, 4.9))
    end

    @testset "insert a single parabola" begin
        dcel = init_bbox_dcel(-5.0, 5.0, -5.0, 5.0)
        # focus (0,2), directrix y=-2 => vertex (0,0), y = x^2/8, crosses left/right
        # walls at x=+-5, y=3.125 (well within the top/bottom range).
        insert_curve!(dcel, Parabola(SVector(0.0, 2.0), SVector(0.0, 1.0), -2.0))
        @test length(dcel.faces) == 3
        @test length(dcel.vertices) == 6

        inside_face = locate_face(dcel, SVector(0.0, 4.0), Set(1:length(dcel.faces)))
        outside_face = locate_face(dcel, SVector(0.0, -4.0), Set(1:length(dcel.faces)))
        @test inside_face !== nothing && outside_face !== nothing
        @test inside_face != outside_face
        @test point_in_face(dcel, inside_face, SVector(0.0, 4.5))    # still above the parabola
        @test point_in_face(dcel, outside_face, SVector(4.0, -4.0))  # still below/outside
    end

    @testset "restrict_to (candidate_faces) confines the split to one face" begin
        dcel = init_bbox_dcel(-5.0, 5.0, -5.0, 5.0)
        result = insert_curve!(dcel, Line(SVector(1.0, 0.0), 0.0))  # x = 0, splits into left/right
        left_face = locate_face(dcel, SVector(-2.0, 0.0), result)
        right_face = locate_face(dcel, SVector(2.0, 0.0), result)

        insert_curve!(dcel, Line(SVector(0.0, 1.0), 0.0); candidate_faces=Set([left_face]))  # y = 0, restricted
        @test length(dcel.faces) == 4   # left face split into two; right face untouched

        # right half should still be a single face (unsplit)
        @test point_in_face(dcel, right_face, SVector(2.0, 3.0))
        @test point_in_face(dcel, right_face, SVector(2.0, -3.0))

        # left half should now be split top/bottom
        top_left = locate_face(dcel, SVector(-2.0, 3.0), Set(1:length(dcel.faces)))
        bottom_left = locate_face(dcel, SVector(-2.0, -3.0), Set(1:length(dcel.faces)))
        @test top_left != bottom_left
        @test top_left != right_face && bottom_left != right_face
    end
end
