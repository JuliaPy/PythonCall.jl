using PythonCall.Utils: size_to_fstrides, size_to_cstrides, isfcontiguous, isccontiguous

@testset "strides" begin
    @test size_to_fstrides(1, (2, 3, 4)) == (1, 2, 6)
    @test size_to_fstrides(1, 2, 3, 4) == (1, 2, 6)
    @test size_to_cstrides(1, (2, 3, 4)) == (12, 4, 1)
    @test size_to_cstrides(1, 2, 3, 4) == (12, 4, 1)

    # A plain old Julia array should be Fortran-continuous.
    a = zeros(2, 3, 4)
    @test isfcontiguous(a)
    @test !isccontiguous(a)

    # But if we reverse the dimensions it should be C-continuous.
    a_perm = PermutedDimsArray(a, (3, 2, 1))
    @test !isfcontiguous(a_perm)
    @test isccontiguous(a_perm)

    # And if we do something crazy it will be neither.
    a_crazy = PermutedDimsArray(a, (2, 3, 1))
    @test !isfcontiguous(a_crazy)
    @test !isccontiguous(a_crazy)
end
