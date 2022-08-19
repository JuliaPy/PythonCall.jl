@testset "mimes_for" begin
    for x in Any[1, "foo", [], 'z']
        @test PythonCall.Utils.mimes_for(x) isa Vector{String}
    end
end
