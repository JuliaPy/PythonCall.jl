@testitem "mimes_for" begin
    for x in Any[1, "foo", [], 'z']
        @test PythonCall.Utils.mimes_for(x) isa Vector{String}
    end
end

@testitem "StaticString length and indexing" begin
    s = PythonCall.Utils.StaticString{UInt32, 44}("ababababb")
    @test length(s) == 9
    @test s[1] == 'a'
    @test s[1:2] == "ab"
    @test s[1:2:end] == "aaaab"
end
