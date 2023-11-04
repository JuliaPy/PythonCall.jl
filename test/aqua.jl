@testitem "Aqua" begin
    # The unbound_args test fails on methods with signature like foo(::Type{Tuple{Vararg{V}}}) where V
    # Seems like a bug.
    # Project.toml formatting is not consistent between Julia versions.
    import Aqua
    Aqua.test_all(PythonCall, unbound_args=false, project_toml_formatting=Base.VERSION >= v"1.9")
end
