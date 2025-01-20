@testitem "Aqua" begin
    # The unbound_args test fails on methods with signature like foo(::Type{Tuple{Vararg{V}}}) where V
    # Seems like a bug.
    import Aqua
    Aqua.test_all(PythonCall, unbound_args = false)
end
