@testitem "Aqua" begin
    import Aqua
    Aqua.test_all(PythonCall; stale_deps=(; ignore=[:CondaPkg]))
end
