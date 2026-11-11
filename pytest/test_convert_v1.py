def test_native_containers_at_juliacall_boundary():
    from juliacall import Main as jl

    jl.jl_eval("using PythonCall")
    assert bool(jl.jl_eval("x -> x isa Vector{Int}")([1, 2]))
    assert bool(jl.jl_eval("x -> x isa Dict{String,Int}")({"a": 1}))
    assert bool(jl.jl_eval("x -> x isa Vector{Real}")([1, 2.5]))
    assert bool(jl.jl_eval("x -> x isa PyIterable")(iter([1, 2])))


def test_native_conversion_does_not_consume_iterators():
    from juliacall import Main as jl

    jl.jl_eval("using PythonCall")
    iterator = iter([1, 2, 3])
    assert bool(jl.jl_eval("x -> x isa PyIterable")(iterator))
    assert next(iterator) == 1
