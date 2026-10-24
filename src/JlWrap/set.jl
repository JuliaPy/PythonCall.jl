const pyjlsettype = pynew()

pyjlset_add(x::AbstractSet, v::Py) =
    (push!(x, pyconvertarg(eltype(x), v, "value")); Py(nothing))

function pyjlset_discard(x::AbstractSet, v_::Py)
    v = @pyconvert(eltype(x), v_, return Py(nothing))
    delete!(x, v)
    Py(nothing)
end

pyjlset_clear(x::AbstractSet) = (empty!(x); Py(nothing))

function pyjlset_pop(x::AbstractSet)
    if isempty(x)
        errset(pybuiltins.KeyError, "pop from an empty set")
        PyNULL
    else
        Py(pop!(x))
    end
end

function pyjlset_remove(x::AbstractSet, v_::Py)
    v = @pyconvert eltype(x) v_ begin
        errset(pybuiltins.KeyError, v_)
        return PyNULL
    end
    if v in x
        delete!(x, v)
        return Py(nothing)
    else
        errset(pybuiltins.KeyError, v_)
        return PyNULL
    end
end

function pyjlset_update(x::AbstractSet, vs_::Py)
    for v_ in vs_
        v = pyconvert(eltype(x), v_)
        push!(x, v)
        pydel!(v_)
    end
    Py(nothing)
end

function pyjlset_difference_update(x::AbstractSet, vs_::Py)
    for v_ in vs_
        v = @pyconvert(eltype(x), v_, continue)
        delete!(x, v)
        pydel!(v_)
    end
    Py(nothing)
end

function pyjlset_intersection_update(x::AbstractSet, vs_::Py)
    vs = Set{eltype(x)}()
    for v_ in vs_
        v = @pyconvert(eltype(x), v_, continue)
        push!(vs, v)
        pydel!(v_)
    end
    intersect!(x, vs)
    Py(nothing)
end

function pyjlset_symmetric_difference_update(x::AbstractSet, vs_::Py)
    vs = Set{eltype(x)}()
    for v_ in vs_
        v = pyconvert(eltype(x), v_)
        push!(vs, v)
        pydel!(v_)
    end
    symdiff!(x, vs)
    Py(nothing)
end

function init_set()
    jl = pyjuliacallmodule
    m = pyimport("juliacall._jlwrap.set")
    methods = m._jl_methods
    methods["add"] = pyjl_methodnum(pyjlset_add)
    methods["discard"] = pyjl_methodnum(pyjlset_discard)
    methods["clear"] = pyjl_methodnum(pyjlset_clear)
    methods["copy"] = pyjl_methodnum(Py ∘ copy)
    methods["pop"] = pyjl_methodnum(pyjlset_pop)
    methods["remove"] = pyjl_methodnum(pyjlset_remove)
    methods["difference_update"] = pyjl_methodnum(pyjlset_difference_update)
    methods["intersection_update"] = pyjl_methodnum(pyjlset_intersection_update)
    methods["symmetric_difference_update"] = pyjl_methodnum(pyjlset_symmetric_difference_update)
    methods["update"] = pyjl_methodnum(pyjlset_update)
    pycopy!(pyjlsettype, jl.SetValue)
end

pyjltype(::AbstractSet) = pyjlsettype
