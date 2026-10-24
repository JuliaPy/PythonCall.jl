const pyjlvectortype = pynew()

function pyjlvector_resize(x::AbstractVector, size_::Py)
    size = pyconvertarg(Int, size_, "size")
    pydel!(size_)
    resize!(x, size)
    Py(nothing)
end

function pyjlvector_sort(x::AbstractVector, reverse_::Py, key_::Py)
    reverse = pyconvertarg(Bool, reverse_, "reverse")
    pydel!(reverse_)
    key = pyconvertarg(Any, key_, "size")
    if key === nothing
        sort!(x, rev = reverse)
        pydel!(key_)
    else
        sort!(x, rev = reverse, by = key)
    end
    Py(nothing)
end

function pyjlvector_reverse(x::AbstractVector)
    reverse!(x)
    Py(nothing)
end

function pyjlvector_clear(x::AbstractVector)
    empty!(x)
    Py(nothing)
end

function pyjlvector_reversed(x::AbstractVector)
    Py(reverse(x))
end

function pyjlvector_insert(x::AbstractVector, k_::Py, v_::Py)
    k = pyconvertarg(Int, k_, "index")
    pydel!(k_)
    a = axes(x, 1)
    k′ = k < 0 ? (last(a) + 1 + k) : (first(a) + k)
    if checkbounds(Bool, x, k′) || k′ == last(a) + 1
        v = pyconvertarg(eltype(x), v_, "value")
        insert!(x, k′, v)
        return Py(nothing)
    else
        errset(pybuiltins.IndexError, "array index out of bounds")
        return PyNULL
    end
end

function pyjlvector_append(x::AbstractVector, v_::Py)
    v = pyconvertarg(eltype(x), v_, "value")
    push!(x, v)
    Py(nothing)
end

function pyjlvector_extend(x::AbstractVector, vs_::Py)
    for v_ in vs_
        v = pyconvert(eltype(x), v_)
        push!(x, v)
    end
    pydel!(vs_)
    Py(nothing)
end

function pyjlvector_pop(x::AbstractVector, k_::Py)
    k = pyconvertarg(Int, k_, "index")
    pydel!(k_)
    a = axes(x, 1)
    k′ = k < 0 ? (last(a) + 1 + k) : (first(a) + k)
    if checkbounds(Bool, x, k′)
        if k′ == last(a)
            v = pop!(x)
        elseif k′ == first(a)
            v = popfirst!(x)
        else
            v = x[k′]
            deleteat!(x, k′)
        end
        return Py(v)
    else
        errset(pybuiltins.IndexError, "pop from empty array")
        return PyNULL
    end
end

function pyjlvector_remove(x::AbstractVector, v_::Py)
    v = @pyconvert eltype(x) v_ begin
        errset(pybuiltins.ValueError, "value not in array")
        return PyNULL
    end
    k = findfirst(==(v), x)
    if k === nothing
        errset(pybuiltins.ValueError, "value not in array")
        return PyNULL
    end
    deleteat!(x, k)
    Py(nothing)
end

function pyjlvector_index(x::AbstractVector, v_::Py)
    v = @pyconvert eltype(x) v_ begin
        errset(pybuiltins.ValueError, "value not in array")
        return PyNULL
    end
    k = findfirst(==(v), x)
    if k === nothing
        errset(pybuiltins.ValueError, "value not in array")
        return PyNULL
    end
    Py(k - first(axes(x, 1)))
end

function pyjlvector_count(x::AbstractVector, v_::Py)
    v = @pyconvert(eltype(x), v_, (return Py(0)))
    Py(count(==(v), x))
end

function init_vector()
    jl = pyjuliacallmodule
    m = pyimport("juliacall._jlwrap.vector")
    methods = m._jl_methods
    methods["resize"] = pyjl_methodnum(pyjlvector_resize)
    methods["sort"] = pyjl_methodnum(pyjlvector_sort)
    methods["reverse"] = pyjl_methodnum(pyjlvector_reverse)
    methods["clear"] = pyjl_methodnum(pyjlvector_clear)
    methods["__reversed__"] = pyjl_methodnum(pyjlvector_reversed)
    methods["insert"] = pyjl_methodnum(pyjlvector_insert)
    methods["append"] = pyjl_methodnum(pyjlvector_append)
    methods["extend"] = pyjl_methodnum(pyjlvector_extend)
    methods["pop"] = pyjl_methodnum(pyjlvector_pop)
    methods["remove"] = pyjl_methodnum(pyjlvector_remove)
    methods["index"] = pyjl_methodnum(pyjlvector_index)
    methods["count"] = pyjl_methodnum(pyjlvector_count)
    pycopy!(pyjlvectortype, jl.VectorValue)
end

pyjltype(::AbstractVector) = pyjlvectortype
