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
        sort!(x, rev=reverse)
        pydel!(key_)
    else
        sort!(x, rev=reverse, by=key)
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
    if checkbounds(Bool, x, k′) || k′ == last(a)+1
        v = pyconvertarg(eltype(x), v_, "value")
        insert!(x, k′, v)
        return Py(nothing)
    else
        errset(pybuiltins.IndexError, "array index out of bounds");
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
    pybuiltins.exec(pybuiltins.compile("""
    $("\n"^(@__LINE__()-1))
    class VectorValue(ArrayValue):
        __slots__ = ()
        def resize(self, size):
            return self._jl_callmethod($(pyjl_methodnum(pyjlvector_resize)), size)
        def sort(self, reverse=False, key=None):
            return self._jl_callmethod($(pyjl_methodnum(pyjlvector_sort)), reverse, key)
        def reverse(self):
            return self._jl_callmethod($(pyjl_methodnum(pyjlvector_reverse)))
        def clear(self):
            return self._jl_callmethod($(pyjl_methodnum(pyjlvector_clear)))
        def __reversed__(self):
            return self._jl_callmethod($(pyjl_methodnum(pyjlvector_reversed)))
        def insert(self, index, value):
            return self._jl_callmethod($(pyjl_methodnum(pyjlvector_insert)), index, value)
        def append(self, value):
            return self._jl_callmethod($(pyjl_methodnum(pyjlvector_append)), value)
        def extend(self, values):
            return self._jl_callmethod($(pyjl_methodnum(pyjlvector_extend)), values)
        def pop(self, index=-1):
            return self._jl_callmethod($(pyjl_methodnum(pyjlvector_pop)), index)
        def remove(self, value):
            return self._jl_callmethod($(pyjl_methodnum(pyjlvector_remove)), value)
        def index(self, value):
            return self._jl_callmethod($(pyjl_methodnum(pyjlvector_index)), value)
        def count(self, value):
            return self._jl_callmethod($(pyjl_methodnum(pyjlvector_count)), value)
    import collections.abc
    collections.abc.MutableSequence.register(VectorValue)
    del collections
    """, @__FILE__(), "exec"), jl.__dict__)
    pycopy!(pyjlvectortype, jl.VectorValue)
end

pyjlarray(x::AbstractVector) = pyjl(pyjlvectortype, x)
