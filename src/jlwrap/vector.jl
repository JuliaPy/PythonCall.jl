const pyjlvectortype = pynew()

function pyjlvector_resize(x::AbstractVector, size_::Py)
    size = pyconvertarg(Int, size_, "size")
    resize!(x, size)
    Py(nothing)
end

function pyjlvector_sort(x::AbstractVector, reverse_::Py, key_::Py)
    reverse = pyconvertarg(Bool, reverse_, "reverse")
    key = pyconvertarg(Any, key_, "size")
    if key === nothing
        sort!(x, rev=reverse)
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
    a = axes(x, 1)
    k′ = k < 0 ? (last(a) + 1 + k) : (first(a) + k)
    if checkbounds(Bool, x, k′) || k′ == last(a)+1
        v = pyconvertarg(eltype(x), v_, "value")
        insert!(x, k′, v)
        return Py(nothing)
    else
        errset(pybuiltins.IndexError, "array index out of bounds");
        return pynew()
    end
end

function pyjlvector_append(x::AbstractVector, v_::Py)
    v = pyconvertarg(Int, v_, "value")
    push!(x, v)
    Py(nothing)
end

function pyjlvector_extend(x::AbstractVector, vs_::Py)
    for v_ in vs_
        v = pyconvert_and_del(eltype(x), v_)
        push!(x, v)
    end
    Py(nothing)
end

function pyjlvector_pop(x::AbstractVector, k_::Py)
    k = pyconvertarg(Int, k_, "index")
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
        return pynew()
    end
end

function pyjlvector_remove(x::AbstractVector, v_::Py)
    v = @pyconvert eltype(x) v_ begin
        errset(pybuiltins.ValueError, "value not in array")
        return pynew()
    end
    v = pyconvert_result(r)
    k = findfirst(==(v), x)
    if k === nothing
        errset(pybuiltins.ValueError, "value not in array")
        return pynew()
    end
    deleteat!(x, k)
    Py(nothing)
end

function pyjlvector_index(x::AbstractVector, v_::Py)
    v = @pyconvert eltype(x) v_ begin
        errset(pybuiltins.ValueError, "value not in array")
        return pynew()
    end
    k = findfirst(==(v), x)
    if k === nothing
        errset(pybuiltins.ValueError, "value not in array")
        return pynew()
    end
    Py(k - first(axes(x, 1)))
end

function pyjlvector_count(x::AbstractVector, v_::Py)
    v = @pyconvert(eltype(x), v_, (return Py(0)))
    Py(count(==(v), x))
end

function init_jlwrap_vector()
    jl = pyjuliacallmodule
    filename = "$(@__FILE__):$(1+@__LINE__)"
    pybuiltins.exec(pybuiltins.compile("""
    class VectorValue(ArrayValue):
        __slots__ = ()
        __module__ = "juliacall"
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
    """, filename, "exec"), jl.__dict__)
    pycopy!(pyjlvectortype, jl.VectorValue)
end

pyjltype(::AbstractVector) = pyjlvectortype
