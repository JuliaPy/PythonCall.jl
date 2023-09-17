const pyjlarraytype = pynew()

function pyjl_getaxisindex(x::AbstractUnitRange{<:Integer}, k::Py)
    if pyisslice(k)
        a = @pyconvert Union{Int,Nothing} k.start begin
            errset(pybuiltins.TypeError, "slice components must be integers")
            pythrow()
        end
        b = @pyconvert Union{Int,Nothing} k.step begin
            errset(pybuiltins.TypeError, "slice components must be integers")
            pythrow()
        end
        c = @pyconvert Union{Int,Nothing} k.stop begin
            errset(pybuiltins.TypeError, "slice components must be integers")
            pythrow()
        end
        # step defaults to 1
        b′ = b === nothing ? 1 : b
        if a === nothing && c === nothing
            # when neither is specified, start and stop default to the full range,
            # which is reversed when the step is negative
            if b′ > 0
                a′ = Int(first(x))
                c′ = Int(last(x))
            elseif b′ < 0
                a′ = Int(last(x))
                c′ = Int(first(x))
            else
                errset(pybuiltins.ValueError, "step must be non-zero")
                pythrow()
            end
        else
            # start defaults
            a′ = Int(a === nothing ? first(x) : a < 0 ? (last(x) + a + 1) : (first(x) + a))
            c′ = Int(
                c === nothing ? last(x) :
                c < 0 ? (last(x) + 1 + c - sign(b′)) : (first(x) + c - sign(b′)),
            )
        end
        r = StepRange{Int,Int}(a′, b′, c′)
        if checkbounds(Bool, x, r)
            return r
        else
            errset(pybuiltins.IndexError, "array index out of bounds")
            pythrow()
        end
    else
        j = @pyconvert Int k begin
            errset(pybuiltins.TypeError, "index must be slice or integer, got '$(pytype(k).__name__)'")
            pythrow()
        end
        r = Int(j < 0 ? (last(x) + j + 1) : (first(x) + j))
        if checkbounds(Bool, x, r)
            return r
        else
            errset(pybuiltins.IndexError, "array index out of bounds")
            pythrow()
        end
    end
end

function pyjl_getarrayindices(x::AbstractArray{T,N}, ks::Py) where {T,N}
    if pyistuple(ks)
        if pylen(ks) == N
            return ntuple(N) do i
                k = pytuple_getitem(ks, i-1)
                ans = pyjl_getaxisindex(axes(x, i), k)
                pydel!(k)
                return ans
            end
        else
            errset(pybuiltins.TypeError, "expecting $N indices, got $(pylen(ks))")
            pythrow()
        end
    elseif N == 1
        return (pyjl_getaxisindex(axes(x, 1), ks),)
    else
        errset(pybuiltins.TypeError, "expecting $N indices, got 1")
    end
end

function pyjlarray_getitem(x::AbstractArray{T,N}, k_::Py) where {T,N}
    k = pyjl_getarrayindices(x, k_)
    if k isa NTuple{N,Int}
        return Py(x[k...])
    else
        return Py(view(x, k...))
    end
end

function pyjlarray_setitem(x::AbstractArray{T,N}, k_::Py, v_::Py) where {T,N}
    k = pyjl_getarrayindices(x, k_)
    if k isa NTuple{N,Int}
        v = pyconvertarg(T, v_, "value")
        x[k...] = v
    else
        v = pyconvertarg(Any, v_, "value")
        x[k...] .= v
    end
    return Py(nothing)
end

function pyjlarray_delitem(x::AbstractArray{T,N}, k_::Py) where {T,N}
    if N == 1
        k = pyjl_getarrayindices(x, k_)
        deleteat!(x, k...)
    else
        errset(pybuiltins.TypeError, "can only delete from 1D arrays")
        pythrow()
    end
    return Py(nothing)
end
pyjl_handle_error_type(::typeof(pyjlarray_delitem), x, exc::MethodError) = exc.f === deleteat! ? pybuiltins.TypeError : PyNULL

function pyjlarray_reshape(x::AbstractArray, shape_::Py)
    shape = pyconvertarg(Union{Int,Vector{Int}}, shape_, "shape")
    return Py(reshape(x, shape...))
end

pyjlarray_isbufferabletype(::Type{T}) where {T} = T in (
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float16,
    Float32,
    Float64,
    Complex{Float16},
    Complex{Float32},
    Complex{Float64},
    Bool,
    Ptr{Cvoid},
)
pyjlarray_isbufferabletype(::Type{T}) where {T<:Tuple} =
    isconcretetype(T) &&
    PythonCall.allocatedinline(T) &&
    all(pyjlarray_isbufferabletype, fieldtypes(T))
pyjlarray_isbufferabletype(::Type{NamedTuple{names,T}}) where {names,T} =
    pyjlarray_isbufferabletype(T)

function pyjlarray_buffer_info(x::AbstractArray{T,N}) where {T,N}
    if pyjlarray_isbufferabletype(T)
        C.PyBufferInfo{N}(
            ptr = Base.unsafe_convert(Ptr{T}, x),
            readonly = !Utils.ismutablearray(x),
            itemsize = sizeof(T),
            format = pybufferformat(T),
            shape = size(x),
            strides = strides(x) .* Base.aligned_sizeof(T),
        )
    else
        error("element type is not bufferable")
    end
end

const PYBUFFERFORMAT = IdDict{Type,String}()

pybufferformat(::Type{T}) where {T} = get!(PYBUFFERFORMAT, T) do
    T == Cchar ? "b" :
    T == Cuchar ? "B" :
    T == Cshort ? "h" :
    T == Cushort ? "H" :
    T == Cint ? "i" :
    T == Cuint ? "I" :
    T == Clong ? "l" :
    T == Culong ? "L" :
    T == Clonglong ? "q" :
    T == Culonglong ? "Q" :
    T == Float16 ? "e" :
    T == Cfloat ? "f" :
    T == Cdouble ? "d" :
    T == Complex{Float16} ? "Ze" :
    T == Complex{Cfloat} ? "Zf" :
    T == Complex{Cdouble} ? "Zd" :
    T == Bool ? "?" :
    T == Ptr{Cvoid} ? "P" :
    if isstructtype(T) && isconcretetype(T) && allocatedinline(T)
        n = fieldcount(T)
        flds = []
        for i = 1:n
            nm = fieldname(T, i)
            tp = fieldtype(T, i)
            push!(flds, string(pybufferformat(tp), nm isa Symbol ? ":$nm:" : ""))
            d =
                (i == n ? sizeof(T) : fieldoffset(T, i + 1)) -
                (fieldoffset(T, i) + sizeof(tp))
            @assert d ≥ 0
            d > 0 && push!(flds, "$(d)x")
        end
        string("T{", join(flds, " "), "}")
    else
        "$(sizeof(T))x"
    end
end

pyjlarray_isarrayabletype(::Type{T}) where {T} = T in (
    UInt8,
    Int8,
    UInt16,
    Int16,
    UInt32,
    Int32,
    UInt64,
    Int64,
    Bool,
    Float16,
    Float32,
    Float64,
    Complex{Float16},
    Complex{Float32},
    Complex{Float64},
)
pyjlarray_isarrayabletype(::Type{T}) where {T<:Tuple} =
    isconcretetype(T) &&
    Base.allocatedinline(T) &&
    all(pyjlarray_isarrayabletype, T.parameters)
pyjlarray_isarrayabletype(::Type{NamedTuple{names,types}}) where {names,types} =
    pyjlarray_isarrayabletype(types)

const PYTYPESTRDESCR = IdDict{Type,Tuple{String,Py}}()

pytypestrdescr(::Type{T}) where {T} = get!(PYTYPESTRDESCR, T) do
    c = Utils.islittleendian() ? '<' : '>'
    T == Bool ? ("$(c)b$(sizeof(Bool))", PyNULL) :
    T == Int8 ? ("$(c)i1", PyNULL) :
    T == UInt8 ? ("$(c)u1", PyNULL) :
    T == Int16 ? ("$(c)i2", PyNULL) :
    T == UInt16 ? ("$(c)u2", PyNULL) :
    T == Int32 ? ("$(c)i4", PyNULL) :
    T == UInt32 ? ("$(c)u4", PyNULL) :
    T == Int64 ? ("$(c)i8", PyNULL) :
    T == UInt64 ? ("$(c)u8", PyNULL) :
    T == Float16 ? ("$(c)f2", PyNULL) :
    T == Float32 ? ("$(c)f4", PyNULL) :
    T == Float64 ? ("$(c)f8", PyNULL) :
    T == Complex{Float16} ? ("$(c)c4", PyNULL) :
    T == Complex{Float32} ? ("$(c)c8", PyNULL) :
    T == Complex{Float64} ? ("$(c)c16", PyNULL) :
    if isstructtype(T) && isconcretetype(T) && Base.allocatedinline(T)
        n = fieldcount(T)
        flds = []
        for i = 1:n
            nm = fieldname(T, i)
            tp = fieldtype(T, i)
            ts, ds = pytypestrdescr(tp)
            isempty(ts) && return ("", PyNULL)
            push!(
                flds,
                (nm isa Integer ? "f$(nm-1)" : string(nm), pyisnull(ds) ? ts : ds),
            )
            d = (i == n ? sizeof(T) : fieldoffset(T, i + 1)) - (fieldoffset(T, i) + sizeof(tp))
            @assert d ≥ 0
            d > 0 && push!(flds, ("", "|V$(d)"))
        end
        ("|$(sizeof(T))V", pylist(flds))
    else
        ("", PyNULL)
    end
end

pyjlarray_array__array(x::AbstractArray) = x isa Array ? Py(nothing) : pyjl(Array(x))
pyjlarray_array__pyobjectarray(x::AbstractArray) = pyjl(PyObjectArray(x))

function pyjlarray_array_interface(x::AbstractArray{T,N}) where {T,N}
    if pyjlarray_isarrayabletype(eltype(x))
        data = (UInt(Base.unsafe_convert(Ptr{T}, x)), !Utils.ismutablearray(x))
        typestr, descr = pytypestrdescr(T)
        if !isempty(typestr)
            d = pydict()
            d["shape"] = size(x)
            d["typestr"] = typestr
            d["data"] = data
            d["strides"] = strides(x) .* Base.aligned_sizeof(T)
            d["version"] = 3
            if !pyisnull(descr)
                d["descr"] = descr
            end
            return d
        end
    end
    errset(pybuiltins.AttributeError, "__array_interface__")
    return PyNULL
end
pyjl_handle_error_type(::typeof(pyjlarray_array_interface), x, exc) = pybuiltins.AttributeError

function init_array()
    jl = pyjuliacallmodule
    pybuiltins.exec(pybuiltins.compile("""
    $("\n"^(@__LINE__()-1))
    class ArrayValue(AnyValue):
        __slots__ = ()
        _jl_buffer_info = $(pyjl_methodnum(pyjlarray_buffer_info))
        @property
        def ndim(self):
            return self._jl_callmethod($(pyjl_methodnum(Py ∘ ndims)))
        @property
        def shape(self):
            return self._jl_callmethod($(pyjl_methodnum(Py ∘ size)))
        def copy(self):
            return self._jl_callmethod($(pyjl_methodnum(Py ∘ copy)))
        def reshape(self, shape):
            return self._jl_callmethod($(pyjl_methodnum(pyjlarray_reshape)), shape)
        def __bool__(self):
            return bool(len(self))
        def __getitem__(self, k):
            return self._jl_callmethod($(pyjl_methodnum(pyjlarray_getitem)), k)
        def __setitem__(self, k, v):
            self._jl_callmethod($(pyjl_methodnum(pyjlarray_setitem)), k, v)
        def __delitem__(self, k):
            self._jl_callmethod($(pyjl_methodnum(pyjlarray_delitem)), k)
        @property
        def __array_interface__(self):
            return self._jl_callmethod($(pyjl_methodnum(pyjlarray_array_interface)))
        def __array__(self, dtype=None):
            # convert to an array-like object
            arr = self
            if not (hasattr(arr, "__array_interface__") or hasattr(arr, "__array_struct__")):
                # the first attempt collects into an Array
                arr = self._jl_callmethod($(pyjl_methodnum(pyjlarray_array__array)))
                if not (hasattr(arr, "__array_interface__") or hasattr(arr, "__array_struct__")):
                    # the second attempt collects into a PyObjectArray
                    arr = self._jl_callmethod($(pyjl_methodnum(pyjlarray_array__pyobjectarray)))
            # convert to a numpy array if numpy is available
            try:
                import numpy
                arr = numpy.array(arr, dtype=dtype)
            except ImportError:
                pass
            return arr
        def to_numpy(self, dtype=None, copy=True, order="K"):
            import numpy
            return numpy.array(self, dtype=dtype, copy=copy, order=order)
    """, @__FILE__(), "exec"), jl.__dict__)
    pycopy!(pyjlarraytype, jl.ArrayValue)
end

pyjltype(::AbstractArray) = pyjlarraytype
