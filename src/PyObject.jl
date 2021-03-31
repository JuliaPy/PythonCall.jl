"""
    PyObject(x)

Convert `x` to a Python object.

This is the default type returned by most API functions.

It supports attribute access (`x.attr`), indexing (`x[i,j]`, `delete!(x, i, j)`),
calling `x(a, b, kw=c)`, `length`, iteration (yielding `PyObject`s), comparisons to other
`PyObject`s (`x == y`), and arithmetic with other `PyObject`s and `Number`s (`x + y`, `x * 3`).

Note that comparisons between `PyObject`s (`==`, `≠`, `≤`, etc.) return `PyObject`,
except `isequal` and `isless` which return `Bool`. Use [`pyeq`](@ref) and friends to
return `Bool`.

A number of special properties are also defined for convenience to convert the object to
another type. To avoid clashes with object attributes, they all have the prefix `jl!`.
- `x.jl!(T)` is `pyconvert(T, x)`
- `x.jl!i` is `pyconvert(Int, x)`
- `x.jl!b` is `pytruth(x)`
- `x.jl!s` is `pystr(String, x)`.
- `x.jl!r` is `pyrepr(String, x)`.
- `x.jl!f` is `pyconvert(Float64, x)`
- `x.jl!c` is `pyconvert(Complex{Float64}, x)`
- `x.jl!iter(T=PyObject)` is `PyIterable{T}(x)`
- `x.jl!list(T=PyObject)` is `PyList{T}(x)`
- `x.jl!set(T=PyObject)` is `PySet{T}(x)`
- `x.jl!dict(K=PyObject, V=PyObject)` is `PyDict{K,V}(x)`
- `x.jl!io(...)` is `PyIO(x; ...)`
- `x.jl!pandasdf(...)` is `PyPandasDataFrame(x; ...)`
- `x.jl!buffer(...)` is `PyBuffer(x, ...)`
- `x.jl!array(...)` is `PyArray{...}(x)`
- `x.jl!vector(...)` is `PyVector{...}(x)`
- `x.jl!matrix(...)` is `PyMatrix{...}(x)`
"""
mutable struct PyObject
    ptr::CPyPtr
    make::Any
    PyObject(::Val{:new}, ptr::Ptr, borrowed::Bool) = begin
        borrowed && C.Py_IncRef(ptr)
        finalizer(pyref_finalize!, new(CPyPtr(ptr), nothing))
    end
    PyObject(::Val{:lazy}, mk) = finalizer(pyref_finalize!, new(CPyPtr(0), mk))
end
PyObject(x) = begin
    ptr = C.PyObject_From(x)
    isnull(ptr) && pythrow()
    pynewobject(ptr)
end
export PyObject

ispyreftype(::Type{PyObject}) = true
pyptr(o::PyObject) = begin
    ptr = getfield(o, :ptr)
    if isnull(ptr)
        val = try
            getfield(o, :make)()
        catch err
            C.PyErr_SetString(C.PyExc_Exception(), "Error retrieving object value: $err")
            return ptr
        end
        ptr = C.PyObject_From(val)
        setfield!(o, :ptr, ptr)
    end
    ptr
end
Base.unsafe_convert(::Type{CPyPtr}, o::PyObject) = checknull(pyptr(o))
pynewobject(p::Ptr, check::Bool = false) =
    (check && isnull(p)) ? pythrow() : PyObject(Val(:new), p, false)
pyborrowedobject(p::Ptr, check::Bool = false) =
    (check && isnull(p)) ? pythrow() : PyObject(Val(:new), p, true)
pylazyobject(mk) = PyObject(Val(:lazy), mk)

C.PyObject_TryConvert__initial(o, ::Type{PyObject}) = C.putresult(pyborrowedobject(o))

# These cause a LOT of method invalidations and slow down package load by a couple of
# seconds. You can always use PyObject and pyconvert to go in either direction.
# Base.convert(::Type{PyObject}, x::PyObject) = x
# Base.convert(::Type{Any}, x::PyObject) = x
# Base.convert(::Type{T}, x::PyObject) where {T} = x isa T ? x : pyconvert(T, x)
# Base.convert(::Type{PyObject}, x) = PyObject(x)

### Cache some common values

const _pynone = pylazyobject(() -> pynone(PyRef))
pynone(::Type{PyObject}) = _pynone

const _pytrue = pylazyobject(() -> pybool(PyRef, true))
const _pyfalse = pylazyobject(() -> pybool(PyRef, false))
pybool(::Type{PyObject}, x::Bool) = x ? _pytrue : _pyfalse

### IO

Base.string(o::PyObject) = pystr(String, o)

Base.print(io::IO, o::PyObject) = print(io, string(o))

Base.show(io::IO, o::PyObject) = begin
    s = pyrepr(String, o)
    if get(io, :typeinfo, Any) == typeof(o)
        print(io, s)
    elseif startswith(s, "<") && endswith(s, ">")
        print(io, "<py ", s[2:end])
    else
        print(io, "<py ", s, ">")
    end
end

function Base.show(io::IO, ::MIME"text/plain", o::PyObject)
    prefix = get(io, :typeinfo, Any) != typeof(o)
    h, w = displaysize(io)
    h -= 3
    x = try
        pystr(String, pypprintmodule().pformat(o, width = w))
    catch
        pyrepr(String, o)
    end
    if get(io, :limit, true)
        if '\n' ∈ x
            # multiple lines
            # each one is truncated to one screen width
            if prefix
                print(io, "py:")
                h -= 1
            end
            xs = split(x, '\n')
            printlines(xs, nl = true) =
                for (i, x) in enumerate(xs)
                    (nl || i > 1) && print(io, '\n')
                    if length(x) ≤ w
                        print(io, x)
                    else
                        print(io, x[1:nextind(x, 0, w - 1)], '…')
                    end
                end
            if length(xs) ≤ h
                # all lines fit on screen
                printlines(xs, prefix)
            else
                # too many lines, skip the middle ones
                h -= 1
                h2 = cld(h, 2)
                h3 = (length(xs) + 1) - (h - h2)
                printlines(xs[1:h2], prefix)
                linelen = min(
                    checkbounds(Bool, xs, h2) ? length(xs[h2]) : 0,
                    checkbounds(Bool, xs, h3) ? length(xs[h3]) : 0,
                    w,
                )
                msg = "... [skipping $(h3-h2-1) lines] ..."
                pad = fld(linelen - length(msg), 2)
                print(io, "\n", pad > 0 ? " "^pad : "", msg)
                printlines(xs[h3:end])
            end
        elseif length(x) ≤ (prefix ? w - 4 : w)
            # one short line
            print(io, prefix ? "py: " : "", x)
            return
        else
            # one long line
            if prefix
                println(io, "py:")
                h -= 1
            end
            a = h * w
            if length(x) ≤ a
                # whole string fits on screen
                print(io, x)
            else
                # too long, skip the middle
                h -= 1
                h2 = cld(h, 2)
                i2 = nextind(x, 0, h2 * w)
                i3 = prevind(x, ncodeunits(x) + 1, (h - h2) * w)
                println(io, x[1:i2])
                println(
                    io,
                    " ... [skipping $(length(x[nextind(x,i2):prevind(x,i3)])) characters] ...",
                )
                print(io, x[i3:end])
            end
        end
    else
        print(io, prefix ? "py: " : "", x)
    end
end

Base.show(io::IO, mime::MIME, o::PyObject) = _py_mime_show(io, mime, o)
Base.show(io::IO, mime::MIME"text/csv", o::PyObject) = _py_mime_show(io, mime, o)
Base.show(io::IO, mime::MIME"text/tab-separated-values", o::PyObject) = _py_mime_show(io, mime, o)
Base.showable(mime::MIME, o::PyObject) = _py_mime_showable(mime, o)

### PROPERTIES

Base.getproperty(o::PyObject, k::Symbol) =
    if k == :jl!
        ((::Type{T}) where {T}) -> pyconvert(T, o)
    elseif k == :jl!i
        pyconvert(Int, o)
    elseif k == :jl!b
        pytruth(o)
    elseif k == :jl!s
        pystr(String, o)
    elseif k == :jl!r
        pyrepr(String, o)
    elseif k == :jl!f
        pyconvert(Cdouble, o)
    elseif k == :jl!c
        pyconvert(Complex{Cdouble}, o)
    elseif k == :jl!iter
        ((::Type{T} = PyObject) where {T}) -> PyIterable{T}(o)
    elseif k == :jl!list
        ((::Type{T} = PyObject) where {T}) -> PyList{T}(o)
    elseif k == :jl!set
        ((::Type{T} = PyObject) where {T}) -> PySet{T}(o)
    elseif k == :jl!dict
        ((::Type{K} = PyObject, ::Type{V} = PyObject) where {K,V}) -> PyDict{K,V}(o)
    elseif k == :jl!io
        (; opts...) -> PyIO(o; opts...)
    elseif k == :jl!pandasdf
        (; opts...) -> PyPandasDataFrame(o; opts...)
    elseif k == :jl!buffer
        (args...) -> PyBuffer(o, args...)
    elseif k == :jl!array
        f_arr() = PyArray(o)
        f_arr(::Type{T}) where {T} = PyArray{T}(o)
        f_arr(::Type{T}, N::Integer) where {T} = PyArray{T,N}(o)
        f_arr(::Type{T}, N::Integer, ::Type{R}) where {T,R} = PyArray{T,N,R}(o)
        f_arr(::Type{T}, N::Integer, ::Type{R}, M::Bool) where {T,R} = PyArray{T,N,R,M}(o)
        f_arr(::Type{T}, N::Integer, ::Type{R}, M::Bool, L::Bool) where {T,R} =
            PyArray{T,N,R,M,L}(o)
        f_arr
    elseif k == :jl!vector
        f_vec() = PyVector(o)
        f_vec(::Type{T}) where {T} = PyVector{T}(o)
        f_vec(::Type{T}, ::Type{R}) where {T,R} = PyVector{T,R}(o)
        f_vec(::Type{T}, ::Type{R}, M::Bool) where {T,R} = PyVector{T,R,M}(o)
        f_vec(::Type{T}, ::Type{R}, M::Bool, L::Bool) where {T,R} = PyVector{T,R,M,L}(o)
        f_vec
    elseif k == :jl!matrix
        f_mat() = PyMatrix(o)
        f_mat(::Type{T}) where {T} = PyMatrix{T}(o)
        f_mat(::Type{T}, ::Type{R}) where {T,R} = PyMatrix{T,R}(o)
        f_mat(::Type{T}, ::Type{R}, M::Bool) where {T,R} = PyMatrix{T,R,M}(o)
        f_mat(::Type{T}, ::Type{R}, M::Bool, L::Bool) where {T,R} = PyMatrix{T,R,M,L}(o)
        f_mat
    else
        pygetattr(PyObject, o, k)
    end

Base.setproperty!(o::PyObject, k::Symbol, v) = pysetattr(o, k, v)

if hasproperty(Base, :hasproperty)
    Base.hasproperty(o::PyObject, k::Symbol) = pyhasattr(o, k)
else
    Compat.hasproperty(o::PyObject, k::Symbol) = pyhasattr(o, k)
end

function Base.propertynames(o::PyObject)
    # this follows the logic of rlcompleter.py
    @py ```
    def members(o):
        def classmembers(c):
            r = dir(c)
            if hasattr(c, '__bases__'):
                for b in c.__bases__:
                    r += classmembers(b)
            return r
        words = set(dir(o))
        words.discard('__builtins__')
        if hasattr(o, '__class__'):
            words.add('__class__')
            words.update(classmembers(o.__class__))
        return words
    $(words::Vector{Symbol}) = members($o)
    ```
    words
end

### CALL

(f::PyObject)(args...; kwargs...) = pycall(PyObject, f, args...; kwargs...)

### ITERATE & INDEX

Base.IteratorSize(::Type{PyObject}) = Base.SizeUnknown()
Base.eltype(::Type{PyObject}) = PyObject

Base.getindex(o::PyObject, k) = pygetitem(PyObject, o, k)
Base.getindex(o::PyObject, k...) = pygetitem(PyObject, o, k)
Base.setindex!(o::PyObject, v, k) = pysetitem(o, k, v)
Base.setindex!(o::PyObject, v, k...) = pysetitem(o, k, v)
Base.delete!(o::PyObject, k) = pydelitem(o, k)
Base.delete!(o::PyObject, k...) = pydelitem(o, k)

Base.length(o::PyObject) = Int(pylen(o))

Base.iterate(o::PyObject, it::PyRef = pyiter(PyRef, o)) = begin
    vo = C.PyIter_Next(it)
    if !isnull(vo)
        (pynewobject(vo), it)
    elseif C.PyErr_IsSet()
        pythrow()
    else
        nothing
    end
end

Base.in(x, o::PyObject) = pycontains(o, x)
Base.hash(o::PyObject) = reinterpret(UInt, Int(pyhash(o)))

### COMPARISON

Base.:(==)(x::PyObject, y::PyObject) = pyeq(PyObject, x, y)
Base.:(!=)(x::PyObject, y::PyObject) = pyne(PyObject, x, y)
Base.:(<=)(x::PyObject, y::PyObject) = pyle(PyObject, x, y)
Base.:(<)(x::PyObject, y::PyObject) = pylt(PyObject, x, y)
Base.:(>=)(x::PyObject, y::PyObject) = pyge(PyObject, x, y)
Base.:(>)(x::PyObject, y::PyObject) = pygt(PyObject, x, y)
Base.isequal(x::PyObject, y::PyObject) = pyeq(Bool, x, y)
Base.isless(x::PyObject, y::PyObject) = pylt(Bool, x, y)

### ARITHMETIC

Base.zero(::Type{PyObject}) = pyint(0)
Base.one(::Type{PyObject}) = pyint(1)

# unary
Base.:(-)(o::PyObject) = pyneg(PyObject, o)
Base.:(+)(o::PyObject) = pypos(PyObject, o)
Base.abs(o::PyObject) = pyabs(PyObject, o)
Base.:(~)(o::PyObject) = pyinv(PyObject, o)

# binary
Base.:(+)(o1::PyObject, o2::PyObject) = pyadd(PyObject, o1, o2)
Base.:(-)(o1::PyObject, o2::PyObject) = pysub(PyObject, o1, o2)
Base.:(*)(o1::PyObject, o2::PyObject) = pymul(PyObject, o1, o2)
Base.:(/)(o1::PyObject, o2::PyObject) = pytruediv(PyObject, o1, o2)
Base.:fld(o1::PyObject, o2::PyObject) = pyfloordiv(PyObject, o1, o2)
Base.:mod(o1::PyObject, o2::PyObject) = pymod(PyObject, o1, o2)
Base.:(^)(o1::PyObject, o2::PyObject) = pypow(PyObject, o1, o2)
Base.:(<<)(o1::PyObject, o2::PyObject) = pylshift(PyObject, o1, o2)
Base.:(>>)(o1::PyObject, o2::PyObject) = pyrshift(PyObject, o1, o2)
Base.:(&)(o1::PyObject, o2::PyObject) = pyand(PyObject, o1, o2)
Base.:xor(o1::PyObject, o2::PyObject) = pyxor(PyObject, o1, o2)
Base.:(|)(o1::PyObject, o2::PyObject) = pyor(PyObject, o1, o2)

Base.:(+)(o1::PyObject, o2::Number) = pyadd(PyObject, o1, o2)
Base.:(-)(o1::PyObject, o2::Number) = pysub(PyObject, o1, o2)
Base.:(*)(o1::PyObject, o2::Number) = pymul(PyObject, o1, o2)
Base.:(/)(o1::PyObject, o2::Number) = pytruediv(PyObject, o1, o2)
Base.:fld(o1::PyObject, o2::Number) = pyfloordiv(PyObject, o1, o2)
Base.:mod(o1::PyObject, o2::Number) = pymod(PyObject, o1, o2)
Base.:(^)(o1::PyObject, o2::Number) = pypow(PyObject, o1, o2)
Base.:(<<)(o1::PyObject, o2::Number) = pylshift(PyObject, o1, o2)
Base.:(>>)(o1::PyObject, o2::Number) = pyrshift(PyObject, o1, o2)
Base.:(&)(o1::PyObject, o2::Number) = pyand(PyObject, o1, o2)
Base.:xor(o1::PyObject, o2::Number) = pyxor(PyObject, o1, o2)
Base.:(|)(o1::PyObject, o2::Number) = pyor(PyObject, o1, o2)

Base.:(+)(o1::Number, o2::PyObject) = pyadd(PyObject, o1, o2) # Defining +(::Any, ::PyObject) like this hangs Julia v1.5.2-v1.5.3 (at least) during precompilation
Base.:(-)(o1::Number, o2::PyObject) = pysub(PyObject, o1, o2)
Base.:(*)(o1::Number, o2::PyObject) = pymul(PyObject, o1, o2)
Base.:(/)(o1::Number, o2::PyObject) = pytruediv(PyObject, o1, o2)
Base.:fld(o1::Number, o2::PyObject) = pyfloordiv(PyObject, o1, o2)
Base.:mod(o1::Number, o2::PyObject) = pymod(PyObject, o1, o2)
# Base.:(^)(o1::Number, o2::PyObject) = pypow(PyObject, o1, o2)
# Base.:(<<)(o1::Number, o2::PyObject) = pylshift(PyObject, o1, o2)
# Base.:(>>)(o1::Number, o2::PyObject) = pyrshift(PyObject, o1, o2)
Base.:(&)(o1::Number, o2::PyObject) = pyand(PyObject, o1, o2)
Base.:xor(o1::Number, o2::PyObject) = pyxor(PyObject, o1, o2)
Base.:(|)(o1::Number, o2::PyObject) = pyor(PyObject, o1, o2)

# ternary
Base.powermod(o1::PyObject, o2::Union{PyObject,Number}, o3::Union{PyObject,Number}) =
    pypow(PyObject, o1, o2, o3)

### DOCUMENTATION

# function Base.Docs.getdoc(o::PyObject)
#     function tryget(f, g=identity)
#         a = try
#             f()
#         catch
#             return nothing
#         end
#         pyisnone(a) ? nothing : g(a)
#     end
#     # function name(o)
#     #     n = tryget(()->o.__name__)
#     #     n === nothing && return nothing
#     #     m = tryget(()->o.__module__)
#     #     (m === nothing || string(m) == "builtins") ? string(n) : string(m, ".", n)
#     # end
#     getname(o) = tryget(()->o.__name__, string)
#     getdoc(o) = tryget(()->o.__doc__, string ∘ ins.cleandoc)

#     docs = []

#     # Short description
#     ins = pyimport("inspect")
#     desc, name = if ins.ismodule(o).jl!b
#         (pyhasattr(o, "__path__") ? "package" : "module"), gname(o)
#     elseif ins.isgetsetdescriptor(o).jl!b
#         ("getset descriptor", "$(getname(o.__objclass__)).$(o.__name__)")
#     elseif ins.ismemberdescriptor(o).jl!b
#         ("member descriptor", "$(getname(o.__objclass__)).$(o.__name__)")
#     elseif ins.isclass(o).jl!b
#         ("class", getname(o))
#     elseif ins.isfunction(o).jl!b || ins.isbuiltin(o).jl!b
#         ("function", getname(o))
#     elseif ins.ismethod(o).jl!b
#         ("method", getname(o))
#     else
#         ("object of type", getname(pytype(o)))
#     end
#     push!(docs, Markdown.Paragraph(["Python ", desc, " ", Markdown.Code(name), "."]))

#     if ins.isroutine(o).jl!b || ins.isclass(o).jl!b
#         try
#             push!(docs, Markdown.Code("python", "$(o.__name__)$(ins.signature(o))"))
#         catch
#         end
#     end

#     # Maybe document the class instead
#     doc = getdoc(o)
#     if doc === nothing && !ins.ismodule(o).jl!b && !ins.isclass(o).jl!b && !ins.isroutine(o).jl!b && !ins.isdatadescriptor(o).jl!b
#         o = pyhasattr(o, "__origin__") ? o.__origin__ : pytype(o)
#         doc = getdoc(o)
#     end
#     doc === nothing || push!(docs, Markdown.Paragraph([Markdown.Text(doc)]))

#     # Done
#     Markdown.MD(docs)
# end
# Base.Docs.Binding(o::PyObject, k::Symbol) = getproperty(o, k)
