mutable struct PyObject
    ref :: PyRef
    make :: Any
    PyObject(::Val{:nocopy}, o::PyRef) = new(o)
    PyObject(o) = new(PyRef(o))
    PyObject(::Val{:lazy}, mk) = new(PyRef(), mk)
end
export PyObject

ispyreftype(::Type{PyObject}) = true
pyptr(o::PyObject) = begin
    ref = getfield(o, :ref)
    ptr = ref.ptr
    if isnull(ptr)
        val = try
            getfield(o, :make)()
        catch err
            C.PyErr_SetString(C.PyExc_Exception(), "Error retrieving object value: $err")
            return ptr
        end
        ptr = ref.ptr = C.PyObject_From(val)
    end
    ptr
end
Base.unsafe_convert(::Type{CPyPtr}, o::PyObject) = checknull(pyptr(o))
pynewobject(p::Ptr, check::Bool=false) = (check && isnull(p)) ? pythrow() : PyObject(Val(:nocopy), pynewref(p))
pyborrowedobject(p::Ptr, check::Bool=false) = (check && isnull(p)) ? pythrow() : PyObject(Val(:nocopy), pyborrowedref(p))
pylazyobject(mk) = PyObject(Val(:lazy), mk)

C.PyObject_TryConvert__initial(o, ::Type{PyObject}) = C.putresult(pyborrowedobject(o))

Base.convert(::Type{Any}, x::PyObject) = x
Base.convert(::Type{PyObject}, x::PyObject) = x
Base.convert(::Type{PyObject}, x) = PyObject(x)

Base.convert(::Type{T}, x::PyObject) where {T} = pyconvert(T, x)

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
    h, w = displaysize(io)
    h -= 3
    x = try
        pystr(String, pypprintmodule().pformat(o, width=w))
    catch
        pyrepr(String, o)
    end
    if get(io, :limit, true)
        if '\n' ∈ x
            # multiple lines
            # each one is truncated to one screen width
            print(io, "py:")
            h -= 1
            xs = split(x, '\n')
            printlines(xs) =
                for x in xs
                    if length(x) ≤ w
                        print(io, '\n', x)
                    else
                        print(io, '\n', x[1:nextind(x, 0, w-1)], '…')
                    end
                end
            if length(xs) ≤ h
                # all lines fit on screen
                printlines(xs)
            else
                # too many lines, skip the middle ones
                h -= 1
                h2 = cld(h, 2)
                h3 = (length(xs)+1)-(h-h2)
                printlines(xs[1:h2])
                linelen = min(
                    checkbounds(Bool, xs, h2) ? length(xs[h2]) : 0,
                    checkbounds(Bool, xs, h3) ? length(xs[h3]) : 0,
                    w
                )
                msg = "... [skipping $(h3-h2-1) lines] ..."
                pad = fld(linelen - length(msg), 2)
                print(io, "\n", pad > 0 ? " "^pad : "", msg)
                printlines(xs[h3:end])
            end
        elseif length(x) ≤ w-4
            # one short line
            print(io, "py: ", x)
            return
        else
            # one long line
            println(io, "py:")
            h -= 1
            a = h * w
            if length(x) ≤ a
                # whole string fits on screen
                print(io, x)
            else
                # too long, skip the middle
                h -= 1
                h2 = cld(h, 2)
                i2 = nextind(x, 0, h2 * w)
                i3 = prevind(x, ncodeunits(x)+1, (h-h2)*w)
                println(io, x[1:i2])
                println(io, " ... [skipping $(length(x[nextind(x,i2):prevind(x,i3)])) characters] ...")
                print(io, x[i3:end])
            end
        end
    else
        print(io, "py: ", x)
    end
end

Base.show(io::IO, mime::MIME, o::PyObject) = _py_mime_show(io, mime, o)
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
        ((::Type{T}=PyObject) where {T}) -> PyIterable{T}(o)
    elseif k == :jl!list
        ((::Type{T}=PyObject) where {T}) -> PyList{T}(o)
    elseif k == :jl!set
        ((::Type{T}=PyObject) where {T}) -> PySet{T}(o)
    elseif k == :jl!dict
        ((::Type{K}=PyObject, ::Type{V}=PyObject) where {K,V}) -> PyDict{K,V}(o)
    elseif k == :jl!io
        (; opts...) -> PyIO(o; opts...)
    elseif k == :jl!pandasdf
        (; opts...) -> PyPandasDataFrame(o; opts...)
    else
        pygetattr(PyObject, o, k)
    end

Base.setproperty!(o::PyObject, k::Symbol, v) = pysetattr(o, k, v)

Base.hasproperty(o::PyObject, k::Symbol) = pyhasattr(o, k)

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

Base.iterate(o::PyObject, it::PyRef=pyiter(PyRef,o)) = begin
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
Base.hash(o::PyObject) = trunc(UInt, pyhash(o))

### COMPARISON

Base.:(==)(x::PyObject, y::PyObject) = pyeq(PyObject, x, y)
Base.:(!=)(x::PyObject, y::PyObject) = pynq(PyObject, x, y)
Base.:(<=)(x::PyObject, y::PyObject) = pyle(PyObject, x, y)
Base.:(< )(x::PyObject, y::PyObject) = pylt(PyObject, x, y)
Base.:(>=)(x::PyObject, y::PyObject) = pyge(PyObject, x, y)
Base.:(> )(x::PyObject, y::PyObject) = pygt(PyObject, x, y)
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
Base.powermod(o1::PyObject, o2::Union{PyObject,Number}, o3::Union{PyObject,Number}) = pypow(PyObject, o1, o2, o3)

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
