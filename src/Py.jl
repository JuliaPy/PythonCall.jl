incref(x::C.PyPtr) = (C.Py_IncRef(x); x)
decref(x::C.PyPtr) = (C.Py_DecRef(x); x)

"""
    ispy(x)

True if `x` is a Python object.

This includes `Py` and Python wrapper types such as `PyList`.
"""
ispy(x) = false
export ispy

"""
    pyisnull(x)

True if the Python object `x` is NULL.
"""
pyisnull(x) = getptr(x) == C.PyNULL

"""
    getptr(x)

Get the underlying pointer from the Python object `x`.
"""
getptr(x) = ispy(x) ? getptr(Py(x)::Py) : throw(MethodError(getptr, (x,)))

"""
    Py(x)

Convert `x` to a Python object, of type `Py`.

Conversion happens according to [`these rules`](@ref jl2py-conversion).

Such an object supports attribute access (`obj.attr`), indexing (`obj[idx]`), calling
(`obj(arg1, arg2)`), iteration (`for x in obj`), arithmetic (`obj + obj2`) and comparison
(`obj > obj2`), among other things. These operations convert all their arguments to `Py` and
return `Py`.
"""
mutable struct Py
    ptr :: C.PyPtr
    Py(::Val{:new}, ptr::C.PyPtr) = finalizer(py_finalizer, new(ptr))
end
export Py

function py_finalizer(x::Py)
    if C.CTX.is_initialized
        C.with_gil(false) do
            C.Py_DecRef(getptr(x))
        end
    end
end

ispy(::Py) = true
getptr(x::Py) = getfield(x, :ptr)

setptr!(x::Py, ptr::C.PyPtr) = (setfield!(x, :ptr, ptr); x)

const PYNULL_CACHE = Py[]

"""
    pynew([ptr])

A new `Py` representing the Python object at `ptr` (NULL by default).

If `ptr` is given and non-NULL, this function steals a reference to the Python object it
points at, i.e. the new `Py` object owns a reference.

Note that NULL Python objects are not safe in the sense that most API functions will probably
crash your Julia session if you pass a NULL argument.
"""
pynew() =
    if isempty(PYNULL_CACHE)
        Py(Val(:new), C.PyNULL)
    else
        pop!(PYNULL_CACHE)
    end

const PyNULL = pynew()

pynew(ptr::C.PyPtr) = setptr!(pynew(), ptr)

pynew(x::Py) = pynew(incref(getptr(x)))

"""
    pycopy!(dst::Py, src)

Copy the Python object `src` into `dst`, so that they both represent the same object.

This function exists to support module-level constant Python objects. It is illegal to call
most PythonCall API functions at the top level of a module (i.e. before `__init__()` has run)
so you cannot do `const x = pything()` at the top level. Instead do `const x = pynew()` at
the top level then `pycopy!(x, pything())` inside `__init__()`.

Assumes `dst` is NULL, otherwise a memory leak will occur.
"""
pycopy!(dst::Py, src) = GC.@preserve src setptr!(dst, incref(getptr(src)))

"""
    pydel!(x::Py)

Delete the Python object `x`.

DANGER! Use this function ONLY IF the Julia object `x` could have been garbage-collected
anyway, i.e. was about to become unreachable. This means you MUST KNOW that no other part of
the program has the Julia object `x`.

This decrements the reference count, sets the pointer to NULL and appends `x` to a cache
of unused objects (`PYNULL_CACHE`).

This is an optimization to avoid excessive allocation and deallocation in Julia, which can
be a significant source of slow-down in code which uses a lot of Python objects. It allows
`pynew()` to pop an item from `PYNULL_CACHE` instead of allocating one, and avoids calling
the relatively slow finalizer on `x`.
"""
function pydel!(x::Py)
    ptr = getptr(x)
    if ptr != C.PyNULL
        C.Py_DecRef(ptr)
        setptr!(x, C.PyNULL)
    end
    push!(PYNULL_CACHE, x)
    return
end

macro autopy(args...)
    vs = args[1:end-1]
    ts = [Symbol(v, "_") for v in vs]
    body = args[end]
    # ans = gensym("ans")
    esc(quote
        # $([:($t = $ispy($v) ? $v : $Py($v)) for (t, v) in zip(ts, vs)]...)
        # $ans = $body
        # $([:($ispy($v) || $pydel!($t)) for (t, v) in zip(ts, vs)]...)
        # $ans
        $([:($t = $Py($v)) for (t, v) in zip(ts, vs)]...)
        $body
    end)
end

Py(x::Py) = x
Py(x::Nothing) = pybuiltins.None
Py(x::Bool) = x ? pybuiltins.True : pybuiltins.False
Py(x::Union{String, SubString{String}, Char}) = pystr(x)
Py(x::Base.CodeUnits{UInt8, String}) = pybytes(x)
Py(x::Base.CodeUnits{UInt8, SubString{String}}) = pybytes(x)
Py(x::Tuple) = pytuple_fromiter(x)
Py(x::Pair) = pytuple_fromiter(x)
Py(x::Union{Int8,Int16,Int32,Int64,Int128,UInt8,UInt16,UInt32,UInt64,UInt128,BigInt}) = pyint(x)
Py(x::Rational{<:Union{Int8,Int16,Int32,Int64,Int128,UInt8,UInt16,UInt32,UInt64,UInt128,BigInt}}) = pyfraction(x)
Py(x::Union{Float16,Float32,Float64}) = pyfloat(x)
Py(x::Complex{<:Union{Float16,Float32,Float64}}) = pycomplex(x)
Py(x::AbstractRange{<:Union{Int8,Int16,Int32,Int64,Int128,UInt8,UInt16,UInt32,UInt64,UInt128,BigInt}}) = pyrange_fromrange(x)
Py(x::Date) = pydate(x)
Py(x::Time) = pytime(x)
Py(x::DateTime) = pydatetime(x)
Py(x) = ispy(x) ? throw(MethodError(Py, (x,))) : pyjl(x)

Base.string(x::Py) = pyisnull(x) ? "<py NULL>" : pystr(String, x)
Base.print(io::IO, x::Py) = print(io, string(x))

function Base.show(io::IO, x::Py)
    if get(io, :typeinfo, Any) == Py
        if getptr(x) == C.PyNULL
            print(io, "NULL")
        else
            print(io, pyrepr(String, x))
        end
    else
        if getptr(x) == C.PyNULL
            print(io, "<py NULL>")
        else
            s = pyrepr(String, x)
            if startswith(s, "<") && endswith(s, ">")
                print(io, "<py ", SubString(s, 2))
            else
                print(io, "<py ", s, ">")
            end
        end
    end
end

function Base.show(io::IO, ::MIME"text/plain", o::Py)
    hasprefix = get(io, :typeinfo, Any) != Py
    if pyisnull(o)
        if hasprefix
            printstyled(io, "Python NULL", bold=true)
        else
            print(io, "NULL")
        end
        return
    elseif pyisnone(o)
        if hasprefix
            printstyled(io, "Python None", bold=true)
        else
            print(io, "None")
        end
        return
    end
    h, w = displaysize(io)
    compact = get(io, :compact, false)
    str = pyrepr(String, o)
    multiline = '\n' in str
    prefix = hasprefix ? compact ? "Py:$(multiline ? '\n' : ' ')" : "Python $(pytype(o).__name__):$(multiline ? '\n' : ' ')" : ""
    printstyled(io, prefix, bold=true)
    if get(io, :limit, true)
        h, w = displaysize(io)
        h = max(h-3, 5) # use 3 fewer lines to allow for the prompt, but always allow at least 5 lines
        if multiline
            h -= 1 # for the prefix
            lines = split(str, '\n')
            function printlines(io, lines, w)
                for (i, line) in enumerate(lines)
                    if i > 1
                        println(io)
                    end
                    if length(line) > w
                        print(io, line[1:nextind(line, 0, w-1)], '…')
                    else
                        print(io, line)
                    end
                end
            end
            if length(lines) ≤ h
                printlines(io, lines, w)
            else
                h0 = cld(h-1, 2)
                h1 = h-1-h0
                i0 = h0
                i1 = length(lines) - h1 + 1
                # this indent computation tries to center the "more lines" message near the
                # middle of the span of non-whitespace in lines
                indent = 0
                for (c0, c1) in zip(lines[i0], lines[i1])
                    if c0 == c1 && isspace(c0)
                        indent += 1
                    else
                        break
                    end
                end
                maxlen = min(w, max(length(lines[i0]), length(lines[i1])))
                msg = "... $(length(lines)-h0-h1) more lines ..."
                indent = max(0, fld(maxlen + indent - length(msg), 2))
                printlines(io, lines[1:h0], w)
                println(io)
                printstyled(io, " "^indent, msg, color=:light_black)
                println(io)
                printlines(io, lines[end-h1+1:end], w)
            end
        else
            maxlen = h*w - length(prefix)
            if length(str) ≤ maxlen
                print(io, str)
            else
                h0 = cld(h-1, 2)
                i0 = nextind(str, 0, h0*w-length(prefix))
                h1 = h-1-h0
                i1 = prevind(str, ncodeunits(str)+1, h1*w)
                println(io, str[1:i0])
                msg = "... $(length(str[nextind(str,i0):prevind(str,i1)])) more chars ..."
                printstyled(io, " "^max(0, fld(w-length(msg), 2)), msg, color=:light_black)
                println(io)
                print(io, str[i1:end])
            end
        end
    else
        print(io, str)
    end
end

Base.show(io::IO, mime::MIME, o::Py) = pyshow(io, mime, o)
Base.show(io::IO, mime::MIME"text/csv", o::Py) = pyshow(io, mime, o)
Base.show(io::IO, mime::MIME"text/tab-separated-values", o::Py) = pyshow(io, mime, o)

Base.showable(::MIME"text/plain", ::Py) = true
Base.showable(mime::MIME, o::Py) = pyshowable(mime, o)

Base.getproperty(x::Py, k::Symbol) = pygetattr(x, string(k))
Base.getproperty(x::Py, k::String) = pygetattr(x, k)

Base.hasproperty(x::Py, k::Symbol) = pyhasattr(x, string(k))
Base.hasproperty(x::Py, k::String) = pyhasattr(x, k)

Base.setproperty!(x::Py, k::Symbol, v) = pysetattr(x, string(k), v)
Base.setproperty!(x::Py, k::String, v) = pysetattr(x, k, v)

function Base.propertynames(x::Py, private::Bool=false)
    # this follows the logic of rlcompleter.py
    function classmembers(c)
        r = pydir(c)
        if pyhasattr(c, "__bases__")
            for b in c.__bases__
                r = pyiadd(r, classmembers(b))
            end
        end
        return r
    end
    words = pyset(pydir(x))
    words.discard("__builtins__")
    if pyhasattr(x, "__class__")
        words.add("__class__")
        words.update(classmembers(x.__class__))
    end
    words = map(pystr_asstring, words)
    # private || filter!(w->!startswith(w, "_"), words)
    map(Symbol, words)
end

Base.Bool(x::Py) = pytruth(x)

Base.length(x::Py) = pylen(x)

Base.getindex(x::Py, i) = pygetitem(x, i)
Base.getindex(x::Py, i...) = pygetitem(x, i)

Base.setindex!(x::Py, v, i) = (pysetitem(x, i, v); x)
Base.setindex!(x::Py, v, i...) = (pysetitem(x, i, v); x)

Base.delete!(x::Py, i) = (pydelitem(x, i); x)

Base.haskey(x::Py, i) = pyhasitem(x, i)

Base.get(x::Py, i, d) = pygetitem(x, i, d)

function Base.get(f::Base.Callable, x::Py, i)
    v = pygetitem(x, i, nothing)
    v === nothing ? f() : v
end

Base.get!(x::Py, i, d) = get(x, i) do
    pysetitem(x, i, d)
    pygetitem(x, i)
end

Base.get!(f::Base.Callable, x::Py, i) = get(x, i) do
    pysetitem(x, i, f())
    pygetitem(x, i)
end

Base.eltype(::Type{Py}) = Py

Base.IteratorSize(::Type{Py}) = Base.SizeUnknown()

function Base.iterate(x::Py, it::Py=pyiter(x))
    v = unsafe_pynext(it)
    if pyisnull(v)
        pydel!(it)
        nothing
    else
        (v, it)
    end
end

Base.in(v, x::Py) = pycontains(x, v)

Base.hash(x::Py, h::UInt) = reinterpret(UInt, Int(pyhash(x))) - 3h

(f::Py)(args...; kwargs...) = pycall(f, args...; kwargs...)

# comparisons
Base.:(==)(x::Py, y::Py) = pyeq(x, y)
Base.:(!=)(x::Py, y::Py) = pyne(x, y)
Base.:(<=)(x::Py, y::Py) = pyle(x, y)
Base.:(< )(x::Py, y::Py) = pylt(x, y)
Base.:(>=)(x::Py, y::Py) = pyge(x, y)
Base.:(> )(x::Py, y::Py) = pygt(x, y)
Base.isless(x::Py, y::Py) = pylt(Bool, x, y)
Base.isequal(x::Py, y::Py) = pyeq(Bool, x, y)

# we also allow comparison with numbers
Base.:(==)(x::Py, y::Number) = pyeq(x, y)
Base.:(!=)(x::Py, y::Number) = pyne(x, y)
Base.:(<=)(x::Py, y::Number) = pyle(x, y)
Base.:(< )(x::Py, y::Number) = pylt(x, y)
Base.:(>=)(x::Py, y::Number) = pyge(x, y)
Base.:(> )(x::Py, y::Number) = pygt(x, y)
Base.isless(x::Py, y::Number) = pylt(Bool, x, y)
Base.isequal(x::Py, y::Number) = pyeq(Bool, x, y)

Base.:(==)(x::Number, y::Py) = pyeq(x, y)
Base.:(!=)(x::Number, y::Py) = pyne(x, y)
Base.:(<=)(x::Number, y::Py) = pyle(x, y)
Base.:(< )(x::Number, y::Py) = pylt(x, y)
Base.:(>=)(x::Number, y::Py) = pyge(x, y)
Base.:(> )(x::Number, y::Py) = pygt(x, y)
Base.isless(x::Number, y::Py) = pylt(Bool, x, y)
Base.isequal(x::Number, y::Py) = pyeq(Bool, x, y)

Base.zero(::Type{Py}) = pyint(0)
Base.one(::Type{Py}) = pyint(1)

# unary arithmetic
Base.:(+)(x::Py) = pypos(x)
Base.:(-)(x::Py) = pyneg(x)
Base.abs(x::Py) = pyabs(x)
Base.:(~)(x::Py) = pyinv(x)

# binary arithmetic
Base.:(+)(x::Py, y::Py) = pyadd(x, y)
Base.:(-)(x::Py, y::Py) = pysub(x, y)
Base.:(*)(x::Py, y::Py) = pymul(x, y)
# Base.:(+)(x::Py, y::Py) = pymatmul(x, y)
Base.div(x::Py, y::Py) = pyfloordiv(x, y)
Base.:(/)(x::Py, y::Py) = pytruediv(x, y)
Base.rem(x::Py, y::Py) = pymod(x, y)
# Base.:(+)(x::Py, y::Py) = pydivmod(x, y)
Base.:(<<)(x::Py, y::Py) = pylshift(x, y)
Base.:(>>)(x::Py, y::Py) = pyrshift(x, y)
Base.:(&)(x::Py, y::Py) = pyand(x, y)
Base.xor(x::Py, y::Py) = pyxor(x, y)
Base.:(|)(x::Py, y::Py) = pyor(x, y)
Base.:(^)(x::Py, y::Py) = pypow(x, y)

# also allow binary arithmetic with numbers
Base.:(+)(x::Number, y::Py) = pyadd(x, y)
Base.:(-)(x::Number, y::Py) = pysub(x, y)
Base.:(*)(x::Number, y::Py) = pymul(x, y)
# Base.:(+)(x::Number, y::Py) = pymatmul(x, y)
Base.div(x::Number, y::Py) = pyfloordiv(x, y)
Base.:(/)(x::Number, y::Py) = pytruediv(x, y)
Base.rem(x::Number, y::Py) = pymod(x, y)
# Base.:(+)(x::Number, y::Py) = pydivmod(x, y)
Base.:(<<)(x::Number, y::Py) = pylshift(x, y)
Base.:(>>)(x::Number, y::Py) = pyrshift(x, y)
Base.:(&)(x::Number, y::Py) = pyand(x, y)
Base.xor(x::Number, y::Py) = pyxor(x, y)
Base.:(|)(x::Number, y::Py) = pyor(x, y)
Base.:(^)(x::Number, y::Py) = pypow(x, y)

Base.:(+)(x::Py, y::Number) = pyadd(x, y)
Base.:(-)(x::Py, y::Number) = pysub(x, y)
Base.:(*)(x::Py, y::Number) = pymul(x, y)
# Base.:(+)(x::Py, y::Number) = pymatmul(x, y)
Base.div(x::Py, y::Number) = pyfloordiv(x, y)
Base.:(/)(x::Py, y::Number) = pytruediv(x, y)
Base.rem(x::Py, y::Number) = pymod(x, y)
# Base.:(+)(x::Py, y::Number) = pydivmod(x, y)
Base.:(<<)(x::Py, y::Number) = pylshift(x, y)
Base.:(>>)(x::Py, y::Number) = pyrshift(x, y)
Base.:(&)(x::Py, y::Number) = pyand(x, y)
Base.xor(x::Py, y::Number) = pyxor(x, y)
Base.:(|)(x::Py, y::Number) = pyor(x, y)
Base.:(^)(x::Py, y::Number) = pypow(x, y)

Base.powermod(x::Py, y::Py, z::Py) = pypow(x, y, z)
Base.powermod(x::Number, y::Py, z::Py) = pypow(x, y, z)
Base.powermod(x::Py, y::Number, z::Py) = pypow(x, y, z)
Base.powermod(x::Py, y::Py, z::Number) = pypow(x, y, z)
Base.powermod(x::Number, y::Number, z::Py) = pypow(x, y, z)
Base.powermod(x::Number, y::Py, z::Number) = pypow(x, y, z)
Base.powermod(x::Py, y::Number, z::Number) = pypow(x, y, z)

# documentation
function Base.Docs.getdoc(x::Py, @nospecialize(sig))
    parts = []
    inspect = pyimport("inspect")
    # head line
    if pytruth(inspect.ismodule(x))
        desc = pyhasattr(x, "__path__") ? "package" : "module"
        name = "$(x.__name__)"
    elseif pytruth(inspect.isgetsetdescriptor(x))
        desc = "getset descriptor"
        name = "$(x.__objclass__.__name__).$(x.__name__)"
    elseif pytruth(inspect.ismemberdescriptor(x))
        desc = "member descriptor"
        name = "$(x.__objclass__.__name__).$(x.__name__)"
    elseif pytruth(inspect.isclass(x))
        desc = "class"
        name = "$(x.__name__)"
    elseif pytruth(inspect.isfunction(x)) || pytruth(inspect.isbuiltin(x))
        desc = "function"
        name = "$(x.__name__)"
    elseif pytruth(inspect.ismethod(x))
        desc = "method"
        name = "$(x.__name__)"
    else
        desc = "object of type"
        name = "$(pytype(x).__name__)"
    end
    push!(parts, Markdown.Paragraph(["Python $desc ", Markdown.Code(name), "."]))
    # docstring
    doc = pyimport("inspect").getdoc(x)
    if !pyisnone(doc)
        push!(parts, Markdown.Code("text", pystr_asstring(doc)))
    end
    return Markdown.MD(parts)
end
Base.Docs.doc(x::Py, sig::Type=Union{}) = Base.Docs.getdoc(x, sig)
Base.Docs.Binding(x::Py, k::Symbol) = getproperty(x, k)
