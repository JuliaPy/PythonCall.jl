incref(x::C.PyPtr) = (C.Py_IncRef(x); x)
decref(x::C.PyPtr) = (C.Py_DecRef(x); x)

ispy(x) = false
ispynull(x) = getptr(x) == C.PyNULL
export ispy, ispynull

"""
    Py(x)

Convert `x` to a Python object.
"""
mutable struct Py
    ptr :: C.PyPtr
    Py(::Val{:new}, ptr::C.PyPtr) = finalizer(new(ptr)) do x
        if C.CTX.is_initialized
            C.with_gil() do
                ptr = getptr(x)
                C.Py_DecRef(ptr)
            end
        end
    end
end
export Py

ispy(::Py) = true
getpy(x::Py) = x
getptr(x::Py) = getfield(x, :ptr)

setptr!(x::Py, ptr::C.PyPtr) = (setfield!(x, :ptr, ptr); x)

const PYNULL_CACHE = Py[]

function pynew()
    if isempty(PYNULL_CACHE)
        Py(Val(:new), C.PyNULL)
    else
        pop!(PYNULL_CACHE)
    end
end

function pynew(ptr::C.PyPtr)
    setptr!(pynew(), ptr)
end

function pycopy!(dst, src)
    # assumes dst is NULL
    setptr!(dst, incref(getptr(src)))
end

function pydel!(x::Py)
    ptr = getptr(x)
    if ptr != C.PyNULL
        C.Py_DecRef(ptr)
    end
    pystolen!(x)
end

function pystolen!(x::Py)
    setptr!(x, C.PyNULL)
    push!(PYNULL_CACHE, x)
end

export pynull, pynew, pydel!, pystolen!

macro autopy(args...)
    vs = args[1:end-1]
    ts = [Symbol(v, "_") for v in vs]
    body = args[end]
    ans = gensym("ans")
    esc(quote
        $([:($t = $ispy($v) ? $v : $Py($v)) for (t, v) in zip(ts, vs)]...)
        $ans = $body
        $([:($ispy($v) || $pydel!($t)) for (t, v) in zip(ts, vs)]...)
        $ans
    end)
end

Py(x::Py) = pynew(incref(getptr(x))) # copy, because Py must always return a new object
Py(x::Nothing) = Py(pybuiltins.None)
Py(x::Bool) = pybool(x)
Py(x::Union{String, SubString{String}, Char}) = pystr(x)
Py(x::Base.CodeUnits{UInt8, String}) = pybytes(x)
Py(x::Base.CodeUnits{UInt8, SubString{String}}) = pybytes(x)
Py(x::Tuple) = pytuple_fromiter(x)
Py(x::Pair) = pytuple_fromiter(x)
Py(x::Union{Int8,Int16,Int32,Int64,Int128,UInt8,UInt16,UInt32,UInt64,UInt128,BigInt}) = pyint(x)
Py(x::Union{Float16,Float32,Float64}) = pyfloat(x)
Py(x::Complex{<:Union{Float16,Float32,Float64}}) = pycomplex(x)
Py(x::AbstractRange{<:Union{Int8,Int16,Int32,Int64,Int128,UInt8,UInt16,UInt32,UInt64,UInt128,BigInt}}) = pyrange_fromrange(x)
Py(x) = ispy(x) ? Py(getpy(x)) : error("cannot convert this $(typeof(x)) to a Python object")

Base.string(x::Py) = ispynull(x) ? "<py NULL>" : pystr(String, x)
Base.print(io::IO, x::Py) = print(io, string(x))

function Base.repr(x::Py)
    if getptr(x) == C.PyNULL
        return "<py NULL>"
    else
        s = pyrepr(String, x)
        if startswith(s, "<") && endswith(s, ">")
            return "<py $(SubString(s, 2))"
        else
            return "<py $s>"
        end
    end
end

function Base.show(io::IO, x::Py)
    if get(io, :typeinfo, Any) == Py
        if getptr(x) == C.PyNULL
            print(io, "NULL")
        else
            print(io, pyrepr(String, x))
        end
    else
        print(io, repr(x))
    end
end

Base.show(io::IO, mime::MIME, o::Py) = py_mime_show(io, mime, o)
Base.show(io::IO, mime::MIME"text/plain", o::Py) = show(io, o)
Base.show(io::IO, mime::MIME"text/csv", o::Py) = py_mime_show(io, mime, o)
Base.show(io::IO, mime::MIME"text/tab-separated-values", o::Py) = py_mime_show(io, mime, o)
Base.showable(mime::MIME, o::Py) = py_mime_showable(mime, o)

Base.getproperty(x::Py, k::Symbol) = pygetattr(x, string(k))

Base.setproperty!(x::Py, k::Symbol, v) = pysetattr(x, string(k), v)

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
    private || filter!(w->!startswith(w, "_"), words)
    map(Symbol, words)
end

Base.Bool(x::Py) = pytruth(x)

Base.length(x::Py) = pylen(x)

Base.getindex(x, i) = pygetitem(x, i)
Base.getindex(x, i...) = pygetitem(x, i)

Base.setindex!(x::Py, v, i) = (pysetitem(x, i, v); x)
Base.setindex!(x::Py, v, i...) = (pysetitem(x, i, v); x)

Base.delete!(x::Py, i) = (pydelitem(x, i); x)

Base.eltype(::Type{Py}) = Py

Base.IteratorSize(::Type{Py}) = Base.SizeUnknown()

function Base.iterate(x::Py, it::Py=pyiter(x))
    v = pynext(it)
    if ispynull(v)
        pydel!(it)
        nothing
    else
        (v, it)
    end
end

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
