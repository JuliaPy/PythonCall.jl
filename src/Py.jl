incref(x::C.PyPtr) = (C.Py_IncRef(x); x)
decref(x::C.PyPtr) = (C.Py_DecRef(x); x)

ispy(x) = false
ispynull(x) = getptr(x) == C.PyNULL
getptr(x) = getptr(getpy(x)::Py)
export ispy, ispynull

"""
    Py(x)

Convert `x` to a Python object.
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
getpy(x::Py) = x
getptr(x::Py) = getfield(x, :ptr)

setptr!(x::Py, ptr::C.PyPtr) = (setfield!(x, :ptr, ptr); x)

const PYNULL_CACHE = Py[]

pynew() =
    if isempty(PYNULL_CACHE)
        Py(Val(:new), C.PyNULL)
    else
        pop!(PYNULL_CACHE)
    end

const PyNULL = pynew()

pynew(ptr::C.PyPtr) = setptr!(pynew(), ptr)

# assumes dst is NULL
pycopy!(dst, src) = setptr!(dst, incref(getptr(src)))

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
    nothing
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
Py(x::Rational{<:Union{Int8,Int16,Int32,Int64,Int128,UInt8,UInt16,UInt32,UInt64,UInt128,BigInt}}) = pyfraction(x)
Py(x::Union{Float16,Float32,Float64}) = pyfloat(x)
Py(x::Complex{<:Union{Float16,Float32,Float64}}) = pycomplex(x)
Py(x::AbstractRange{<:Union{Int8,Int16,Int32,Int64,Int128,UInt8,UInt16,UInt32,UInt64,UInt128,BigInt}}) = pyrange_fromrange(x)
Py(x) = ispy(x) ? Py(getpy(x)) : pyjl(x)

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

function Base.show(io::IO, mime::MIME"text/plain", o::Py)
    if ispynull(o)
        printstyled(io, "Python NULL", bold=true)
        return
    elseif pyisnone(o)
        printstyled(io, "Python None", bold=true)
        return
    end
    h, w = displaysize(io)
    compact = get(io, :compact, false)
    hasprefix = get(io, :typeinfo, Any) != Py
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

Base.show(io::IO, mime::MIME, o::Py) = py_mime_show(io, mime, o)
Base.show(io::IO, mime::MIME"text/csv", o::Py) = py_mime_show(io, mime, o)
Base.show(io::IO, mime::MIME"text/tab-separated-values", o::Py) = py_mime_show(io, mime, o)
Base.showable(mime::MIME, o::Py) = py_mime_showable(mime, o)

Base.getproperty(x::Py, k::Symbol) = pygetattr(x, string(k))
Base.getproperty(x::Py, k::String) = pygetattr(x, k)

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

Base.in(v, x::Py) = pycontains(x, v)

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
