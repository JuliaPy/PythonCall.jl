function Base.show(io::IO, o::AbstractPyObject)
    s = pyrepr(String, o)
    if get(io, :typeinfo, Any) == typeof(o)
        print(io, s)
    elseif startswith(s, "<") && endswith(s, ">")
        print(io, "<py ", s[2:end])
    else
        print(io, "<py ", s, ">")
    end
end

Base.print(io::IO, o::AbstractPyObject) = print(io, pystr(String, o))

function Base.show(io::IO, ::MIME"text/plain", o::AbstractPyObject)
    h, w = displaysize(io)
    w -= 1
    h -= 5
    x = try
        pystr_asjuliastring(pypprintmodule.pformat(o, width=w))
    catch
        pyrepr(String, o)
    end
    if get(io, :limit, true)
        if '\n' ∈ x
            # multiple lines
            xs = split(x, '\n')
        elseif length(x) ≤ w-4
            # one short line
            print(io, "py: ", x)
            return
        else
            # one long line
            xs = SubString{String}[]
            i = 1
            for _ in 2:cld(length(x), w)
                j = nextind(x, i, w-1)
                push!(xs, SubString(x, i, j))
                i = nextind(x, j, 1)
            end
            push!(xs, SubString(x, i, ncodeunits(x)))
        end
        printlines(xs) =
            for x in xs
                if length(x) ≤ w
                    print(io, '\n', x)
                else
                    print(io, '\n', x[1:nextind(x, 0, w-1)], '…')
                end
            end
        print(io, "py:")
        if length(xs) ≤ h
            printlines(xs)
        else
            printlines(xs[1:cld(h-1,2)])
            print(io, "\n ...")
            printlines(xs[end-(h-cld(h-1,2))+1:end])
        end
    else
        print(io, x)
    end
end

Base.hasproperty(o::AbstractPyObject, k::Union{Symbol,AbstractString}) = pyhasattr(o, k)

Base.getproperty(o::AbstractPyObject, k::Symbol) =
    if k == :jl!
        T -> pyconvert(T, o)
    elseif k == :jl!i
        pyconvert(Int, o)
    elseif k == :jl!u
        pyconvert(UInt, o)
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
    elseif k == :jl!list
        (args...) -> PyList{args...}(o)
    elseif k == :jl!dict
        (args...) -> PyDict{args...}(o)
    elseif k == :jl!buffer
        () -> PyBuffer(o)
    elseif k == :jl!array
        (args...) -> PyArray{args...}(o)
    elseif k == :jl!pandasdf
        (; opts...) -> PyPandasDataFrame(o; opts...)
    elseif k == :jl!io
        (; opts...) -> PyIO(o; opts...)
    else
        pygetattr(o, k)
    end
Base.getproperty(o::AbstractPyObject, k::AbstractString) =
    startswith(k, "jl!") ? getproperty(o, Symbol(k)) : pygetattr(o, k)

Base.setproperty!(o::AbstractPyObject, k::Symbol, v) = (pysetattr(o, k, v); o)
Base.setproperty!(o::AbstractPyObject, k::AbstractString, v) = (pysetattr(o, k, v); o)

function Base.propertynames(o::AbstractPyObject)
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
    words = pyset(pydir(o))
    words.discard("__builtins__")
    if pyhasattr(o, "__class__")
        words.add("__class__")
        words.update(classmembers(o.__class__))
    end
    r = [Symbol(pystr(String, x)) for x in words]
    return r
end


Base.getindex(o::AbstractPyObject, k) = pygetitem(o, k)
Base.getindex(o::AbstractPyObject, k...) = pygetitem(o, k)

Base.setindex!(o::AbstractPyObject, v, k) = (pysetitem(o, k, v); o)
Base.setindex!(o::AbstractPyObject, v, k...) = (pysetitem(o, k, v); o)

Base.delete!(o::AbstractPyObject, k) = (pydelitem(o, k); o)

Base.length(o::AbstractPyObject) = Int(pylen(o))

Base.in(v, o::AbstractPyObject) = pycontains(o, v)

(f::AbstractPyObject)(args...; kwargs...) = pycall(f, args...; kwargs...)

Base.eltype(::Type{T}) where {T<:AbstractPyObject} = PyObject
Base.IteratorSize(::Type{T}) where {T<:AbstractPyObject} = Base.SizeUnknown()
function Base.iterate(o::AbstractPyObject, it=pyiter(o))
    ptr = C.PyIter_Next(it)
    if ptr == C_NULL
        pyerrcheck()
        nothing
    else
        (pynewobject(ptr), it)
    end
end

# comparisons
# TODO: allow comparison with non-python objects?
Base.:(==)(o1::AbstractPyObject, o2::AbstractPyObject) = pyeq(Bool, o1, o2)
Base.:(!=)(o1::AbstractPyObject, o2::AbstractPyObject) = pyne(Bool, o1, o2)
Base.:(< )(o1::AbstractPyObject, o2::AbstractPyObject) = pylt(Bool, o1, o2)
Base.:(<=)(o1::AbstractPyObject, o2::AbstractPyObject) = pyle(Bool, o1, o2)
Base.:(> )(o1::AbstractPyObject, o2::AbstractPyObject) = pygt(Bool, o1, o2)
Base.:(>=)(o1::AbstractPyObject, o2::AbstractPyObject) = pyge(Bool, o1, o2)
Base.isequal(o1::AbstractPyObject, o2::AbstractPyObject) = pyeq(Bool, o1, o2)
Base.isless(o1::AbstractPyObject, o2::AbstractPyObject) = pylt(Bool, o1, o2)

# unary arithmetic
Base.:(-)(o::AbstractPyObject) = pyneg(o)
Base.:(+)(o::AbstractPyObject) = pypos(o)
Base.abs(o::AbstractPyObject) = pyabs(o)
Base.:(~)(o::AbstractPyObject) = pyinv(o)

# binary arithmetic
Base.:(+)(o1::AbstractPyObject, o2::AbstractPyObject) = pyadd(o1, o2)
Base.:(-)(o1::AbstractPyObject, o2::AbstractPyObject) = pysub(o1, o2)
Base.:(*)(o1::AbstractPyObject, o2::AbstractPyObject) = pymul(o1, o2)
Base.:(/)(o1::AbstractPyObject, o2::AbstractPyObject) = pytruediv(o1, o2)
Base.:fld(o1::AbstractPyObject, o2::AbstractPyObject) = pyfloordiv(o1, o2)
Base.:mod(o1::AbstractPyObject, o2::AbstractPyObject) = pymod(o1, o2)
Base.:(^)(o1::AbstractPyObject, o2::AbstractPyObject) = pypow(o1, o2)
Base.:(<<)(o1::AbstractPyObject, o2::AbstractPyObject) = pylshift(o1, o2)
Base.:(>>)(o1::AbstractPyObject, o2::AbstractPyObject) = pyrshift(o1, o2)
Base.:(&)(o1::AbstractPyObject, o2::AbstractPyObject) = pyand(o1, o2)
Base.:xor(o1::AbstractPyObject, o2::AbstractPyObject) = pyxor(o1, o2)
Base.:(|)(o1::AbstractPyObject, o2::AbstractPyObject) = pyor(o1, o2)

Base.:(+)(o1::AbstractPyObject, o2::Number) = pyadd(o1, o2)
Base.:(-)(o1::AbstractPyObject, o2::Number) = pysub(o1, o2)
Base.:(*)(o1::AbstractPyObject, o2::Number) = pymul(o1, o2)
Base.:(/)(o1::AbstractPyObject, o2::Number) = pytruediv(o1, o2)
Base.:fld(o1::AbstractPyObject, o2::Number) = pyfloordiv(o1, o2)
Base.:mod(o1::AbstractPyObject, o2::Number) = pymod(o1, o2)
Base.:(^)(o1::AbstractPyObject, o2::Number) = pypow(o1, o2)
Base.:(<<)(o1::AbstractPyObject, o2::Number) = pylshift(o1, o2)
Base.:(>>)(o1::AbstractPyObject, o2::Number) = pyrshift(o1, o2)
Base.:(&)(o1::AbstractPyObject, o2::Number) = pyand(o1, o2)
Base.:xor(o1::AbstractPyObject, o2::Number) = pyxor(o1, o2)
Base.:(|)(o1::AbstractPyObject, o2::Number) = pyor(o1, o2)

Base.:(+)(o1::Number, o2::AbstractPyObject) = pyadd(o1, o2) # Defining +(::Any, ::AbstractPyObject) like this hangs Julia v1.5.2-v1.5.3 (at least) during precompilation
Base.:(-)(o1::Number, o2::AbstractPyObject) = pysub(o1, o2)
Base.:(*)(o1::Number, o2::AbstractPyObject) = pymul(o1, o2)
Base.:(/)(o1::Number, o2::AbstractPyObject) = pytruediv(o1, o2)
Base.:fld(o1::Number, o2::AbstractPyObject) = pyfloordiv(o1, o2)
Base.:mod(o1::Number, o2::AbstractPyObject) = pymod(o1, o2)
# Base.:(^)(o1::Number, o2::AbstractPyObject) = pypow(o1, o2)
# Base.:(<<)(o1::Number, o2::AbstractPyObject) = pylshift(o1, o2)
# Base.:(>>)(o1::Number, o2::AbstractPyObject) = pyrshift(o1, o2)
Base.:(&)(o1::Number, o2::AbstractPyObject) = pyand(o1, o2)
Base.:xor(o1::Number, o2::AbstractPyObject) = pyxor(o1, o2)
Base.:(|)(o1::Number, o2::AbstractPyObject) = pyor(o1, o2)

# ternary arithmetic
Base.powermod(o1::AbstractPyObject, o2::Union{AbstractPyObject,Number}, o3::Union{AbstractPyObject,Number}) = pypow(o1, o2, o3)

Base.zero(::Type{PyObject}) = pyint(0)
Base.one(::Type{PyObject}) = pyint(1)

function Base.Docs.getdoc(o::AbstractPyObject)
    docs = []
    function typename(t)
        n = string(t.__name__)
        m = string(t.__module__)
        m == "builtins" ? n : "$m.$n"
    end

    # Say what it is
    if pyistype(o)
        push!(docs, Markdown.Paragraph(["Python class ", Markdown.Code("$(typename(o))($(join([typename(t) for t in o.__bases__], ", ")))"), "."]))
    elseif pyismodule(o)
        push!(docs, Markdown.Paragraph(["Python module ", Markdown.Code("$(o.__name__)"), "."]))
    else
        push!(docs, Markdown.Paragraph(["Python object of type ", Markdown.Code("$(typename(pytype(o)))"), "."]))
    end

    # Print its docstring
    doc = try
        o.__doc__
    catch
        nothing
    end
    if doc !== nothing && !pyisnone(doc)
        push!(docs, Text(string(doc)))
    else
        # If that failed, print the docstring from its type
        tdoc = try
            pytype(o).__doc__
        catch
            nothing
        end
        if tdoc !== nothing && !pyisnone(tdoc)
            push!(docs, Text(string(tdoc)))
        end
    end
    Markdown.MD(docs)
end
Base.Docs.Binding(o::AbstractPyObject, k::Symbol) = getproperty(o, k)

for (mime, method) in ((MIME"text/html", "_repr_html_"),
                       (MIME"text/markdown", "_repr_markdown_"),
                       (MIME"text/json", "_repr_json_"),
                       (MIME"application/javascript", "_repr_javascript_"),
                       (MIME"application/pdf", "_repr_pdf_"),
                       (MIME"image/jpeg", "_repr_jpeg_"),
                       (MIME"image/png", "_repr_png_"),
                       (MIME"image/svg+xml", "_repr_svg_"),
                       (MIME"text/latex", "_repr_latex_"))
    T = istextmime(mime()) ? String : Vector{UInt8}
    @eval begin
        function Base.show(io::IO, mime::$mime, o::PyObject)
            try
                x = pygetattr(o, $method)()
                pyisnone(x) || return write(io, pyconvert($T, x))
            catch
            end
            throw(MethodError(show, (io, mime, o)))
        end
        function Base.showable(::$mime, o::PyObject)
            try
                x = pygetattr(o, $method)()
                !pyisnone(x)
            catch
                false
            end
        end
    end
end
