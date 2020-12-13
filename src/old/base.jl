function Base.show(io::IO, o::PyObject)
    s = pyrepr(String, o)
    if get(io, :typeinfo, Any) == typeof(o)
        print(io, s)
    elseif startswith(s, "<") && endswith(s, ">")
        print(io, "<py ", s[2:end])
    else
        print(io, "<py ", s, ">")
    end
end

Base.print(io::IO, o::PyObject) = print(io, pystr(String, o))

function Base.show(io::IO, ::MIME"text/plain", o::PyObject)
    h, w = displaysize(io)
    h -= 3
    x = try
        pystr_asjuliastring(pypprintmodule.pformat(o, width=w))
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
                print(io, "\n ... [skipping $(h3-h2-1) lines] ...")
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

Base.hasproperty(o::PyObject, k::Union{Symbol,AbstractString}) = pyhasattr(o, k)

Base.getproperty(o::PyObject, k::Symbol) =
    if k == :jl!
        (T=Any) -> pyconvert(T, o)
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
    elseif k == :jl!set
        (args...) -> PySet{args...}(o)
    elseif k == :jl!buffer
        () -> PyBuffer(o)
    elseif k == :jl!array
        (args...) -> PyArray{args...}(o)
    elseif k == :jl!vector
        (args...) -> PyVector{args...}(o)
    elseif k == :jl!matrix
        (args...) -> PyMatrix{args...}(o)
    elseif k == :jl!pandasdf
        (; opts...) -> PyPandasDataFrame(o; opts...)
    elseif k == :jl!io
        (; opts...) -> PyIO(o; opts...)
    else
        pygetattr(o, k)
    end
Base.getproperty(o::PyObject, k::AbstractString) =
    startswith(k, "jl!") ? getproperty(o, Symbol(k)) : pygetattr(o, k)

Base.setproperty!(o::PyObject, k::Symbol, v) = (pysetattr(o, k, v); o)
Base.setproperty!(o::PyObject, k::AbstractString, v) = (pysetattr(o, k, v); o)

function Base.propertynames(o::PyObject)
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


Base.getindex(o::PyObject, k) = pygetitem(o, k)
Base.getindex(o::PyObject, k...) = pygetitem(o, k)

Base.setindex!(o::PyObject, v, k) = (pysetitem(o, k, v); o)
Base.setindex!(o::PyObject, v, k...) = (pysetitem(o, k, v); o)

Base.delete!(o::PyObject, k) = (pydelitem(o, k); o)

Base.length(o::PyObject) = Int(pylen(o))

Base.in(v, o::PyObject) = pycontains(o, v)

(f::PyObject)(args...; kwargs...) = pycall(f, args...; kwargs...)

Base.eltype(::Type{T}) where {T<:PyObject} = PyObject
Base.IteratorSize(::Type{T}) where {T<:PyObject} = Base.SizeUnknown()
function Base.iterate(o::PyObject, it=pyiter(o))
    ptr = C.PyIter_Next(it)
    if ptr == C_NULL
        pyerrcheck()
        nothing
    else
        (pynewobject(ptr), it)
    end
end

# comparisons
Base.:(==)(o1::PyObject, o2::PyObject) = pyeq(Bool, o1, o2)
Base.:(!=)(o1::PyObject, o2::PyObject) = pyne(Bool, o1, o2)
Base.:(< )(o1::PyObject, o2::PyObject) = pylt(Bool, o1, o2)
Base.:(<=)(o1::PyObject, o2::PyObject) = pyle(Bool, o1, o2)
Base.:(> )(o1::PyObject, o2::PyObject) = pygt(Bool, o1, o2)
Base.:(>=)(o1::PyObject, o2::PyObject) = pyge(Bool, o1, o2)
Base.isequal(o1::PyObject, o2::PyObject) = pyeq(Bool, o1, o2)
Base.isless(o1::PyObject, o2::PyObject) = pylt(Bool, o1, o2)

# unary arithmetic
Base.:(-)(o::PyObject) = pyneg(o)
Base.:(+)(o::PyObject) = pypos(o)
Base.abs(o::PyObject) = pyabs(o)
Base.:(~)(o::PyObject) = pyinv(o)

# binary arithmetic
Base.:(+)(o1::PyObject, o2::PyObject) = pyadd(o1, o2)
Base.:(-)(o1::PyObject, o2::PyObject) = pysub(o1, o2)
Base.:(*)(o1::PyObject, o2::PyObject) = pymul(o1, o2)
Base.:(/)(o1::PyObject, o2::PyObject) = pytruediv(o1, o2)
Base.:fld(o1::PyObject, o2::PyObject) = pyfloordiv(o1, o2)
Base.:mod(o1::PyObject, o2::PyObject) = pymod(o1, o2)
Base.:(^)(o1::PyObject, o2::PyObject) = pypow(o1, o2)
Base.:(<<)(o1::PyObject, o2::PyObject) = pylshift(o1, o2)
Base.:(>>)(o1::PyObject, o2::PyObject) = pyrshift(o1, o2)
Base.:(&)(o1::PyObject, o2::PyObject) = pyand(o1, o2)
Base.:xor(o1::PyObject, o2::PyObject) = pyxor(o1, o2)
Base.:(|)(o1::PyObject, o2::PyObject) = pyor(o1, o2)

Base.:(+)(o1::PyObject, o2::Number) = pyadd(o1, o2)
Base.:(-)(o1::PyObject, o2::Number) = pysub(o1, o2)
Base.:(*)(o1::PyObject, o2::Number) = pymul(o1, o2)
Base.:(/)(o1::PyObject, o2::Number) = pytruediv(o1, o2)
Base.:fld(o1::PyObject, o2::Number) = pyfloordiv(o1, o2)
Base.:mod(o1::PyObject, o2::Number) = pymod(o1, o2)
Base.:(^)(o1::PyObject, o2::Number) = pypow(o1, o2)
Base.:(<<)(o1::PyObject, o2::Number) = pylshift(o1, o2)
Base.:(>>)(o1::PyObject, o2::Number) = pyrshift(o1, o2)
Base.:(&)(o1::PyObject, o2::Number) = pyand(o1, o2)
Base.:xor(o1::PyObject, o2::Number) = pyxor(o1, o2)
Base.:(|)(o1::PyObject, o2::Number) = pyor(o1, o2)

Base.:(+)(o1::Number, o2::PyObject) = pyadd(o1, o2) # Defining +(::Any, ::PyObject) like this hangs Julia v1.5.2-v1.5.3 (at least) during precompilation
Base.:(-)(o1::Number, o2::PyObject) = pysub(o1, o2)
Base.:(*)(o1::Number, o2::PyObject) = pymul(o1, o2)
Base.:(/)(o1::Number, o2::PyObject) = pytruediv(o1, o2)
Base.:fld(o1::Number, o2::PyObject) = pyfloordiv(o1, o2)
Base.:mod(o1::Number, o2::PyObject) = pymod(o1, o2)
# Base.:(^)(o1::Number, o2::PyObject) = pypow(o1, o2)
# Base.:(<<)(o1::Number, o2::PyObject) = pylshift(o1, o2)
# Base.:(>>)(o1::Number, o2::PyObject) = pyrshift(o1, o2)
Base.:(&)(o1::Number, o2::PyObject) = pyand(o1, o2)
Base.:xor(o1::Number, o2::PyObject) = pyxor(o1, o2)
Base.:(|)(o1::Number, o2::PyObject) = pyor(o1, o2)

# ternary arithmetic
Base.powermod(o1::PyObject, o2::Union{PyObject,Number}, o3::Union{PyObject,Number}) = pypow(o1, o2, o3)

Base.zero(::Type{PyObject}) = pyint(0)
Base.one(::Type{PyObject}) = pyint(1)

function Base.Docs.getdoc(o::PyObject)
    function tryget(f, g=identity)
        a = try
            f()
        catch
            return nothing
        end
        pyisnone(a) ? nothing : g(a)
    end
    # function name(o)
    #     n = tryget(()->o.__name__)
    #     n === nothing && return nothing
    #     m = tryget(()->o.__module__)
    #     (m === nothing || string(m) == "builtins") ? string(n) : string(m, ".", n)
    # end
    getname(o) = tryget(()->o.__name__, string)
    getdoc(o) = tryget(()->o.__doc__, string ∘ ins.cleandoc)

    docs = []

    # Short description
    ins = pyimport("inspect")
    desc, name = if ins.ismodule(o).jl!b
        (pyhasattr(o, "__path__") ? "package" : "module"), gname(o)
    elseif ins.isgetsetdescriptor(o).jl!b
        ("getset descriptor", "$(getname(o.__objclass__)).$(o.__name__)")
    elseif ins.ismemberdescriptor(o).jl!b
        ("member descriptor", "$(getname(o.__objclass__)).$(o.__name__)")
    elseif ins.isclass(o).jl!b
        ("class", getname(o))
    elseif ins.isfunction(o).jl!b || ins.isbuiltin(o).jl!b
        ("function", getname(o))
    elseif ins.ismethod(o).jl!b
        ("method", getname(o))
    else
        ("object of type", getname(pytype(o)))
    end
    push!(docs, Markdown.Paragraph(["Python ", desc, " ", Markdown.Code(name), "."]))

    if ins.isroutine(o).jl!b || ins.isclass(o).jl!b
        try
            push!(docs, Markdown.Code("python", "$(o.__name__)$(ins.signature(o))"))
        catch
        end
    end

    # Maybe document the class instead
    doc = getdoc(o)
    if doc === nothing && !ins.ismodule(o).jl!b && !ins.isclass(o).jl!b && !ins.isroutine(o).jl!b && !ins.isdatadescriptor(o).jl!b
        o = pyhasattr(o, "__origin__") ? o.__origin__ : pytype(o)
        doc = getdoc(o)
    end
    doc === nothing || push!(docs, Markdown.Paragraph([Markdown.Text(doc)]))

    # Done
    Markdown.MD(docs)
end
Base.Docs.Binding(o::PyObject, k::Symbol) = getproperty(o, k)

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
                if pyisnone(x)
                    false
                else
                    pyconvert($T, x)
                    true
                end
            catch
                false
            end
        end
    end
end
