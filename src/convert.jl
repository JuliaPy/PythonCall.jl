function pyconvert(::Type{T}, o::AbstractPyObject) where {T}
    r = pytryconvert(T, o)
    r === PyConvertFail() ? error("cannot convert this Python `$(pytype(o).__name__)` to a Julia `$T`") : r
end
export pyconvert

function pytryconvert(::Type{T}, o::AbstractPyObject) where {T}
    # special cases
    if T == PyObject
        return PyObject(o)
    end

    # types
    for t in pytype(o).__mro__
        n = "$(t.__module__).$(t.__name__)"
        c = get(PYTRYCONVERT_TYPE_RULES, n, (T,o)->PyConvertFail())
        r = c(T, o) :: Union{T, PyConvertFail}
        r === PyConvertFail() || return r
    end

    # interfaces
    # TODO: buffer
    # TODO: IO
    # TODO: number?
    if pyisinstance(o, pyiterableabc)
        r = pyiterable_tryconvert(T, o) :: Union{T, PyConvertFail}
        r === PyConvertFail() || return r
    end

    # so that T=Any always succeeds
    if o isa T
        return o
    end

    # failure
    return PyConvertFail()
end
export pytryconvert

const PYTRYCONVERT_TYPE_RULES = Dict{String,Function}(
    "builtins.NoneType" => pynone_tryconvert,
    "builtins.bool" => pybool_tryconvert,
    "builtins.str" => pystr_tryconvert,
    "builtins.int" => pyint_tryconvert,
    "builtins.float" => pyfloat_tryconvert,
    "builtins.complex" => pycomplex_tryconvert,
    "builtins.Fraction" => pyfraction_tryconvert,
    "builtins.range" => pyrange_tryconvert,
    "builtins.tuple" => pytuple_tryconvert,
    "pandas.core.frame.DataFrame" => pypandasdataframe_tryconvert,
    # TODO: datetime, date, time
    # NOTE: we don't need to include standard containers here because we can access them via standard interfaces (Sequence, Mapping, Buffer, etc.)
)

Base.convert(::Type{T}, o::AbstractPyObject) where {T} = pyconvert(T, o)
Base.convert(::Type{Any}, o::AbstractPyObject) = o

### SPECIAL CONVERSIONS

@generated _eltype(o) = try eltype(o); catch; missing; end

@generated _keytype(o) = try keytype(o); catch; missing; end
_keytype(o::Base.RefValue) = Tuple{}
_keytype(o::NamedTuple) = Union{Symbol,Int}
_keytype(o::Tuple) = Int

@generated _valtype(o, k...) = try valtype(o); catch; missing; end
_valtype(o::NamedTuple, k::Int) = fieldtype(typeof(o), k)
_valtype(o::NamedTuple, k::Symbol) = fieldtype(typeof(o), k)
_valtype(o::Tuple, k::Int) = fieldtype(typeof(o), k)

hasmultiindex(o) = true
hasmultiindex(o::AbstractDict) = false
hasmultiindex(o::NamedTuple) = false
hasmultiindex(o::Tuple) = false

"""
    pyconvert_element(o, v::AbstractPyObject)

Convert `v` to be of the right type to be an element of `o`.
"""
pytryconvert_element(o, v) = pytryconvert(_eltype(o)===missing ? Any : _eltype(o), v)
pyconvert_element(args...) =
    let r = pytryconvert_element(args...)
        r === PyConvertFail() ? error("cannot convert this to an element") : r
    end

"""
    pytryconvert_indices(o, k::AbstractPyObject)

Convert `k` to be of the right type to be a tuple of indices for `o`.
"""
function pytryconvert_indices(o, k)
    if _keytype(o) !== missing
        i = pytryconvert(_keytype(o), k)
        i === PyConvertFail() ? PyConvertFail() : (i,)
    elseif hasmultiindex(o) && pyistuple(k)
        Tuple(pyconvert(Any, x) for x in k)
    else
        (pyconvert(Any, k),)
    end
end
pyconvert_indices(args...) =
    let r = pytryconvert_indices(args...)
        r === PyConvertFail() ? error("cannot convert this to indices") : r
    end

"""
    pyconvert_value(o, v::AbstractPyObject, k...)

Convert `v` to be of the right type to be a value of `o` at indices `k`.
"""
pytryconvert_value(o, v, k...) = pytryconvert(_valtype(o)===missing ? Any : _valtype(o), v)
pyconvert_value(args...) =
    let r = pytryconvert_value(args...)
        r === PyConvertFail() ? error("cannot convert this to a value") : r
    end

"""
    pyconvert_args(NamedTuple{argnames, argtypes}, args, kwargs=nothing; defaults=nothing, numpos=len(names), numposonly=0)

Parse the Python tuple `args` and optionally Python dict `kwargs` as function arguments, returning a `NamedTuple{names,types}`.

- `defaults` is a named tuple of default parameter values for optional arguments.
- `numpos` is the number of possibly-positional arguments.
- `numposonly` is the number of positional-only arguments.
"""
@generated function pyconvert_args(
    ::Type{NamedTuple{argnames,argtypes}},
    args::AbstractPyObject,
    kwargs::Union{AbstractPyObject,Nothing}=nothing;
    defaults::Union{NamedTuple,Nothing}=nothing,
    numposonly::Integer=0,
    numpos::Integer=length(argnames),
) where {argnames, argtypes}
    code = []
    argvars = []
    push!(code, quote
        0 ≤ numposonly ≤ numpos ≤ length(argnames) || error("require 0 ≤ numposonly ≤ numpos ≤ numargs")
        nargs = pylen(args)
        nargs ≤ numpos || pythrow(pytypeerror("at most $numpos positional arguments expected, got $nargs"))
    end)
    for (i,(argname,argtype)) in enumerate(zip(argnames,argtypes.parameters))
        argvar = gensym()
        push!(argvars, argvar)
        push!(code, quote
            $argvar =
                if $i ≤ numpos && $i ≤ nargs
                    kwargs !== nothing && $(string(argname)) in kwargs && pythrow(pytypeerror($("argument '$argname' got multiple values")))
                    let r = pytryconvert($argtype, args[$(i-1)])
                        r===PyConvertFail() ? pythrow(pytypeerror($("argument '$argname' has wrong type"))) : r
                    end
                elseif kwargs !== nothing && $i > numposonly && $(string(argname)) in kwargs
                    let r = pytryconvert($argtype, kwargs.pop($(string(argname))))
                        r===PyConvertFail() ? pythrow(pytypeerror($("argument '$argname' has wrong type"))) : r
                    end
                elseif defaults !== nothing && haskey(defaults, $(QuoteNode(argname)))
                    convert($argtype, defaults[$(QuoteNode(argname))])
                else
                    pythrow(pytypeerror($("argument '$argname' not given")))
                end
        end)
    end
    push!(code, quote
        kwargs !== nothing && pylen(kwargs) > 0 && pythrow(pytypeerror("invalid keyword arguments: $(join(kwargs, ", "))"))
        NamedTuple{$argnames, $argtypes}(($(argvars...),))
    end)
    ex = Expr(:block, code...)
    @show ex
end

macro pyconvert_args(spec, args, kwargs=nothing)
    spec isa Expr && spec.head == :tuple || @goto error
    argnames = []
    argtypes = []
    defaults = []
    numposonly = 0
    numpos = -1
    for argspec in spec.args
        if argspec === :*
            numpos = length(argnames)
            continue
        elseif argspec == :/
            numposonly = length(argnames)
            continue
        end
        if argspec isa Expr && argspec.head == :(=)
            argspec, dflt = argspec.args
        else
            dflt = nothing
        end
        if argspec isa Expr && argspec.head == :(::)
            argspec, tp = argspec.args
        else
            tp = PyObject
        end
        if argspec isa Symbol
            nm = argspec
        else
            @goto error
        end
        push!(argnames, nm)
        push!(argtypes, tp)
        dflt === nothing || push!(defaults, nm => dflt)
    end
    numpos < 0 && (numpos = length(argnames))
    return :(pyconvert_args(NamedTuple{($(map(QuoteNode, argnames)...),),Tuple{$(map(esc, argtypes)...)}}, $(esc(args)), $(esc(kwargs)), defaults=NamedTuple{($([QuoteNode(n) for (n,d) in defaults]...),)}(($([esc(d) for (n,d) in defaults]...),)), numpos=$numpos, numposonly=$numposonly))
    @label error
    error("invalid argument specification")
end
