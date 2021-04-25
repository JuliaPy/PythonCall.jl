_PyObject_New(o) = ccall(POINTERS._PyObject_New, PyPtr, (PyPtr,), o)
PyObject_ClearWeakRefs(o) = ccall(POINTERS.PyObject_ClearWeakRefs, Cvoid, (PyPtr,), o)
PyObject_HasAttrString(o, k) = ccall(POINTERS.PyObject_HasAttrString, Cint, (PyPtr, Cstring), o, k)
PyObject_HasAttr(o, k) = ccall(POINTERS.PyObject_HasAttr, Cint, (PyPtr, PyPtr), o, k)
PyObject_GetAttrString(o, k) = ccall(POINTERS.PyObject_GetAttrString, PyPtr, (PyPtr, Cstring), o, k)
PyObject_GetAttr(o, k) = ccall(POINTERS.PyObject_GetAttr, PyPtr, (PyPtr, PyPtr), o, k)
PyObject_GenericGetAttr(o, k) = ccall(POINTERS.PyObject_GenericGetAttr, PyPtr, (PyPtr, PyPtr), o, k)
PyObject_SetAttrString(o, k, v) = ccall(POINTERS.PyObject_SetAttrString, Cint, (PyPtr, Cstring, PyPtr), o, k, v)
PyObject_SetAttr(o, k, v) = ccall(POINTERS.PyObject_SetAttr, Cint, (PyPtr, PyPtr, PyPtr), o, k, v)
PyObject_GenericSetAttr(o, k, v) = ccall(POINTERS.PyObject_GenericSetAttr, Cint, (PyPtr, PyPtr, PyPtr), o, k, v)
PyObject_RichCompare(x, y, op) = ccall(POINTERS.PyObject_RichCompare, PyPtr, (PyPtr, PyPtr, Cint), x, y, op)
PyObject_RichCompareBool(x, y, op) = ccall(POINTERS.PyObject_RichCompareBool, Cint, (PyPtr, PyPtr, Cint), x, y, op)
PyObject_Repr(o) = ccall(POINTERS.PyObject_Repr, PyPtr, (PyPtr,), o)
PyObject_ASCII(o) = ccall(POINTERS.PyObject_ASCII, PyPtr, (PyPtr,), o)
PyObject_Str(o) = ccall(POINTERS.PyObject_Str, PyPtr, (PyPtr,), o)
PyObject_Bytes(o) = ccall(POINTERS.PyObject_Bytes, PyPtr, (PyPtr,), o)
PyObject_IsSubclass(o, t) = ccall(POINTERS.PyObject_IsSubclass, Cint, (PyPtr, PyPtr), o, t)
PyObject_IsInstance(t1, t2) = ccall(POINTERS.PyObject_IsInstance, Cint, (PyPtr, PyPtr), t1, t2)
PyObject_Hash(o) = ccall(POINTERS.PyObject_Hash, Py_hash_t, (PyPtr,), o)
PyObject_IsTrue(o) = ccall(POINTERS.PyObject_IsTrue, Cint, (PyPtr,), o)
PyObject_Length(o) = ccall(POINTERS.PyObject_Length, Py_ssize_t, (PyPtr,), o)
PyObject_GetItem(o, k) = ccall(POINTERS.PyObject_GetItem, PyPtr, (PyPtr, PyPtr), o, k)
PyObject_SetItem(o, k, v) = ccall(POINTERS.PyObject_SetItem, Cint, (PyPtr, PyPtr, PyPtr), o, k, v)
PyObject_DelItem(o, k) = ccall(POINTERS.PyObject_DelItem, Cint, (PyPtr, PyPtr), o, k)
PyObject_Dir(o) = ccall(POINTERS.PyObject_Dir, PyPtr, (PyPtr,), o)
PyObject_GetIter(o) = ccall(POINTERS.PyObject_GetIter, PyPtr, (PyPtr,), o)
PyObject_Call(f, args, kwargs) = ccall(POINTERS.PyObject_Call, PyPtr, (PyPtr, PyPtr, PyPtr), f, args, kwargs)
PyObject_CallObject(f, args) = ccall(POINTERS.PyObject_CallObject, PyPtr, (PyPtr, PyPtr), f, args)

PyObject_Type() = POINTERS.PyBaseObject_Type

PyObject_From(x::PyObjectRef) = (Py_IncRef(x.ptr); x.ptr)
PyObject_From(x::Nothing) = PyNone_New()
PyObject_From(x::Missing) = PyNone_New()
PyObject_From(x::Bool) = PyBool_From(x)
PyObject_From(
    x::Union{Int8,Int16,Int32,Int64,Int128,UInt8,UInt16,UInt32,UInt64,UInt128,BigInt},
) = PyLong_From(x)
PyObject_From(
    x::Rational{
        <:Union{Int8,Int16,Int32,Int64,Int128,UInt8,UInt16,UInt32,UInt64,UInt128,BigInt},
    },
) = PyFraction_From(x)
PyObject_From(x::Union{Float16,Float32,Float64}) = PyFloat_From(x)
PyObject_From(x::Complex{<:Union{Float16,Float32,Float64}}) = PyComplex_From(x)
PyObject_From(x::Union{String,SubString{String}}) = PyUnicode_From(x)
PyObject_From(x::Union{Base.CodeUnits{UInt8,String},Base.CodeUnits{UInt8,SubString{String}}}) = PyBytes_From(x)
PyObject_From(x::Char) = PyUnicode_From(string(x))
PyObject_From(x::Tuple) = PyTuple_From(x)
PyObject_From(
    x::AbstractRange{
        <:Union{
            Bool,
            Int8,
            Int16,
            Int32,
            Int64,
            Int128,
            UInt8,
            UInt16,
            UInt32,
            UInt64,
            UInt128,
            BigInt,
        },
    },
) = PyRange_From(x)
PyObject_From(x::DateTime) = PyDateTime_From(x)
PyObject_From(x::Date) = PyDate_From(x)
PyObject_From(x::Time) = PyTime_From(x)
PyObject_From(x::Union{Second,Millisecond,Microsecond,Nanosecond}) = PyTimeDelta_From(x)
PyObject_From(x) =
    if ispyref(x)
        GC.@preserve x begin
            ptr = pyptr(x)
            Py_IncRef(ptr)
            ptr
        end
    else
        PyJuliaValue_From(x)
        # PyErr_SetString(PyExc_TypeError(), "Cannot convert this Julia '$(typeof(x))' to a Python object.")
        # PyNULL
    end

function PyObject_CallArgs(
    f,
    args::Tuple,
    kwargs::Union{
        Nothing,
        NamedTuple,
        Base.Iterators.Pairs{Symbol},
        Base.Iterators.Pairs{Union{}},
    } = nothing,
)
    if kwargs !== nothing && !isempty(kwargs)
        argso = PyTuple_From(args)
        isnull(argso) && return PyNULL
        kwargso = PyDict_From(kwargs)
        isnull(kwargso) && (Py_DecRef(argso); return PyNULL)
        r = PyObject_Call(f, argso, kwargso)
        Py_DecRef(argso)
        Py_DecRef(kwargso)
        r
    elseif !isempty(args)
        argso = PyTuple_From(Tuple(args))
        isnull(argso) && return PyNULL
        r = PyObject_CallObject(f, argso)
        Py_DecRef(argso)
        return r
    else
        PyObject_CallObject(f, C_NULL)
    end
end

PyObject_CallNice(f, args...; kwargs...) = PyObject_CallArgs(f, args, kwargs)

"""
If true, we precompile C functions implementing conversion.

Otherwise, we use ordinary Julia functions.
"""
const TRYCONVERT_C = true
const TRYCONVERT_FUNCTYPE = TRYCONVERT_C ? Ptr{Cvoid} : Function

"""
Mapping of Julia types to mappings of Python types to vectors of compiled functions implementing the conversion.
"""
const TRYCONVERT_COMPILED_RULES = Dict{Type,Dict{PyPtr,Vector{TRYCONVERT_FUNCTYPE}}}()

const TRYCONVERT_COMPILED_RULES_CACHE = Dict{Any,TRYCONVERT_FUNCTYPE}()

"""
Mapping of type names to lists of rules.

A rule is a triple `(priority, type, impl)`.

Rules are applied in priority order, then in MRO order, then in the order in this list. You should currently only use the following values:
- `0` (the default) for most rules.
- `100` for "canonical" rules which are the preferred way to convert a type, e.g. `float` to `Float64`.
- `-100` for "last ditch" rules once all other options have been exhausted (currently used for anything to `PyObject` or `PyRef`).

The rule is applied only when the `type` has non-trivial intersection with the target type `T`.

Finally `impl` is the function implementing the conversion.
Its signature is `(o, T, S)` where `o` is the `PyPtr` object to convert, `T` is the desired type, and `S = typeintersect(T, type)`.
On success it returns `putresult(x)` where `x::T` is the converted value (this stores the result and returns `1`).
If conversion was not possible, returns `0` (indicating we move on to the next rule in the list).
On error, returns `-1`.
"""
const TRYCONVERT_RULES = Dict{String,Vector{Tuple{Int,Type,Any}}}()

"""
List of niladic functions returning a pointer to a type. We always check for subtypes of these types in TryConvert.

Can also return NULL. Without an error set, this indicates the type is not loaded (e.g. its containing module is not loaded) and therefore is skipped over.
"""
const TRYCONVERT_EXTRATYPES = Vector{Any}()

@generated PyObject_TryConvert_CompiledRules(T::Type) =
    get!(valtype(TRYCONVERT_COMPILED_RULES), TRYCONVERT_COMPILED_RULES, T)

PyObject_TryConvert_Rules(n::String) =
    get!(Vector{Tuple{Int,Type,Function}}, TRYCONVERT_RULES, n)
PyObject_TryConvert_AddRule(n::String, @nospecialize(T), @nospecialize(rule), priority::Int = 0) =
    push!(PyObject_TryConvert_Rules(n), (priority, T, rule))
PyObject_TryConvert_AddExtraType(@nospecialize(tfunc)) = push!(TRYCONVERT_EXTRATYPES, tfunc)

PyObject_TryConvert_CompileRule(@nospecialize(T::Type), t::PyPtr) = begin

    ### STAGE 1: Get a list of supertypes to consider.
    #
    # This includes the MRO of t, but also the MRO of any extra type (such as an ABC)
    # that we explicitly check for because it may not appear the the MRO of t.

    # MRO of t
    tmro = PyType_MROAsVector(t)
    # find the "MROs" of all base types we are considering
    basetypes = PyPtr[t]
    basemros = Vector{PyPtr}[tmro]
    for xtf in TRYCONVERT_EXTRATYPES
        xt = xtf()
        isnull(xt) && (PyErr_IsSet() ? (return PYERR()) : continue)
        xb = PyNULL
        for b in tmro
            r = PyObject_IsSubclass(b, xt)
            ism1(r) && return PYERR()
            r != 0 && (xb = b)
        end
        if xb != PyNULL
            push!(basetypes, xt)
            xmro = PyType_MROAsVector(xt)
            push!(basemros, xb in xmro ? xmro : [xb; xmro])
        end
    end
    for b in basetypes[2:end]
        push!(basemros, [b])
    end
    # merge the MROs
    # this is a port of the merge() function at the bottom of:
    # https://www.python.org/download/releases/2.3/mro/
    alltypes = PyPtr[]
    while !isempty(basemros)
        # find the first head not contained in any tail
        ok = false
        b = PyNULL
        for bmro in basemros
            b = bmro[1]
            if all(bmro -> b ∉ bmro[2:end], basemros)
                ok = true
                break
            end
        end
        ok || error(
            "Fatal inheritence error: could not merge MROs (alltypes=$alltypes, basemros=$basemros)",
        )
        # add it to the list
        push!(alltypes, b)
        # remove it from consideration
        for bmro in basemros
            filter!(x -> x != b, bmro)
        end
        # remove empty lists
        filter!(x -> !isempty(x), basemros)
    end
    # check the original MRO is preserved
    @assert filter(x -> x in tmro, alltypes) == tmro
    # some special cases
    extranames = Dict()
    for (i, b) in reverse(collect(enumerate(alltypes)))
        if PyObject_HasAttrString(b, "__array_struct__") != 0
            push!(get!(Vector, extranames, i), "<arraystruct>")
            break
        end
    end
    for (i, b) in reverse(collect(enumerate(alltypes)))
        if PyObject_HasAttrString(b, "__array_interface__") != 0
            push!(get!(Vector, extranames, i), "<arrayinterface>")
            break
        end
    end
    for (i, b) in reverse(collect(enumerate(alltypes)))
        if PyObject_HasAttrString(b, "__array__") != 0
            push!(get!(Vector, extranames, i), "<array>")
            break
        end
    end
    for (i, b) in reverse(collect(enumerate(alltypes)))
        if PyType_CheckBuffer(b)
            push!(get!(Vector, extranames, i), "<buffer>")
            break
        end
    end
    allnames = String[]
    for (i, t) in enumerate(alltypes)
        n = PyType_FullName(t)
        if n === PYERR()
            # if we cannot get the fully qualified name of the type, just skip it (it was probably dynamically generated)
            PyErr_Clear()
        else
            push!(allnames, n)
        end
        append!(allnames, get(Vector, extranames, i))
    end

    ### STAGE 2: Get a list of applicable conversion rules.
    #
    # These are the conversion rules of the types found above

    # gather rules of the form (priority, order, S, rule) from these types
    rules = Tuple{Int,Int,Type,Any}[]
    for n in allnames
        for (p, S, r) in PyObject_TryConvert_Rules(n)
            push!(rules, (p, length(rules) + 1, S, r))
        end
    end
    # sort by priority
    sort!(rules, by = x -> (-x[1], x[2]))
    # intersect S with T
    rules = [Pair{Type,Any}(typeintersect(S, T), r) for (p, i, S, r) in rules]
    # discard rules where S is a subtype of the union of all previous S for rules with the same implementation
    # in particular this removes rules with S==Union{} and removes duplicates
    rules = [Pair{Type,Any}(S, rule) for (i, (S, rule)) in enumerate(rules) if
     !(S <: Union{[S′ for (S′, rule′) in rules[1:i-1] if rule == rule′]...})]

    @debug "PYTHON CONVERSION FOR '$(PyType_Name(t))' to '$T'" basetypes =
        map(PyType_Name, basetypes) alltypes = allnames rules = rules

    ### STAGE 3: Define and compile functions implementing these rules.

    mkrulefunc(::Type{S}, rule) where {S} = (o::PyPtr) -> rule(o, S)

    map(rules) do x
        get!(TRYCONVERT_COMPILED_RULES_CACHE, x) do
            S, rule = x
            # make the function implementing the rule
            rulefunc = mkrulefunc(S, rule)
            precompile(rulefunc, (PyPtr,))
            if TRYCONVERT_C
                # compile it
                crulefunc = @cfunction($rulefunc, Int, (PyPtr,))
                # cache it
                push!(CACHE, crulefunc)
                # get the function pointer
                Base.unsafe_convert(Ptr{Cvoid}, crulefunc)
            else
                rulefunc
            end
        end
    end::Vector{TRYCONVERT_FUNCTYPE}
end

const TRYCONVERT_ERR_ARRAY = TRYCONVERT_FUNCTYPE[]

PyObject_TryConvert(o::PyPtr, ::Type{T}) where {T} = begin
    # First try based only on the type T
    # Used mainly by wrapper types for immediate conversion.
    r = PyObject_TryConvert__initial(o, T)::Int
    r == 0 || return r

    # Try to find an appropriate conversion based on `T` and the type of `o`.
    rules = PyObject_TryConvert_CompiledRules(T)
    t = Py_Type(o)
    crules = get(rules, t, TRYCONVERT_ERR_ARRAY)
    if crules === TRYCONVERT_ERR_ARRAY
        _crules = PyObject_TryConvert_CompileRule(T, t)
        _crules === PYERR() && return -1
        rules[t] = crules = _crules
    end
    for crule in crules
        r = TRYCONVERT_C ? ccall(crule, Int, (PyPtr,), o) : crule(o)::Int
        r == 0 || return r
    end

    # Failed to convert
    return 0
end
PyObject_TryConvert(o, ::Type{T}) where {T} =
    GC.@preserve o PyObject_TryConvert(Base.unsafe_convert(PyPtr, o), T)

PyObject_TryConvert__initial(o, T::Type) = 0

PyObject_Convert(o::PyPtr, ::Type{T}) where {T} = begin
    r = PyObject_TryConvert(o, T)
    if r == 0
        PyErr_SetString(
            PyExc_TypeError(),
            "Cannot convert this '$(PyType_Name(Py_Type(o)))' to a Julia '$T'",
        )
        -1
    elseif r == -1
        -1
    else
        0
    end
end
PyObject_Convert(o, ::Type{T}) where {T} =
    GC.@preserve o PyObject_Convert(Base.unsafe_convert(PyPtr, o), T)
