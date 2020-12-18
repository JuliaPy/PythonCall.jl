@cdef :_PyObject_New PyPtr (PyPtr,)
@cdef :PyObject_ClearWeakRefs Cvoid (PyPtr,)
@cdef :PyObject_HasAttrString Cint (PyPtr, Cstring)
@cdef :PyObject_HasAttr Cint (PyPtr, PyPtr)
@cdef :PyObject_GetAttrString PyPtr (PyPtr, Cstring)
@cdef :PyObject_GetAttr PyPtr (PyPtr, PyPtr)
@cdef :PyObject_GenericGetAttr PyPtr (PyPtr, PyPtr)
@cdef :PyObject_SetAttrString Cint (PyPtr, Cstring, PyPtr)
@cdef :PyObject_SetAttr Cint (PyPtr, PyPtr, PyPtr)
@cdef :PyObject_GenericSetAttr Cint (PyPtr, PyPtr, PyPtr)
@cdef :PyObject_DelAttrString Cint (PyPtr, Cstring)
@cdef :PyObject_DelAttr Cint (PyPtr, PyPtr)
@cdef :PyObject_RichCompare PyPtr (PyPtr, PyPtr, Cint)
@cdef :PyObject_RichCompareBool Cint (PyPtr, PyPtr, Cint)
@cdef :PyObject_Repr PyPtr (PyPtr,)
@cdef :PyObject_ASCII PyPtr (PyPtr,)
@cdef :PyObject_Str PyPtr (PyPtr,)
@cdef :PyObject_Bytes PyPtr (PyPtr,)
@cdef :PyObject_IsSubclass Cint (PyPtr, PyPtr)
@cdef :PyObject_IsInstance Cint (PyPtr, PyPtr)
@cdef :PyObject_Hash Py_hash_t (PyPtr,)
@cdef :PyObject_IsTrue Cint (PyPtr,)
@cdef :PyObject_Length Py_ssize_t (PyPtr,)
@cdef :PyObject_GetItem PyPtr (PyPtr, PyPtr)
@cdef :PyObject_SetItem Cint (PyPtr, PyPtr, PyPtr)
@cdef :PyObject_DelItem Cint (PyPtr, PyPtr)
@cdef :PyObject_Dir PyPtr (PyPtr,)
@cdef :PyObject_GetIter PyPtr (PyPtr,)
@cdef :PyObject_Call PyPtr (PyPtr, PyPtr, PyPtr)
@cdef :PyObject_CallObject PyPtr (PyPtr, PyPtr)

PyObject_From(x::PyObjectRef) = (Py_IncRef(x.ptr); x.ptr)
PyObject_From(x::Nothing) = PyNone_New()
PyObject_From(x::Bool) = PyBool_From(x)
PyObject_From(x::Union{Int8,Int16,Int32,Int64,Int128,UInt8,UInt16,UInt32,UInt64,UInt128,BigInt}) = PyLong_From(x)
PyObject_From(x::Union{Float16,Float32,Float64}) = PyFloat_From(x)
PyObject_From(x::Complex{<:Union{Float16,Float32,Float64}}) = PyComplex_From(x)
PyObject_From(x::Union{String,SubString{String}}) = PyUnicode_From(x)
PyObject_From(x::Tuple) = PyTuple_From(x)
PyObject_From(x::AbstractRange{<:Union{Bool,Int8,Int16,Int32,Int64,Int128,UInt8,UInt16,UInt32,UInt64,UInt128,BigInt}}) = PyRange_From(x)
PyObject_From(x::T) where {T} =
    if ispyreftype(T)
        GC.@preserve x begin
            ptr = pyptr(x)
            Py_IncRef(ptr)
            ptr
        end
    else
        PyErr_SetString(PyExc_TypeError(), "Cannot convert this Julia '$(typeof(x))' to a Python object.")
        PyPtr()
    end

function PyObject_CallArgs(f, args::Tuple, kwargs::Union{Nothing,NamedTuple,Base.Iterators.Pairs{Symbol},Base.Iterators.Pairs{Union{}}}=nothing)
    if kwargs!==nothing && !isempty(kwargs)
        argso = PyTuple_From(args)
        isnull(argso) && return PyPtr()
        kwargso = PyDict_From(kwargs)
        isnull(kwargso) && (Py_DecRef(argso); return PyPtr())
        r = PyObject_Call(f, argso, kwargso)
        Py_DecRef(argso)
        Py_DecRef(kwargso)
        r
    elseif !isempty(args)
        argso = PyTuple_From(Tuple(args))
        isnull(argso) && return PyPtr()
        r = PyObject_CallObject(f, argso)
        Py_DecRef(argso)
        return r
    else
        PyObject_CallObject(f, C_NULL)
    end
end

PyObject_CallNice(f, args...; kwargs...) = PyObject_CallArgs(f, args, kwargs)

const ERRPTR = Ptr{Cvoid}(1)

"""
Mapping of Julia types to mappings of Python types to compiled functions implementing the conversion.

That is, `ccall(TRYCONVERT_COMPILED_RULES[T][Py_Type(o)], Int, (PyPtr,), o)` attempts to convert `o` to a `T`.
On success, this returns `1` and the result can be obtained with `takeresult(T)`.
Otherwise returns `0` if no conversion was possible, or `-1` on error.
"""
const TRYCONVERT_COMPILED_RULES = IdDict{Type, Dict{PyPtr, Ptr{Cvoid}}}()

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
On success it returns `putresult(T, x)` where `x` is the converted value (this stores the result and returns `1`).
If conversion was not possible, returns `0` (indicating we move on to the next rule in the list).
On error, returns `-1`.
"""
const TRYCONVERT_RULES = Dict{String, Vector{Tuple{Int, Type, Any}}}()

"""
List of niladic functions returning a pointer to a type. We always check for subtypes of these types in TryConvert.

Can also return NULL. Without an error set, this indicates the type is not loaded (e.g. its containing module is not loaded) and therefore is skipped over.
"""
const TRYCONVERT_EXTRATYPES = Vector{Any}()

@generated PyObject_TryConvert_CompiledRules(::Type{T}) where {T} =
    get!(Dict{PyPtr, Ptr{Cvoid}}, TRYCONVERT_COMPILED_RULES, T)
PyObject_TryConvert_Rules(n::String) =
    get!(Vector{Tuple{Int, Type, Function}}, TRYCONVERT_RULES, n)
PyObject_TryConvert_AddRule(n::String, T, rule, priority=0) =
    push!(PyObject_TryConvert_Rules(n), (priority, T, rule))
PyObject_TryConvert_AddRules(n::String, xs) =
    for x in xs
        PyObject_TryConvert_AddRule(n, x...)
    end
PyObject_TryConvert_AddExtraType(tfunc) =
    push!(TRYCONVERT_EXTRATYPES, tfunc)
PyObject_TryConvert_AddExtraTypes(xs) =
    for x in xs
        PyObject_TryConvert_AddExtraType(x)
    end

PyObject_TryConvert_CompileRule(::Type{T}, t::PyPtr) where {T} = begin

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
        isnull(xt) && (PyErr_IsSet() ? (return ERRPTR) : continue)
        xb = PyPtr()
        for b in tmro
            r = PyObject_IsSubclass(b, xt)
            ism1(r) && return ERRPTR
            r != 0 && (xb = b)
        end
        if xb != PyPtr()
            push!(basetypes, xt)
            push!(basemros, [xb; PyType_MROAsVector(xt)])
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
        b = PyPtr()
        for bmro in basemros
            b = bmro[1]
            if all(bmro -> b ∉ bmro[2:end], basemros)
                ok = true
                break
            end
        end
        ok || error("Fatal inheritence error: could not merge MROs")
        # add it to the list
        push!(alltypes, b)
        # remove it from consideration
        for bmro in basemros
            filter!(x -> x != b, bmro)
        end
        # remove empty lists
        filter!(x -> !isempty(x), basemros)
    end
    allnames = map(PyType_FullName, alltypes)
    # as a special case, we check for the buffer type explicitly
    for (i,b) in reverse(collect(enumerate(alltypes)))
        if PyType_CheckBuffer(b)
            insert!(allnames, i+1, "<buffer>")
            break
        end
    end
    # check the original MRO is preserved
    @assert filter(x -> x in tmro, alltypes) == tmro

    ### STAGE 2: Get a list of applicable conversion rules.
    #
    # These are the conversion rules of the types found above

    # gather rules of the form (priority, order, S, rule) from these types
    rules = Tuple{Int, Int, Type, Any}[]
    for n in allnames
        for (p, S, r) in PyObject_TryConvert_Rules(n)
            push!(rules, (p, length(rules)+1, S, r))
        end
    end
    # sort by priority
    sort!(rules, by=x->(-x[1], x[2]))
    # intersect S with T
    rules = [typeintersect(S,T) => r for (p,i,S,r) in rules]
    # discard rules where S is a subtype of the union of all previous S for rules with the same implementation
    # in particular this removes rules with S==Union{} and removes duplicates
    rules = [S=>rule for (i,(S,rule)) in enumerate(rules) if !(S <: Union{[S′ for (S′,rule′) in rules[1:i-1] if rule==rule′]...})]

    @debug "PYTHON CONVERSION FOR '$(PyType_FullName(t))' to '$T'" basetypes=map(PyType_FullName, basetypes) alltypes=allnames rules=rules

    ### STAGE 3: Define and compile a function implementing these rules.

    # make the function implementing these rules
    rulefunc = @eval (o::PyPtr) -> begin
        $((:(r = $rule(o, $T, $S)::Int; r == 0 || return r) for (S,rule) in rules)...)
        return 0
    end
    # compile it
    rulecfunc = @cfunction($rulefunc, Int, (PyPtr,))
    push!(CACHE, rulecfunc)
    Base.unsafe_convert(Ptr{Cvoid}, rulecfunc)
end

PyObject_TryConvert(o::PyPtr, ::Type{T}) where {T} = begin
    # First try based only on the type T
    # Used mainly by wrapper types.
    r = PyObject_TryConvert__initial(o, T) :: Int
    r == 0 || return r

    # Try to find an appropriate conversion based on `T` and the type of `o`.
    rules = PyObject_TryConvert_CompiledRules(T)
    t = Py_Type(o)
    rule = get(rules, t, ERRPTR)
    if rule == ERRPTR
        rule = PyObject_TryConvert_CompileRule(T, t)
        rule == ERRPTR && return -1
        rules[t] = rule
    end
    if !isnull(rule)
        r = ccall(rule, Int, (PyPtr,), o)
        r == 0 || return r
    end

    0
end
PyObject_TryConvert(o, ::Type{T}) where {T} = GC.@preserve o PyObject_TryConvert(Base.unsafe_convert(PyPtr, o), T)

PyObject_TryConvert__initial(o, ::Type{T}) where {T} = 0

PyObject_Convert(o::PyPtr, ::Type{T}) where {T} = begin
    r = PyObject_TryConvert(o, T)
    if r == 0
        PyErr_SetString(PyExc_TypeError(), "Cannot convert this '$(PyType_Name(Py_Type(o)))' to a Julia '$T'")
        -1
    elseif r == -1
        -1
    else
        0
    end
end
PyObject_Convert(o, ::Type{T}) where {T} = GC.@preserve o PyObject_Convert(Base.unsafe_convert(PyPtr, o), T)
