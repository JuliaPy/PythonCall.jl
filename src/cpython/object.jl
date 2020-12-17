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

# const PYCONVERT_RULES = IdDict{Type, Dict{PyPtr, Ptr{Cvoid}}}()
# const PYCONVERT_RULES_CACHE = IdDict{Type, Dict{PyPtr, Any}}()

# @generated PyConvert_GetRules(::Type{T}) where {T} =
#     get!(Dict{PyPtr, Ptr{Cvoid}}, PYCONVERT_RULES, T)

# function PyObject_Convert__rule(t::PyPtr)
#     name = Py_DecRef(PyObject_GetAttrString(t, "__name__")) do tnameo
#         Py_DecRef(PyObject_GetAttrString(t, "__module__")) do mnameo
#             r = PyUnicode_TryConvert(tnameo, String)
#             r == 1 || return PYERR()
#             tname = takeresult(String)
#             r = PyUnicode_TryConvert(mnameo, String)
#             r == 1 || return PYERR()
#             mname = takeresult(String)
#             "$mname.$tname"
#         end
#     end
#     name == PYERR() && return PYERR()
#     PyObject_Convert__rule(t, Val(Symbol(name)))
# end

# PyObject_Convert__rule(t::PyPtr, ::Val) = nothing
# PyObject_Convert__rule(t::PyPtr, ::Val{Symbol("builtins.NoneType")}) = Py_Is(t, Py_Type(Py_None())) ? PyNone_TryConvert : nothing
# PyObject_Convert__rule(t::PyPtr, ::Val{Symbol("builtins.bool")}) = Py_Is(t, PyBool_Type()) ? PyBool_TryConvert : nothing
# PyObject_Convert__rule(t::PyPtr, ::Val{Symbol("builtins.str")}) = Py_Is(t, PyUnicode_Type()) ? PyUnicode_TryConvert : nothing
# PyObject_Convert__rule(t::PyPtr, ::Val{Symbol("builtins.bytes")}) = Py_Is(t, PyBytes_Type()) ? PyBytes_TryConvert : nothing
# PyObject_Convert__rule(t::PyPtr, ::Val{Symbol("builtins.int")}) = Py_Is(t, PyLong_Type()) ? PyLong_TryConvert : nothing
# PyObject_Convert__rule(t::PyPtr, ::Val{Symbol("builtins.float")}) = Py_Is(t, PyFloat_Type()) ? PyFloat_TryConvert : nothing
# PyObject_Convert__rule(t::PyPtr, ::Val{Symbol("builtins.complex")}) = Py_Is(t, PyComplex_Type()) ? PyComplex_TryConvert : nothing

# struct PyObject_Convert__rule_struct{T,F}
#     f :: F
# end
# (r::PyObject_Convert__rule_struct{T,F})(o::PyPtr) where {T,F} = r.f(o, T)::Int

const ERRPTR = Ptr{Cvoid}(1)

const TRYCONVERT_COMPILED_RULES = IdDict{Type, Dict{PyPtr, Ptr{Cvoid}}}()
const TRYCONVERT_RULES = Dict{String, Vector{Tuple{Int, Type, Function}}}()
const TRYCONVERT_EXTRATYPES = Vector{Tuple{String, Function}}()

@generated PyObject_TryConvert_CompiledRules(::Type{T}) where {T} =
    get!(Dict{PyPtr, Ptr{Cvoid}}, TRYCONVERT_COMPILED_RULES, T)
PyObject_TryConvert_Rules(n::String) =
    get!(Vector{Tuple{Int, Type, Function}}, TRYCONVERT_RULES, n)
PyObject_TryConvert_AddRule(n::String, T::Type, rule::Function, priority::Int=0) =
    push!(PyObject_TryConvert_Rules(n), (priority, T, rule))
PyObject_TryConvert_AddRules(n::String, xs) =
    for x in xs
        PyObject_TryConvert_AddRule(n, x...)
    end
PyObject_TryConvert_AddExtraType(n::String, pred::Function) =
    pushfirst!(TRYCONVERT_EXTRATYPES, (n, pred))
PyObject_TryConvert_AddExtraTypes(xs) =
    for x in xs
        PyObject_TryConvert_AddExtraType(x...)
    end

PyObject_TryConvert_CompileRule(::Type{T}, t::PyPtr) where {T} = begin
    # first get the MRO
    mrotuple = PyType_MRO(t)
    mrotypes = PyPtr[]
    mronames = String[]
    for i in 1:PyTuple_Size(mrotuple)
        b = PyTuple_GetItem(mrotuple, i-1)
        isnull(b) && return ERRPTR
        push!(mrotypes, b)
        n = PyType_FullName(b)
        n === PYERR() && return ERRPTR
        push!(mronames, n)
    end
    # find any extra types
    extranames = Dict{Int, Vector{String}}()
    for (name, pred) in TRYCONVERT_EXTRATYPES
        i = nothing
        for (j,t) in enumerate(mrotypes)
            r = pred(t)
            if r === PYERR()
                return ERRPTR
            elseif r isa Bool
                if r
                    i = j
                end
            elseif r isa Cint
                r == -1 && return ERRPTR
                if r != 0
                    i = j
                end
            else
                @warn "Unexpected return type from a Python convert extra-types predicate: `$(typeof(r))`"
            end
        end
        i === nothing || push!(get!(Vector{String}, extranames, i), name)
    end
    # merge all the type names together
    allnames = [n for (i,mroname) in enumerate(mronames) for n in [mroname; get(Vector{String}, extranames, i)]]
    # gather rules of the form (priority, order, S, rule) from these types
    rules = Tuple{Int, Int, Type, Function}[]
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
    println("CONVERSION RULES FOR '$(PyType_Name(t))' TO '$T':")
    display(rules)
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
