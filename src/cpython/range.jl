const PyRange_Type__ref = Ref(PyPtr())
PyRange_Type() = begin
    ptr = PyRange_Type__ref[]
    isnull(ptr) || return ptr
    bs = PyEval_GetBuiltins()
    ptr = PyMapping_GetItemString(bs, "range")
    PyRange_Type__ref[] = ptr
end

PyRange_From(a::Integer) = begin
    t = PyRange_Type()
    isnull(t) && return t
    PyObject_CallNice(t, a)
end

PyRange_From(a::Integer, b::Integer) = begin
    t = PyRange_Type()
    isnull(t) && return t
    PyObject_CallNice(t, a, b)
end

PyRange_From(a::Integer, b::Integer, c::Integer) = begin
    t = PyRange_Type()
    isnull(t) && return t
    PyObject_CallNice(t, a, b, c)
end

PyRange_From(r::AbstractRange{<:Integer}) =
    PyRange_From(first(r), last(r) + sign(step(r)), step(r))

steptype(::Type{<:(StepRange{A,B} where {A})}) where {B} = B
steptype(::Type{<:StepRange}) = Any

PyRange_TryConvertRule_steprange(o, ::Type{T}, ::Type{S}) where {T,S<:StepRange} = begin
    A = _typeintersect(Integer, eltype(S))
    B = _typeintersect(Integer, steptype(S))
    # get start
    ao = PyObject_GetAttrString(o, "start")
    isnull(ao) && return -1
    r = PyObject_TryConvert(ao, A)
    Py_DecRef(ao)
    r == 1 || return r
    a = takeresult(A)
    # get step
    bo = PyObject_GetAttrString(o, "step")
    isnull(bo) && return -1
    r = PyObject_TryConvert(bo, B)
    Py_DecRef(bo)
    r == 1 || return r
    b = takeresult(B)
    # get stop
    co = PyObject_GetAttrString(o, "stop")
    isnull(co) && return -1
    r = PyObject_TryConvert(co, A)
    Py_DecRef(co)
    r == 1 || return r
    c = takeresult(A)
    # success
    a′, c′ = promote(a, c - oftype(c, sign(b)))
    putresult(T, tryconvert(S, StepRange(a′, b, c′)))
end

PyRange_TryConvertRule_unitrange(o, ::Type{T}, ::Type{S}) where {T,S<:UnitRange} = begin
    A = _typeintersect(Integer, eltype(S))
    # get step
    bo = PyObject_GetAttrString(o, "step")
    isnull(bo) && return -1
    r = PyObject_TryConvert(bo, Int)
    Py_DecRef(bo)
    r == 1 || return r
    b = takeresult(Int)
    b == 1 || return 0
    # get start
    ao = PyObject_GetAttrString(o, "start")
    isnull(ao) && return -1
    r = PyObject_TryConvert(ao, A)
    Py_DecRef(ao)
    r == 1 || return r
    a = takeresult(A)
    # get stop
    co = PyObject_GetAttrString(o, "stop")
    isnull(co) && return -1
    r = PyObject_TryConvert(co, A)
    Py_DecRef(co)
    r == 1 || return r
    c = takeresult(A)
    # success
    a′, c′ = promote(a, c - oftype(c, 1))
    putresult(T, tryconvert(S, UnitRange(a′, c′)))
end
