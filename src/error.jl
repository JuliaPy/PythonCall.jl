cpyerrtype() = cpycall_raw(Val(:PyErr_Occurred), CPyPtr)

pyerroccurred() = cpyerrtype() != C_NULL
function pyerroccurred(t::AbstractPyObject)
    e = cpyerrtype()
    e != C_NULL && cpycall_boolx(Val(:PyErr_GivenExceptionMatches), e, t)
end

pyerrclear() = cpycall_raw(Val(:PyErr_Clear), Cvoid)

pyerrcheck() = pyerroccurred() ? pythrow() : nothing

pyerrset(t::AbstractPyObject) = cpycall_voidx(Val(:PyErr_SetNone), t)
pyerrset(t::AbstractPyObject, x::AbstractString) = cpycall_voidx(Val(:PyErr_SetString), t, x)
pyerrset(t::AbstractPyObject, x::AbstractPyObject) = cpycall_voidx(Val(:PyErr_SetObject), t, x)

struct PythonRuntimeError <: Exception
    t :: PyObject
    v :: PyObject
    b :: PyObject
end

function pythrow()
    t = Ref{CPyPtr}()
    v = Ref{CPyPtr}()
    b = Ref{CPyPtr}()
    cpycall_raw(Val(:PyErr_Fetch), Cvoid, t, v, b)
    cpycall_raw(Val(:PyErr_NormalizeException), Cvoid, t, v, b)
    to = t[]==C_NULL ? PyObject(pynone) : pynewobject(t[])
    vo = v[]==C_NULL ? PyObject(pynone) : pynewobject(v[])
    bo = b[]==C_NULL ? PyObject(pynone) : pynewobject(b[])
    throw(PythonRuntimeError(to, vo, bo))
end

function pythrow(v::AbstractPyObject)
    throw(PythonRuntimeError(pytype(v), v, pynone))
end
