const PyAnyHdl = Union{PyHdl, PyAutoHdl}

Base.getproperty(h::PyAutoHdl, k::Symbol) =
    k == :unauto ? h.hdl :
    getfield(h, k)

Base.propertynames(h::PyAutoHdl, private::Bool=false) = (fieldnames(PyAutoHdl)..., :unauto)

iserr(py::Context, h::PyAutoHdl) = iserr(py, h.hdl)
cptr(py::Context, h::PyAutoHdl) = cptr(py, h.hdl)
duphdl(py::Context, h::PyAutoHdl{check,close}) where {check,close} =
    autoiserr(py, h) ? PyNULL : close ? h.hdl : duphdl(py, h.hdl)

function _autohdl(py::Context, ::Builtin{name}, ::Val{cname}) where {name, cname}
    h = getproperty(py._builtins, name)
    if py.iserr(h)
        h = py.borrowhdl(getproperty(py._c.pointers, cname))
        setproperty!(py._builtins, name, h)
    end
    return h.autocheck
end

autohdl(py::Context, x::PyAnyHdl) = x
autohdl(py::Context, x::Builtin{name}) where {name} = (py.errset(py.TypeError, "this builtin does not correspond to a python object: $name"); PyNULL.autocheck)
autohdl(py::Context, x::Builtin{:bool}) = _autohdl(py, x, Val(:PyBool_Type))
autohdl(py::Context, x::Builtin{:bytes}) = _autohdl(py, x, Val(:PyBytes_Type))
autohdl(py::Context, x::Builtin{:complex}) = _autohdl(py, x, Val(:PyComplex_Type))
autohdl(py::Context, x::Builtin{:dict}) = _autohdl(py, x, Val(:PyDict_Type))
autohdl(py::Context, x::Builtin{:float}) = _autohdl(py, x, Val(:PyFloat_Type))
autohdl(py::Context, x::Builtin{:int}) = _autohdl(py, x, Val(:PyLong_Type))
autohdl(py::Context, x::Builtin{:list}) = _autohdl(py, x, Val(:PyList_Type))
autohdl(py::Context, x::Builtin{:object}) = _autohdl(py, x, Val(:PyBaseObject_Type))
autohdl(py::Context, x::Builtin{:set}) = _autohdl(py, x, Val(:PySet_Type))
autohdl(py::Context, x::Builtin{:frozenset}) = _autohdl(py, x, Val(:PyFrozenSet_Type))
autohdl(py::Context, x::Builtin{:slice}) = _autohdl(py, x, Val(:PySlice_Type))
autohdl(py::Context, x::Builtin{:str}) = _autohdl(py, x, Val(:PyUnicode_Type))
autohdl(py::Context, x::Builtin{:tuple}) = _autohdl(py, x, Val(:PyTuple_Type))
autohdl(py::Context, x::Builtin{:type}) = _autohdl(py, x, Val(:PyType_Type))
autohdl(py::Context, x::Builtin{:True}) = _autohdl(py, x, Val(:_Py_TrueStruct))
autohdl(py::Context, x::Builtin{:False}) = _autohdl(py, x, Val(:_Py_FalseStruct))
autohdl(py::Context, x::Builtin{:NotImplemented}) = _autohdl(py, x, Val(:_Py_NotImplementedStruct))
autohdl(py::Context, x::Builtin{:None}) = _autohdl(py, x, Val(:_Py_NoneStruct))
autohdl(py::Context, x::Builtin{:Ellipsis}) = _autohdl(py, x, Val(:_Py_EllipsisObject))
autohdl(py::Context, x::Builtin{:BaseException}) = _autohdl(py, x, Val(:PyExc_BaseException))
autohdl(py::Context, x::Builtin{:Exception}) = _autohdl(py, x, Val(:PyExc_Exception))
autohdl(py::Context, x::Builtin{:StopIteration}) = _autohdl(py, x, Val(:PyExc_StopIteration))
autohdl(py::Context, x::Builtin{:GeneratorExit}) = _autohdl(py, x, Val(:PyExc_GeneratorExit))
autohdl(py::Context, x::Builtin{:ArithmeticError}) = _autohdl(py, x, Val(:PyExc_ArithmeticError))
autohdl(py::Context, x::Builtin{:LookupError}) = _autohdl(py, x, Val(:PyExc_LookupError))
autohdl(py::Context, x::Builtin{:AssertionError}) = _autohdl(py, x, Val(:PyExc_AssertionError))
autohdl(py::Context, x::Builtin{:AttributeError}) = _autohdl(py, x, Val(:PyExc_AttributeError))
autohdl(py::Context, x::Builtin{:BufferError}) = _autohdl(py, x, Val(:PyExc_BufferError))
autohdl(py::Context, x::Builtin{:EOFError}) = _autohdl(py, x, Val(:PyExc_EOFError))
autohdl(py::Context, x::Builtin{:FloatingPointError}) = _autohdl(py, x, Val(:PyExc_FloatingPointError))
autohdl(py::Context, x::Builtin{:OSError}) = _autohdl(py, x, Val(:PyExc_OSError))
autohdl(py::Context, x::Builtin{:ImportError}) = _autohdl(py, x, Val(:PyExc_ImportError))
autohdl(py::Context, x::Builtin{:IndexError}) = _autohdl(py, x, Val(:PyExc_IndexError))
autohdl(py::Context, x::Builtin{:KeyError}) = _autohdl(py, x, Val(:PyExc_KeyError))
autohdl(py::Context, x::Builtin{:KeyboardInterrupt}) = _autohdl(py, x, Val(:PyExc_KeyboardInterrupt))
autohdl(py::Context, x::Builtin{:MemoryError}) = _autohdl(py, x, Val(:PyExc_MemoryError))
autohdl(py::Context, x::Builtin{:NameError}) = _autohdl(py, x, Val(:PyExc_NameError))
autohdl(py::Context, x::Builtin{:OverflowError}) = _autohdl(py, x, Val(:PyExc_OverflowError))
autohdl(py::Context, x::Builtin{:RuntimeError}) = _autohdl(py, x, Val(:PyExc_RuntimeError))
autohdl(py::Context, x::Builtin{:RecursionError}) = _autohdl(py, x, Val(:PyExc_RecursionError))
autohdl(py::Context, x::Builtin{:NotImplementedError}) = _autohdl(py, x, Val(:PyExc_NotImplementedError))
autohdl(py::Context, x::Builtin{:SyntaxError}) = _autohdl(py, x, Val(:PyExc_SyntaxError))
autohdl(py::Context, x::Builtin{:IndentationError}) = _autohdl(py, x, Val(:PyExc_IndentationError))
autohdl(py::Context, x::Builtin{:TabError}) = _autohdl(py, x, Val(:PyExc_TabError))
autohdl(py::Context, x::Builtin{:ReferenceError}) = _autohdl(py, x, Val(:PyExc_ReferenceError))
autohdl(py::Context, x::Builtin{:SystemError}) = _autohdl(py, x, Val(:PyExc_SystemError))
autohdl(py::Context, x::Builtin{:SystemExit}) = _autohdl(py, x, Val(:PyExc_SystemExit))
autohdl(py::Context, x::Builtin{:TypeError}) = _autohdl(py, x, Val(:PyExc_TypeError))
autohdl(py::Context, x::Builtin{:UnboundLocalError}) = _autohdl(py, x, Val(:PyExc_UnboundLocalError))
autohdl(py::Context, x::Builtin{:UnicodeError}) = _autohdl(py, x, Val(:PyExc_UnicodeError))
autohdl(py::Context, x::Builtin{:UnicodeEncodeError}) = _autohdl(py, x, Val(:PyExc_UnicodeEncodeError))
autohdl(py::Context, x::Builtin{:UnicodeDecodeError}) = _autohdl(py, x, Val(:PyExc_UnicodeDecodeError))
autohdl(py::Context, x::Builtin{:UnicodeTranslateError}) = _autohdl(py, x, Val(:PyExc_UnicodeTranslateError))
autohdl(py::Context, x::Builtin{:ValueError}) = _autohdl(py, x, Val(:PyExc_ValueError))
autohdl(py::Context, x::Builtin{:ZeroDivisionError}) = _autohdl(py, x, Val(:PyExc_ZeroDivisionError))
autohdl(py::Context, x::Builtin{:BlockingIOError}) = _autohdl(py, x, Val(:PyExc_BlockingIOError))
autohdl(py::Context, x::Builtin{:BrokenPipeError}) = _autohdl(py, x, Val(:PyExc_BrokenPipeError))
autohdl(py::Context, x::Builtin{:ChildProcessError}) = _autohdl(py, x, Val(:PyExc_ChildProcessError))
autohdl(py::Context, x::Builtin{:ConnectionError}) = _autohdl(py, x, Val(:PyExc_ConnectionError))
autohdl(py::Context, x::Builtin{:ConnectionAbortedError}) = _autohdl(py, x, Val(:PyExc_ConnectionAbortedError))
autohdl(py::Context, x::Builtin{:ConnectionRefusedError}) = _autohdl(py, x, Val(:PyExc_ConnectionRefusedError))
autohdl(py::Context, x::Builtin{:FileExistsError}) = _autohdl(py, x, Val(:PyExc_FileExistsError))
autohdl(py::Context, x::Builtin{:FileNotFoundError}) = _autohdl(py, x, Val(:PyExc_FileNotFoundError))
autohdl(py::Context, x::Builtin{:InterruptedError}) = _autohdl(py, x, Val(:PyExc_InterruptedError))
autohdl(py::Context, x::Builtin{:IsADirectoryError}) = _autohdl(py, x, Val(:PyExc_IsADirectoryError))
autohdl(py::Context, x::Builtin{:NotADirectoryError}) = _autohdl(py, x, Val(:PyExc_NotADirectoryError))
autohdl(py::Context, x::Builtin{:PermissionError}) = _autohdl(py, x, Val(:PyExc_PermissionError))
autohdl(py::Context, x::Builtin{:ProcessLookupError}) = _autohdl(py, x, Val(:PyExc_ProcessLookupError))
autohdl(py::Context, x::Builtin{:TimeoutError}) = _autohdl(py, x, Val(:PyExc_TimeoutError))
autohdl(py::Context, x::Builtin{:EnvironmentError}) = _autohdl(py, x, Val(:PyExc_EnvironmentError))
autohdl(py::Context, x::Builtin{:IOError}) = _autohdl(py, x, Val(:PyExc_IOError))
autohdl(py::Context, x::Builtin{:Warning}) = _autohdl(py, x, Val(:PyExc_Warning))
autohdl(py::Context, x::Builtin{:UserWarning}) = _autohdl(py, x, Val(:PyExc_UserWarning))
autohdl(py::Context, x::Builtin{:DeprecationWarning}) = _autohdl(py, x, Val(:PyExc_DeprecationWarning))
autohdl(py::Context, x::Builtin{:PendingDeprecationWarning}) = _autohdl(py, x, Val(:PyExc_PendingDeprecationWarning))
autohdl(py::Context, x::Builtin{:SyntaxWarning}) = _autohdl(py, x, Val(:PyExc_SyntaxWarning))
autohdl(py::Context, x::Builtin{:RuntimeWarning}) = _autohdl(py, x, Val(:PyExc_RuntimeWarning))
autohdl(py::Context, x::Builtin{:FutureWarning}) = _autohdl(py, x, Val(:PyExc_FutureWarning))
autohdl(py::Context, x::Builtin{:ImportWarning}) = _autohdl(py, x, Val(:PyExc_ImportWarning))
autohdl(py::Context, x::Builtin{:UnicodeWarning}) = _autohdl(py, x, Val(:PyExc_UnicodeWarning))
autohdl(py::Context, x::Builtin{:BytesWarning}) = _autohdl(py, x, Val(:PyExc_BytesWarning))
autohdl(py::Context, x::Builtin{:ResourceWarning}) = _autohdl(py, x, Val(:PyExc_ResourceWarning))
autohdl(py::Context, x) = pyobject(py, x).auto
@inline (f::Builtin{:autohdl})(x) = autohdl(f.ctx, x)

autoiserr(py::Context, h) = false
autoiserr(py::Context, h::PyAutoHdl{check}) where {check} = check && iserr(py, h.hdl)
@inline (f::Builtin{:autoiserr})(h) = autoiserr(f.ctx, h)

autoclosehdl(py::Context, h) = nothing
autoclosehdl(py::Context, h::PyAutoHdl{check,close}) where {check,close} = autoiserr(py, h) ? nothing : close ? closehdl(py, h.hdl) : nothing
@inline (f::Builtin{:autoclosehdl})(h) = autoclosehdl(f.ctx, h)

macro autohdl(py, vars...)
    ans = Expr(:block)
    for var in vars
        push!(ans.args, :($var = $py.autohdl($var)))
        push!(ans.args, :($py.autoiserr($var) && @goto done))
    end
    return esc(ans)
end

macro autoclosehdl(py, vars...)
    ans = Expr(:block, :(@label done))
    for var in vars
        push!(ans.args, :($py.autoclosehdl($var)))
    end
    return esc(ans)
end
