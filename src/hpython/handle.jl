if DEBUG

    function newhdl(ctx::Context, ptr::C.PyPtr)
        ptr == C.PyNULL && return PyNULL
        key = ctx._nextkey
        @assert key > 0
        ctx._nextkey += 1
        @assert !haskey(ctx._handles, key)
        ctx._handles[key] = ptr
        PyHdl(key)
    end

    function forgethdl(ctx::Context, h::PyHdl)
        delete!(ctx._handles, h.key)
        nothing
    end

    cptr(ctx::Context, h::PyHdl) =
        if iserr(ctx, h)
            error("null handle: $h")
        elseif haskey(ctx._handles, h.key)
            ctx._handles[h.key]
        else
            error("invalid handle (possible double-free): $h")
        end

    iserr(ctx::Context, h::PyHdl) = h.key == 0

else

    newhdl(ctx::Context, ptr::C.PyPtr) = PyHdl(ptr)

    forgethdl(ctx::Context, h::PyHdl) = nothing

    cptr(ctx::Context, h::PyHdl) = h.ptr

    iserr(ctx::Context, h::PyHdl) = h.ptr == C.PyNULL

end

function closehdl(ctx::Context, h::PyHdl)
    ctx._c.Py_DecRef(cptr(ctx, h))
    forgethdl(ctx, h)
end

function borrowhdl(ctx::Context, ptr::C.PyPtr)
    ctx._c.Py_IncRef(ptr)
    newhdl(ctx, ptr)
end

function stealcptr(ctx::Context, h::PyHdl)
    ptr = cptr(ctx, h)
    forgethdl(ctx, h)
    ptr
end

duphdl(ctx::Context, h::PyHdl) = borrowhdl(ctx, cptr(ctx, h))

@inline (f::Builtin{:newhdl})(args...) = newhdl(f.ctx, args...)
@inline (f::Builtin{:borrowhdl})(args...) = borrowhdl(f.ctx, args...)
@inline (f::Builtin{:duphdl})(args...) = duphdl(f.ctx, args...)
@inline (f::Builtin{:closehdl})(args...) = closehdl(f.ctx, args...)
@inline (f::Builtin{:cptr})(args...) = cptr(f.ctx, args...)
@inline (f::Builtin{:stealcptr})(args...) = stealcptr(f.ctx, args...)
@inline (f::Builtin{:iserr})(args...) = iserr(f.ctx, args...)

Base.getproperty(h::PyHdl, k::Symbol) =
    k == :autocheck ? PyAutoHdl{true,false}(h) :
    k == :autoclose ? PyAutoHdl{false,true}(h) :
    k == :auto ? PyAutoHdl{true,true}(h) :
    k == :unauto ? h :
    getfield(h, k)

Base.propertynames(h::PyHdl) = (fieldnames(PyHdl)..., :autocheck, :autoclose, :auto, :unauto)

### AutoHdl

const PyAnyHdl = Union{PyHdl, PyAutoHdl}

Base.getproperty(h::PyAutoHdl, k::Symbol) =
    k == :unauto ? h.hdl :
    getfield(h, k)

Base.propertynames(h::PyAutoHdl, private::Bool=false) = (fieldnames(PyAutoHdl)..., :unauto)

iserr(ctx::Context, h::PyAutoHdl) = iserr(ctx, h.hdl)
cptr(ctx::Context, h::PyAutoHdl) = cptr(ctx, h.hdl)
duphdl(ctx::Context, h::PyAutoHdl{check,close}) where {check,close} =
    autoiserr(ctx, h) ? PyNULL : close ? h.hdl : duphdl(ctx, h.hdl)

function _autohdl(ctx::Context, ::Builtin{name}, ::Val{cname}) where {name, cname}
    h = getproperty(ctx._builtins, name)
    if ctx.iserr(h)
        h = ctx.borrowhdl(getproperty(ctx._c.pointers, cname))
        setproperty!(ctx._builtins, name, h)
    end
    return h.autocheck
end

autohdl(ctx::Context, x::PyAnyHdl) = x
autohdl(ctx::Context, x::Builtin{name}) where {name} = (ctx.errset(ctx.TypeError, "this builtin does not correspond to a python object: $name"); PyNULL.autocheck)
autohdl(ctx::Context, x::Builtin{:bool}) = _autohdl(ctx, x, Val(:PyBool_Type))
autohdl(ctx::Context, x::Builtin{:bytes}) = _autohdl(ctx, x, Val(:PyBytes_Type))
autohdl(ctx::Context, x::Builtin{:complex}) = _autohdl(ctx, x, Val(:PyComplex_Type))
autohdl(ctx::Context, x::Builtin{:dict}) = _autohdl(ctx, x, Val(:PyDict_Type))
autohdl(ctx::Context, x::Builtin{:float}) = _autohdl(ctx, x, Val(:PyFloat_Type))
autohdl(ctx::Context, x::Builtin{:int}) = _autohdl(ctx, x, Val(:PyLong_Type))
autohdl(ctx::Context, x::Builtin{:list}) = _autohdl(ctx, x, Val(:PyList_Type))
autohdl(ctx::Context, x::Builtin{:object}) = _autohdl(ctx, x, Val(:PyBaseObject_Type))
autohdl(ctx::Context, x::Builtin{:set}) = _autohdl(ctx, x, Val(:PySet_Type))
autohdl(ctx::Context, x::Builtin{:frozenset}) = _autohdl(ctx, x, Val(:PyFrozenSet_Type))
autohdl(ctx::Context, x::Builtin{:slice}) = _autohdl(ctx, x, Val(:PySlice_Type))
autohdl(ctx::Context, x::Builtin{:str}) = _autohdl(ctx, x, Val(:PyUnicode_Type))
autohdl(ctx::Context, x::Builtin{:tuple}) = _autohdl(ctx, x, Val(:PyTuple_Type))
autohdl(ctx::Context, x::Builtin{:type}) = _autohdl(ctx, x, Val(:PyType_Type))
autohdl(ctx::Context, x::Builtin{:True}) = _autohdl(ctx, x, Val(:_Py_TrueStruct))
autohdl(ctx::Context, x::Builtin{:False}) = _autohdl(ctx, x, Val(:_Py_FalseStruct))
autohdl(ctx::Context, x::Builtin{:NotImplemented}) = _autohdl(ctx, x, Val(:_Py_NotImplementedStruct))
autohdl(ctx::Context, x::Builtin{:None}) = _autohdl(ctx, x, Val(:_Py_NoneStruct))
autohdl(ctx::Context, x::Builtin{:Ellipsis}) = _autohdl(ctx, x, Val(:_Py_EllipsisObject))
autohdl(ctx::Context, x::Builtin{:BaseException}) = _autohdl(ctx, x, Val(:PyExc_BaseException))
autohdl(ctx::Context, x::Builtin{:Exception}) = _autohdl(ctx, x, Val(:PyExc_Exception))
autohdl(ctx::Context, x::Builtin{:StopIteration}) = _autohdl(ctx, x, Val(:PyExc_StopIteration))
autohdl(ctx::Context, x::Builtin{:GeneratorExit}) = _autohdl(ctx, x, Val(:PyExc_GeneratorExit))
autohdl(ctx::Context, x::Builtin{:ArithmeticError}) = _autohdl(ctx, x, Val(:PyExc_ArithmeticError))
autohdl(ctx::Context, x::Builtin{:LookupError}) = _autohdl(ctx, x, Val(:PyExc_LookupError))
autohdl(ctx::Context, x::Builtin{:AssertionError}) = _autohdl(ctx, x, Val(:PyExc_AssertionError))
autohdl(ctx::Context, x::Builtin{:AttributeError}) = _autohdl(ctx, x, Val(:PyExc_AttributeError))
autohdl(ctx::Context, x::Builtin{:BufferError}) = _autohdl(ctx, x, Val(:PyExc_BufferError))
autohdl(ctx::Context, x::Builtin{:EOFError}) = _autohdl(ctx, x, Val(:PyExc_EOFError))
autohdl(ctx::Context, x::Builtin{:FloatingPointError}) = _autohdl(ctx, x, Val(:PyExc_FloatingPointError))
autohdl(ctx::Context, x::Builtin{:OSError}) = _autohdl(ctx, x, Val(:PyExc_OSError))
autohdl(ctx::Context, x::Builtin{:ImportError}) = _autohdl(ctx, x, Val(:PyExc_ImportError))
autohdl(ctx::Context, x::Builtin{:IndexError}) = _autohdl(ctx, x, Val(:PyExc_IndexError))
autohdl(ctx::Context, x::Builtin{:KeyError}) = _autohdl(ctx, x, Val(:PyExc_KeyError))
autohdl(ctx::Context, x::Builtin{:KeyboardInterrupt}) = _autohdl(ctx, x, Val(:PyExc_KeyboardInterrupt))
autohdl(ctx::Context, x::Builtin{:MemoryError}) = _autohdl(ctx, x, Val(:PyExc_MemoryError))
autohdl(ctx::Context, x::Builtin{:NameError}) = _autohdl(ctx, x, Val(:PyExc_NameError))
autohdl(ctx::Context, x::Builtin{:OverflowError}) = _autohdl(ctx, x, Val(:PyExc_OverflowError))
autohdl(ctx::Context, x::Builtin{:RuntimeError}) = _autohdl(ctx, x, Val(:PyExc_RuntimeError))
autohdl(ctx::Context, x::Builtin{:RecursionError}) = _autohdl(ctx, x, Val(:PyExc_RecursionError))
autohdl(ctx::Context, x::Builtin{:NotImplementedError}) = _autohdl(ctx, x, Val(:PyExc_NotImplementedError))
autohdl(ctx::Context, x::Builtin{:SyntaxError}) = _autohdl(ctx, x, Val(:PyExc_SyntaxError))
autohdl(ctx::Context, x::Builtin{:IndentationError}) = _autohdl(ctx, x, Val(:PyExc_IndentationError))
autohdl(ctx::Context, x::Builtin{:TabError}) = _autohdl(ctx, x, Val(:PyExc_TabError))
autohdl(ctx::Context, x::Builtin{:ReferenceError}) = _autohdl(ctx, x, Val(:PyExc_ReferenceError))
autohdl(ctx::Context, x::Builtin{:SystemError}) = _autohdl(ctx, x, Val(:PyExc_SystemError))
autohdl(ctx::Context, x::Builtin{:SystemExit}) = _autohdl(ctx, x, Val(:PyExc_SystemExit))
autohdl(ctx::Context, x::Builtin{:TypeError}) = _autohdl(ctx, x, Val(:PyExc_TypeError))
autohdl(ctx::Context, x::Builtin{:UnboundLocalError}) = _autohdl(ctx, x, Val(:PyExc_UnboundLocalError))
autohdl(ctx::Context, x::Builtin{:UnicodeError}) = _autohdl(ctx, x, Val(:PyExc_UnicodeError))
autohdl(ctx::Context, x::Builtin{:UnicodeEncodeError}) = _autohdl(ctx, x, Val(:PyExc_UnicodeEncodeError))
autohdl(ctx::Context, x::Builtin{:UnicodeDecodeError}) = _autohdl(ctx, x, Val(:PyExc_UnicodeDecodeError))
autohdl(ctx::Context, x::Builtin{:UnicodeTranslateError}) = _autohdl(ctx, x, Val(:PyExc_UnicodeTranslateError))
autohdl(ctx::Context, x::Builtin{:ValueError}) = _autohdl(ctx, x, Val(:PyExc_ValueError))
autohdl(ctx::Context, x::Builtin{:ZeroDivisionError}) = _autohdl(ctx, x, Val(:PyExc_ZeroDivisionError))
autohdl(ctx::Context, x::Builtin{:BlockingIOError}) = _autohdl(ctx, x, Val(:PyExc_BlockingIOError))
autohdl(ctx::Context, x::Builtin{:BrokenPipeError}) = _autohdl(ctx, x, Val(:PyExc_BrokenPipeError))
autohdl(ctx::Context, x::Builtin{:ChildProcessError}) = _autohdl(ctx, x, Val(:PyExc_ChildProcessError))
autohdl(ctx::Context, x::Builtin{:ConnectionError}) = _autohdl(ctx, x, Val(:PyExc_ConnectionError))
autohdl(ctx::Context, x::Builtin{:ConnectionAbortedError}) = _autohdl(ctx, x, Val(:PyExc_ConnectionAbortedError))
autohdl(ctx::Context, x::Builtin{:ConnectionRefusedError}) = _autohdl(ctx, x, Val(:PyExc_ConnectionRefusedError))
autohdl(ctx::Context, x::Builtin{:FileExistsError}) = _autohdl(ctx, x, Val(:PyExc_FileExistsError))
autohdl(ctx::Context, x::Builtin{:FileNotFoundError}) = _autohdl(ctx, x, Val(:PyExc_FileNotFoundError))
autohdl(ctx::Context, x::Builtin{:InterruptedError}) = _autohdl(ctx, x, Val(:PyExc_InterruptedError))
autohdl(ctx::Context, x::Builtin{:IsADirectoryError}) = _autohdl(ctx, x, Val(:PyExc_IsADirectoryError))
autohdl(ctx::Context, x::Builtin{:NotADirectoryError}) = _autohdl(ctx, x, Val(:PyExc_NotADirectoryError))
autohdl(ctx::Context, x::Builtin{:PermissionError}) = _autohdl(ctx, x, Val(:PyExc_PermissionError))
autohdl(ctx::Context, x::Builtin{:ProcessLookupError}) = _autohdl(ctx, x, Val(:PyExc_ProcessLookupError))
autohdl(ctx::Context, x::Builtin{:TimeoutError}) = _autohdl(ctx, x, Val(:PyExc_TimeoutError))
autohdl(ctx::Context, x::Builtin{:EnvironmentError}) = _autohdl(ctx, x, Val(:PyExc_EnvironmentError))
autohdl(ctx::Context, x::Builtin{:IOError}) = _autohdl(ctx, x, Val(:PyExc_IOError))
autohdl(ctx::Context, x::Builtin{:Warning}) = _autohdl(ctx, x, Val(:PyExc_Warning))
autohdl(ctx::Context, x::Builtin{:UserWarning}) = _autohdl(ctx, x, Val(:PyExc_UserWarning))
autohdl(ctx::Context, x::Builtin{:DeprecationWarning}) = _autohdl(ctx, x, Val(:PyExc_DeprecationWarning))
autohdl(ctx::Context, x::Builtin{:PendingDeprecationWarning}) = _autohdl(ctx, x, Val(:PyExc_PendingDeprecationWarning))
autohdl(ctx::Context, x::Builtin{:SyntaxWarning}) = _autohdl(ctx, x, Val(:PyExc_SyntaxWarning))
autohdl(ctx::Context, x::Builtin{:RuntimeWarning}) = _autohdl(ctx, x, Val(:PyExc_RuntimeWarning))
autohdl(ctx::Context, x::Builtin{:FutureWarning}) = _autohdl(ctx, x, Val(:PyExc_FutureWarning))
autohdl(ctx::Context, x::Builtin{:ImportWarning}) = _autohdl(ctx, x, Val(:PyExc_ImportWarning))
autohdl(ctx::Context, x::Builtin{:UnicodeWarning}) = _autohdl(ctx, x, Val(:PyExc_UnicodeWarning))
autohdl(ctx::Context, x::Builtin{:BytesWarning}) = _autohdl(ctx, x, Val(:PyExc_BytesWarning))
autohdl(ctx::Context, x::Builtin{:ResourceWarning}) = _autohdl(ctx, x, Val(:PyExc_ResourceWarning))
autohdl(ctx::Context, x) = pyobject(ctx, x).auto
@inline (f::Builtin{:autohdl})(x) = autohdl(f.ctx, x)

autoiserr(ctx::Context, h) = false
autoiserr(ctx::Context, h::PyAutoHdl{check}) where {check} = check && iserr(ctx, h.hdl)
@inline (f::Builtin{:autoiserr})(h) = autoiserr(f.ctx, h)

autoclosehdl(ctx::Context, h) = nothing
autoclosehdl(ctx::Context, h::PyAutoHdl{check,close}) where {check,close} = autoiserr(ctx, h) ? nothing : close ? closehdl(ctx, h.hdl) : nothing
@inline (f::Builtin{:autoclosehdl})(h) = autoclosehdl(f.ctx, h)

macro autohdl(ctx, vars...)
    ans = Expr(:block)
    for var in vars
        push!(ans.args, :($var = $ctx.autohdl($var)))
        push!(ans.args, :($ctx.autoiserr($var) && @goto done))
    end
    return esc(ans)
end

macro autoclosehdl(ctx, vars...)
    ans = Expr(:block, :(@label done))
    for var in vars
        push!(ans.args, :($ctx.autoclosehdl($var)))
    end
    return esc(ans)
end

### OrErr

iserr(ctx::Context, x) = false
value(ctx::Context, x) = x

struct VoidOrErr
    val :: Cint
end
VoidOrErr() = VoidOrErr(-1)
iserr(ctx::Context, x::VoidOrErr) = c.val == -1
value(ctx::Context, x::VoidOrErr) = nothing

struct BoolOrErr
    val :: Cint
end
BoolOrErr() = BoolOrErr(-1)
iserr(ctx::Context, x::BoolOrErr) = c.val == -1
value(ctx::Context, x::BoolOrErr) = c.val != 0
