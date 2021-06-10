if DEBUG

    function newhdl(py::Context, ptr::C.PyPtr)
        ptr == C.PyNULL && return PyNULL
        key = py._nextkey
        @assert key > 0
        py._nextkey += 1
        @assert !haskey(py._handles, key)
        py._handles[key] = ptr
        PyHdl(key)
    end

    function forgethdl(py::Context, h::PyHdl)
        delete!(py._handles, h.key)
        nothing
    end

    cptr(py::Context, h::PyHdl) =
        if iserr(py, h)
            error("null handle: $h")
        elseif haskey(py._handles, h.key)
            py._handles[h.key]
        else
            error("invalid handle (possible double-free): $h")
        end

else

    newhdl(::Context, ptr::C.PyPtr) = PyHdl(ptr)

    forgethdl(::Context, h::PyHdl) = nothing

    cptr(::Context, h::PyHdl) = h.ptr

end

iserr(::Context, h::PyHdl) = h === PyNULL

function closehdl(py::Context, h::PyHdl)
    py._c.Py_DecRef(cptr(py, h))
    forgethdl(py, h)
end

function borrowhdl(py::Context, ptr::C.PyPtr)
    py._c.Py_IncRef(ptr)
    newhdl(py, ptr)
end

function stealcptr(py::Context, h::PyHdl)
    ptr = cptr(py, h)
    forgethdl(py, h)
    ptr
end

duphdl(py::Context, h::PyHdl) = borrowhdl(py, cptr(py, h))

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
