function Base.getproperty(ctx::Context, k::Symbol)
    k == :Py_DecRef ? (x) -> ccall(ctx.pointers.Py_DecRef, Cvoid, (PyPtr,), x) :
    k == :Py_IncRef ? (x) -> ccall(ctx.pointers.Py_IncRef, Cvoid, (PyPtr,), x) :
    k == :PyObject_Repr ? (x) -> ccall(ctx.pointers.PyObject_Repr, PyPtr, (PyPtr,), x) :
    k == :PyObject_Str ? (x) -> ccall(ctx.pointers.PyObject_Str, PyPtr, (PyPtr,), x) :
    k == :PyObject_ASCII ? (x) -> ccall(ctx.pointers.PyObject_ASCII, PyPtr, (PyPtr,), x) :
    k == :PyObject_Bytes ? (x) -> ccall(ctx.pointers.PyObject_Bytes, PyPtr, (PyPtr,), x) :
    k == :PyUnicode_DecodeUTF8 ? (buf, len=sizeof(buf), errors=C_NULL) -> ccall(ctx.pointers.PyUnicode_DecodeUTF8, PyPtr, (Ptr{UInt8}, Py_ssize_t, Ptr{Cvoid}), buf, len, errors) :
    k == :PyUnicode_AsUTF8String ? (x) -> ccall(ctx.pointers.PyUnicode_AsUTF8String, PyPtr, (PyPtr,), x) :
    k == :PyUnicode_AsJuliaString ? function (x)
        b = ctx.PyUnicode_AsUTF8String(x)
        if b == PyNULL
            ""
        else
            s = ctx.PyBytes_AsJuliaString(b)
            ctx.Py_DecRef(b)
            s
        end
    end :
    k == :PyUnicode_AsJuliaVector ? function (x)
        b = ctx.PyUnicode_AsUTF8String(x)
        if b == PyNULL
            UInt8[]
        else
            v = ctx.PyBytes_AsJuliaVector(b)
            ctx.Py_DecRef(b)
            v
        end
    end :
    k == :PyBytes_AsStringAndSize ? function (x)
        buf = Ref(Ptr{UInt8}(0))
        len = Ref(Py_ssize_t(0))
        err = ccall(ctx.pointers.PyBytes_AsStringAndSize, Cint, (PyPtr, Ptr{Ptr{UInt8}}, Ptr{Py_ssize_t}), x, buf, len)
        err == -1 ? (Ptr{UInt8}(0), Py_ssize_t(0)) : (buf[], len[])
    end :
    k == :PyBytes_AsJuliaString ? function (x)
        buf, len = ctx.PyBytes_AsStringAndSize(x)
        buf == C_NULL ? "" : Base.unsafe_string(buf, len)
    end :
    k == :PyBytes_AsJuliaVector ? function (x)
        buf, len = ctx.PyBytes_AsStringAndSize(x)
        buf == C_NULL ? UInt8[] : Base.unsafe_wrap(Array, buf, len)
    end :
    getfield(ctx, k)
end

function Base.propertynames(ctx::Context, private::Bool=false)
    (
        fieldnames(Context)...,
        :Py_DecRef,
        :Py_IncRef,
        :PyObject_Repr,
        :PyObject_Str,
        :PyObject_ASCII,
        :PyObject_Bytes,
        :PyUnicode_DecodeUTF8,
        :PyUnicode_AsUTF8String,
        :PyUnicode_AsJuliaString,
        :PyUnicode_AsJuliaVector,
        :PyBytes_AsStringAndSize,
        :PyBytes_AsJuliaString,
        :PyBytes_AsJuliaVector,
    )
end
