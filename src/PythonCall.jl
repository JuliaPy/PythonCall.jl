module PythonCall

const ROOT_DIR = dirname(@__DIR__)

using Dates: Date, DateTime, Microsecond, Millisecond, Nanosecond, Second, Time

include("API/API.jl")
include("Utils/Utils.jl")
include("NumpyDates/NumpyDates.jl")
include("C/C.jl")
include("GIL/GIL.jl")
include("GC/GC.jl")
include("Core/Core.jl")
include("Convert/Convert.jl")
include("PyMacro/PyMacro.jl")
include("Wrap/Wrap.jl")
include("JlWrap/JlWrap.jl")
include("Compat/Compat.jl")

# non-exported API
using .Core: PyNULL, CONFIG

# not API but used in tests
for k in [
    :pyjlanytype,
    :pyjlarraytype,
    :pyjlvectortype,
    :pyjlbinaryiotype,
    :pyjltextiotype,
    :pyjldicttype,
    :pyjlsettype,
]
    @eval using .JlWrap: $k
end

function __init__()
    # Core pyconvert rules
    pyconvert_add_rule(Convert.pyconvert_rule_none, "builtins:NoneType", Nothing, Any)
    pyconvert_add_rule(Convert.pyconvert_rule_bool, "builtins:bool", Bool, Any)
    pyconvert_add_rule(Convert.pyconvert_rule_float, "builtins:float", Float64, Any)
    pyconvert_add_rule(Convert.pyconvert_rule_complex, "builtins:complex", Complex{Float64}, Any)
    pyconvert_add_rule(Convert.pyconvert_rule_int, "numbers:Integral", Integer, Any)
    pyconvert_add_rule(Convert.pyconvert_rule_str, "builtins:str", String, Any)
    pyconvert_add_rule(Convert.pyconvert_rule_bytes, "builtins:bytes", Base.CodeUnits{UInt8,String}, Any)
    pyconvert_add_rule(
        Convert.pyconvert_rule_range,
        "builtins:range",
        StepRange{<:Integer,<:Integer},
        Any,
    )
    pyconvert_add_rule(Convert.pyconvert_rule_fraction, "numbers:Rational", Rational{<:Integer}, Any)
    pyconvert_add_rule(Convert.pyconvert_rule_iterable, "builtins:tuple", NamedTuple, Any)
    pyconvert_add_rule(Convert.pyconvert_rule_iterable, "builtins:tuple", Tuple, Any)
    pyconvert_add_rule(Convert.pyconvert_rule_datetime, "datetime:datetime", DateTime, Any)
    pyconvert_add_rule(Convert.pyconvert_rule_date, "datetime:date", Date, Any)
    pyconvert_add_rule(Convert.pyconvert_rule_time, "datetime:time", Time, Any)
    pyconvert_add_rule(Convert.pyconvert_rule_timedelta, "datetime:timedelta", Microsecond, Any)
    pyconvert_add_rule(Convert.pyconvert_rule_exception, "builtins:BaseException", PyException, Any)
    pyconvert_add_rule(Convert.pyconvert_rule_none, "builtins:NoneType", Missing, Missing)
    pyconvert_add_rule(Convert.pyconvert_rule_bool, "builtins:bool", Number, Number)
    pyconvert_add_rule(Convert.pyconvert_rule_float, "numbers:Real", Number, Number)
    pyconvert_add_rule(Convert.pyconvert_rule_float, "builtins:float", Nothing, Nothing)
    pyconvert_add_rule(Convert.pyconvert_rule_float, "builtins:float", Missing, Missing)
    pyconvert_add_rule(Convert.pyconvert_rule_complex, "numbers:Complex", Number, Number)
    pyconvert_add_rule(Convert.pyconvert_rule_int, "numbers:Integral", Number, Number)
    pyconvert_add_rule(Convert.pyconvert_rule_str, "builtins:str", Symbol, Symbol)
    pyconvert_add_rule(Convert.pyconvert_rule_str, "builtins:str", Char, Char)
    pyconvert_add_rule(Convert.pyconvert_rule_bytes, "builtins:bytes", Vector{UInt8}, Vector{UInt8})
    pyconvert_add_rule(
        Convert.pyconvert_rule_range,
        "builtins:range",
        UnitRange{<:Integer},
        UnitRange{<:Integer},
    )
    pyconvert_add_rule(Convert.pyconvert_rule_fraction, "numbers:Rational", Number, Number)
    pyconvert_add_rule(Convert.pyconvert_rule_iterable, "collections.abc:Iterable", Vector, Vector)
    pyconvert_add_rule(Convert.pyconvert_rule_iterable, "collections.abc:Iterable", Tuple, Tuple)
    pyconvert_add_rule(Convert.pyconvert_rule_iterable, "collections.abc:Iterable", Pair, Pair)
    pyconvert_add_rule(Convert.pyconvert_rule_iterable, "collections.abc:Iterable", Set, Set)
    pyconvert_add_rule(Convert.pyconvert_rule_iterable, "collections.abc:Sequence", Vector, Vector)
    pyconvert_add_rule(Convert.pyconvert_rule_iterable, "collections.abc:Sequence", Tuple, Tuple)
    pyconvert_add_rule(Convert.pyconvert_rule_iterable, "collections.abc:Set", Set, Set)
    pyconvert_add_rule(Convert.pyconvert_rule_mapping, "collections.abc:Mapping", Dict, Dict)
    pyconvert_add_rule(Convert.pyconvert_rule_timedelta, "datetime:timedelta", Millisecond, Millisecond)
    pyconvert_add_rule(Convert.pyconvert_rule_timedelta, "datetime:timedelta", Second, Second)
    pyconvert_add_rule(Convert.pyconvert_rule_timedelta, "datetime:timedelta", Nanosecond, Nanosecond)

    # ctypes rules
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Cchar,true}(), "ctypes:c_char", Cchar, Cchar)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Cchar,true}(), "ctypes:c_char", Int, Int)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Cchar,false}(), "ctypes:c_char", Integer, Integer)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Cchar,false}(), "ctypes:c_char", Real, Real)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Cchar,false}(), "ctypes:c_char", Number, Number)
    pyconvert_add_rule(
        Convert.pyconvert_rule_ctypessimplevalue{Cwchar_t,true}(),
        "ctypes:c_wchar",
        Cwchar_t,
        Cwchar_t,
    )
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Cwchar_t,true}(), "ctypes:c_wchar", Int, Int)
    pyconvert_add_rule(
        Convert.pyconvert_rule_ctypessimplevalue{Cwchar_t,false}(),
        "ctypes:c_wchar",
        Integer,
        Integer,
    )
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Cwchar_t,false}(), "ctypes:c_wchar", Real, Real)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Cwchar_t,false}(), "ctypes:c_wchar", Number, Number)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Cchar,true}(), "ctypes:c_byte", Cchar, Cchar)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Cchar,true}(), "ctypes:c_byte", Int, Int)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Cchar,false}(), "ctypes:c_byte", Integer, Integer)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Cchar,false}(), "ctypes:c_byte", Real, Real)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Cchar,false}(), "ctypes:c_byte", Number, Number)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Cuchar,true}(), "ctypes:c_ubyte", Cuchar, Cuchar)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Cuchar,true}(), "ctypes:c_ubyte", UInt, UInt)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Cuchar,true}(), "ctypes:c_ubyte", Int, Int)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Cuchar,false}(), "ctypes:c_ubyte", Integer, Integer)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Cuchar,false}(), "ctypes:c_ubyte", Real, Real)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Cuchar,false}(), "ctypes:c_ubyte", Number, Number)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Cshort,true}(), "ctypes:c_short", Cshort, Cshort)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Cshort,true}(), "ctypes:c_short", Int, Int)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Cshort,false}(), "ctypes:c_short", Integer, Integer)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Cshort,false}(), "ctypes:c_short", Real, Real)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Cshort,false}(), "ctypes:c_short", Number, Number)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Cushort,true}(), "ctypes:c_ushort", Cushort, Cushort)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Cushort,true}(), "ctypes:c_ushort", UInt, UInt)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Cushort,true}(), "ctypes:c_ushort", Int, Int)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Cushort,false}(), "ctypes:c_ushort", Integer, Integer)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Cushort,false}(), "ctypes:c_ushort", Real, Real)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Cushort,false}(), "ctypes:c_ushort", Number, Number)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Cint,true}(), "ctypes:c_int", Cint, Cint)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Cint,true}(), "ctypes:c_int", Int, Int)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Cint,false}(), "ctypes:c_int", Integer, Integer)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Cint,false}(), "ctypes:c_int", Real, Real)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Cint,false}(), "ctypes:c_int", Number, Number)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Cuint,true}(), "ctypes:c_uint", Cuint, Cuint)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Cuint,true}(), "ctypes:c_uint", UInt, UInt)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Cuint,true}(), "ctypes:c_uint", Int, Int)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Cuint,false}(), "ctypes:c_uint", Integer, Integer)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Cuint,false}(), "ctypes:c_uint", Real, Real)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Cuint,false}(), "ctypes:c_uint", Number, Number)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Clong,true}(), "ctypes:c_long", Clong, Clong)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Clong,true}(), "ctypes:c_long", Int, Int)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Clong,false}(), "ctypes:c_long", Integer, Integer)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Clong,false}(), "ctypes:c_long", Real, Real)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Clong,false}(), "ctypes:c_long", Number, Number)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Culong,true}(), "ctypes:c_ulong", Culong, Culong)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Culong,true}(), "ctypes:c_ulong", UInt, UInt)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Culong,false}(), "ctypes:c_ulong", Int, Int)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Culong,false}(), "ctypes:c_ulong", Integer, Integer)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Culong,false}(), "ctypes:c_ulong", Real, Real)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Culong,false}(), "ctypes:c_ulong", Number, Number)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Clonglong,true}(), "ctypes:c_longlong", Clonglong, Clonglong)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Clonglong,true}(), "ctypes:c_longlong", Int, Int)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Clonglong,false}(), "ctypes:c_longlong", Integer, Integer)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Clonglong,false}(), "ctypes:c_longlong", Real, Real)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Clonglong,false}(), "ctypes:c_longlong", Number, Number)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Culonglong,true}(), "ctypes:c_ulonglong", Culonglong, Culonglong)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Culonglong,true}(), "ctypes:c_ulonglong", UInt, UInt)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Culonglong,false}(), "ctypes:c_ulonglong", Int, Int)
    pyconvert_add_rule(
        Convert.pyconvert_rule_ctypessimplevalue{Culonglong,false}(),
        "ctypes:c_ulonglong",
        Integer,
        Integer,
    )
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Culonglong,false}(), "ctypes:c_ulonglong", Real, Real)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Culonglong,false}(), "ctypes:c_ulonglong", Number, Number)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Csize_t,true}(), "ctypes:c_size_t", Csize_t, Csize_t)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Csize_t,true}(), "ctypes:c_size_t", UInt, UInt)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Csize_t,false}(), "ctypes:c_size_t", Int, Int)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Csize_t,false}(), "ctypes:c_size_t", Integer, Integer)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Csize_t,false}(), "ctypes:c_size_t", Real, Real)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Csize_t,false}(), "ctypes:c_size_t", Number, Number)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Cssize_t,true}(), "ctypes:c_ssize_t", Cssize_t, Cssize_t)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Cssize_t,true}(), "ctypes:c_ssize_t", Int, Int)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Cssize_t,false}(), "ctypes:c_ssize_t", Integer, Integer)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Cssize_t,false}(), "ctypes:c_ssize_t", Real, Real)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Cssize_t,false}(), "ctypes:c_ssize_t", Number, Number)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Cfloat,true}(), "ctypes:c_float", Cfloat, Cfloat)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Cfloat,true}(), "ctypes:c_float", Float64, Float64)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Cfloat,false}(), "ctypes:c_float", Real, Real)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Cfloat,false}(), "ctypes:c_float", Number, Number)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Cdouble,true}(), "ctypes:c_double", Cdouble, Cdouble)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Cdouble,true}(), "ctypes:c_double", Float64, Float64)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Cdouble,false}(), "ctypes:c_double", Real, Real)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Cdouble,false}(), "ctypes:c_double", Number, Number)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Ptr{Cchar},true}(), "ctypes:c_char_p", Cstring, Cstring)
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Ptr{Cchar},true}(), "ctypes:c_char_p", Ptr{Cchar}, Ptr{Cchar})
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Ptr{Cchar},true}(), "ctypes:c_char_p", Ptr, Ptr)
    pyconvert_add_rule(
        Convert.pyconvert_rule_ctypessimplevalue{Ptr{Cwchar_t},true}(),
        "ctypes:c_wchar_p",
        Cwstring,
        Cwstring,
    )
    pyconvert_add_rule(
        Convert.pyconvert_rule_ctypessimplevalue{Ptr{Cwchar_t},true}(),
        "ctypes:c_wchar_p",
        Ptr{Cwchar_t},
        Ptr{Cwchar_t},
    )
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Ptr{Cwchar_t},true}(), "ctypes:c_wchar_p", Ptr, Ptr)
    pyconvert_add_rule(
        Convert.pyconvert_rule_ctypessimplevalue{Ptr{Cvoid},true}(),
        "ctypes:c_void_p",
        Ptr{Cvoid},
        Ptr{Cvoid},
    )
    pyconvert_add_rule(Convert.pyconvert_rule_ctypessimplevalue{Ptr{Cvoid},true}(), "ctypes:c_void_p", Ptr, Ptr)

    # numpy rules
    pyconvert_add_rule(Convert.pyconvert_rule_numpysimplevalue{Bool,true}(), "numpy:bool_", Bool, Any)
    pyconvert_add_rule(Convert.pyconvert_rule_numpysimplevalue{Bool,true}(), "numpy:bool_", UInt, UInt)
    pyconvert_add_rule(Convert.pyconvert_rule_numpysimplevalue{Bool,true}(), "numpy:bool_", Int, Int)
    pyconvert_add_rule(Convert.pyconvert_rule_numpysimplevalue{Bool,false}(), "numpy:bool_", Integer, Integer)
    pyconvert_add_rule(Convert.pyconvert_rule_numpysimplevalue{Bool,false}(), "numpy:bool_", Real, Real)
    pyconvert_add_rule(Convert.pyconvert_rule_numpysimplevalue{Bool,false}(), "numpy:bool_", Number, Number)
    pyconvert_add_rule(Convert.pyconvert_rule_numpysimplevalue{Int8,true}(), "numpy:int8", Int8, Any)
    pyconvert_add_rule(Convert.pyconvert_rule_numpysimplevalue{Int8,true}(), "numpy:int8", Int, Int)
    pyconvert_add_rule(Convert.pyconvert_rule_numpysimplevalue{Int8,false}(), "numpy:int8", Integer, Integer)
    pyconvert_add_rule(Convert.pyconvert_rule_numpysimplevalue{Int8,false}(), "numpy:int8", Real, Real)
    pyconvert_add_rule(Convert.pyconvert_rule_numpysimplevalue{Int8,false}(), "numpy:int8", Number, Number)
    pyconvert_add_rule(Convert.pyconvert_rule_numpysimplevalue{Int16,true}(), "numpy:int16", Int16, Any)
    pyconvert_add_rule(Convert.pyconvert_rule_numpysimplevalue{Int16,true}(), "numpy:int16", Int, Int)
    pyconvert_add_rule(Convert.pyconvert_rule_numpysimplevalue{Int16,false}(), "numpy:int16", Integer, Integer)
    pyconvert_add_rule(Convert.pyconvert_rule_numpysimplevalue{Int16,false}(), "numpy:int16", Real, Real)
    pyconvert_add_rule(Convert.pyconvert_rule_numpysimplevalue{Int16,false}(), "numpy:int16", Number, Number)
    pyconvert_add_rule(Convert.pyconvert_rule_numpysimplevalue{Int32,true}(), "numpy:int32", Int32, Any)
    pyconvert_add_rule(Convert.pyconvert_rule_numpysimplevalue{Int32,true}(), "numpy:int32", Int, Int)
    pyconvert_add_rule(Convert.pyconvert_rule_numpysimplevalue{Int32,false}(), "numpy:int32", Integer, Integer)
    pyconvert_add_rule(Convert.pyconvert_rule_numpysimplevalue{Int32,false}(), "numpy:int32", Real, Real)
    pyconvert_add_rule(Convert.pyconvert_rule_numpysimplevalue{Int32,false}(), "numpy:int32", Number, Number)
    pyconvert_add_rule(Convert.pyconvert_rule_numpysimplevalue{Int64,true}(), "numpy:int64", Int64, Any)
    pyconvert_add_rule(Convert.pyconvert_rule_numpysimplevalue{Int64,true}(), "numpy:int64", Int, Int)
    pyconvert_add_rule(Convert.pyconvert_rule_numpysimplevalue{Int64,false}(), "numpy:int64", Integer, Integer)
    pyconvert_add_rule(Convert.pyconvert_rule_numpysimplevalue{Int64,false}(), "numpy:int64", Real, Real)
    pyconvert_add_rule(Convert.pyconvert_rule_numpysimplevalue{Int64,false}(), "numpy:int64", Number, Number)
    pyconvert_add_rule(Convert.pyconvert_rule_numpysimplevalue{UInt8,true}(), "numpy:uint8", UInt8, Any)
    pyconvert_add_rule(Convert.pyconvert_rule_numpysimplevalue{UInt8,true}(), "numpy:uint8", UInt, UInt)
    pyconvert_add_rule(Convert.pyconvert_rule_numpysimplevalue{UInt8,true}(), "numpy:uint8", Int, Int)
    pyconvert_add_rule(Convert.pyconvert_rule_numpysimplevalue{UInt8,false}(), "numpy:uint8", Integer, Integer)
    pyconvert_add_rule(Convert.pyconvert_rule_numpysimplevalue{UInt8,false}(), "numpy:uint8", Real, Real)
    pyconvert_add_rule(Convert.pyconvert_rule_numpysimplevalue{UInt8,false}(), "numpy:uint8", Number, Number)
    pyconvert_add_rule(Convert.pyconvert_rule_numpysimplevalue{UInt16,true}(), "numpy:uint16", UInt16, Any)
    pyconvert_add_rule(Convert.pyconvert_rule_numpysimplevalue{UInt16,true}(), "numpy:uint16", UInt, UInt)
    pyconvert_add_rule(Convert.pyconvert_rule_numpysimplevalue{UInt16,true}(), "numpy:uint16", Int, Int)
    pyconvert_add_rule(Convert.pyconvert_rule_numpysimplevalue{UInt16,false}(), "numpy:uint16", Integer, Integer)
    pyconvert_add_rule(Convert.pyconvert_rule_numpysimplevalue{UInt16,false}(), "numpy:uint16", Real, Real)
    pyconvert_add_rule(Convert.pyconvert_rule_numpysimplevalue{UInt16,false}(), "numpy:uint16", Number, Number)
    pyconvert_add_rule(Convert.pyconvert_rule_numpysimplevalue{UInt32,true}(), "numpy:uint32", UInt32, Any)
    pyconvert_add_rule(Convert.pyconvert_rule_numpysimplevalue{UInt32,true}(), "numpy:uint32", UInt, UInt)
    pyconvert_add_rule(Convert.pyconvert_rule_numpysimplevalue{UInt32,true}(), "numpy:uint32", Int, Int)
    pyconvert_add_rule(Convert.pyconvert_rule_numpysimplevalue{UInt32,false}(), "numpy:uint32", Integer, Integer)
    pyconvert_add_rule(Convert.pyconvert_rule_numpysimplevalue{UInt32,false}(), "numpy:uint32", Real, Real)
    pyconvert_add_rule(Convert.pyconvert_rule_numpysimplevalue{UInt32,false}(), "numpy:uint32", Number, Number)
    pyconvert_add_rule(Convert.pyconvert_rule_numpysimplevalue{UInt64,true}(), "numpy:uint64", UInt64, Any)
    pyconvert_add_rule(Convert.pyconvert_rule_numpysimplevalue{UInt64,true}(), "numpy:uint64", UInt, UInt)
    pyconvert_add_rule(Convert.pyconvert_rule_numpysimplevalue{UInt64,false}(), "numpy:uint64", Int, Int)
    pyconvert_add_rule(Convert.pyconvert_rule_numpysimplevalue{UInt64,false}(), "numpy:uint64", Integer, Integer)
    pyconvert_add_rule(Convert.pyconvert_rule_numpysimplevalue{UInt64,false}(), "numpy:uint64", Real, Real)
    pyconvert_add_rule(Convert.pyconvert_rule_numpysimplevalue{UInt64,false}(), "numpy:uint64", Number, Number)
    pyconvert_add_rule(Convert.pyconvert_rule_numpysimplevalue{Float16,true}(), "numpy:float16", Float16, Any)
    pyconvert_add_rule(Convert.pyconvert_rule_numpysimplevalue{Float16,true}(), "numpy:float16", Float64, Float64)
    pyconvert_add_rule(Convert.pyconvert_rule_numpysimplevalue{Float16,false}(), "numpy:float16", Real, Real)
    pyconvert_add_rule(Convert.pyconvert_rule_numpysimplevalue{Float16,false}(), "numpy:float16", Number, Number)
    pyconvert_add_rule(Convert.pyconvert_rule_numpysimplevalue{Float32,true}(), "numpy:float32", Float32, Any)
    pyconvert_add_rule(Convert.pyconvert_rule_numpysimplevalue{Float32,true}(), "numpy:float32", Float64, Float64)
    pyconvert_add_rule(Convert.pyconvert_rule_numpysimplevalue{Float32,false}(), "numpy:float32", Real, Real)
    pyconvert_add_rule(Convert.pyconvert_rule_numpysimplevalue{Float32,false}(), "numpy:float32", Number, Number)
    pyconvert_add_rule(Convert.pyconvert_rule_numpysimplevalue{Float64,true}(), "numpy:float64", Float64, Any)
    pyconvert_add_rule(Convert.pyconvert_rule_numpysimplevalue{Float64,true}(), "numpy:float64", Float64, Float64)
    pyconvert_add_rule(Convert.pyconvert_rule_numpysimplevalue{Float64,false}(), "numpy:float64", Real, Real)
    pyconvert_add_rule(Convert.pyconvert_rule_numpysimplevalue{Float64,false}(), "numpy:float64", Number, Number)
    pyconvert_add_rule(Convert.pyconvert_rule_numpysimplevalue{ComplexF16,true}(), "numpy:complex32", ComplexF16, Any)
    pyconvert_add_rule(Convert.pyconvert_rule_numpysimplevalue{ComplexF16,true}(), "numpy:complex32", ComplexF64, ComplexF64)
    pyconvert_add_rule(Convert.pyconvert_rule_numpysimplevalue{ComplexF16,false}(), "numpy:complex32", Complex, Complex)
    pyconvert_add_rule(Convert.pyconvert_rule_numpysimplevalue{ComplexF16,false}(), "numpy:complex32", Number, Number)
    pyconvert_add_rule(Convert.pyconvert_rule_numpysimplevalue{ComplexF32,true}(), "numpy:complex64", ComplexF32, Any)
    pyconvert_add_rule(Convert.pyconvert_rule_numpysimplevalue{ComplexF32,true}(), "numpy:complex64", ComplexF64, ComplexF64)
    pyconvert_add_rule(Convert.pyconvert_rule_numpysimplevalue{ComplexF32,false}(), "numpy:complex64", Complex, Complex)
    pyconvert_add_rule(Convert.pyconvert_rule_numpysimplevalue{ComplexF32,false}(), "numpy:complex64", Number, Number)
    pyconvert_add_rule(Convert.pyconvert_rule_numpysimplevalue{ComplexF64,true}(), "numpy:complex128", ComplexF64, Any)
    pyconvert_add_rule(Convert.pyconvert_rule_numpysimplevalue{ComplexF64,true}(), "numpy:complex128", ComplexF64, ComplexF64)
    pyconvert_add_rule(Convert.pyconvert_rule_numpysimplevalue{ComplexF64,false}(), "numpy:complex128", Complex, Complex)
    pyconvert_add_rule(Convert.pyconvert_rule_numpysimplevalue{ComplexF64,false}(), "numpy:complex128", Number, Number)
    pyconvert_add_rule(Convert.pyconvert_rule_datetime64, "numpy:datetime64", NumpyDates.DateTime64, Any)
    pyconvert_add_rule(Convert.pyconvert_rule_datetime64, "numpy:datetime64", NumpyDates.InlineDateTime64, NumpyDates.InlineDateTime64)
    pyconvert_add_rule(
        Convert.pyconvert_rule_datetime64,
        "numpy:datetime64",
        NumpyDates.DatesInstant,
        NumpyDates.DatesInstant,
    )
    pyconvert_add_rule(Convert.pyconvert_rule_datetime64, "numpy:datetime64", Missing, Missing)
    pyconvert_add_rule(Convert.pyconvert_rule_datetime64, "numpy:datetime64", Nothing, Nothing)
    pyconvert_add_rule(Convert.pyconvert_rule_timedelta64, "numpy:timedelta64", NumpyDates.TimeDelta64, Any)
    pyconvert_add_rule(
        Convert.pyconvert_rule_timedelta64,
        "numpy:timedelta64",
        NumpyDates.InlineTimeDelta64,
        NumpyDates.InlineTimeDelta64,
    )
    pyconvert_add_rule(
        Convert.pyconvert_rule_timedelta64,
        "numpy:timedelta64",
        NumpyDates.DatesPeriod,
        NumpyDates.DatesPeriod,
    )
    pyconvert_add_rule(Convert.pyconvert_rule_timedelta64, "numpy:timedelta64", Missing, Missing)
    pyconvert_add_rule(Convert.pyconvert_rule_timedelta64, "numpy:timedelta64", Nothing, Nothing)

    # pandas rules
    pyconvert_add_rule(
        Convert.pyconvert_rule_pandas_na,
        "pandas._libs.missing:NAType",
        Missing,
        Any,
    )
    pyconvert_add_rule(Convert.pyconvert_rule_pandas_na, "pandas._libs.missing:NAType", Nothing, Nothing)

    # wrapper rules
    pyconvert_add_rule(Wrap.pyconvert_rule_array_nocopy, "<arraystruct>", Wrap.PyArray, Any)
    pyconvert_add_rule(Wrap.pyconvert_rule_array_nocopy, "<arrayinterface>", Wrap.PyArray, Any)
    pyconvert_add_rule(Wrap.pyconvert_rule_array_nocopy, "<array>", Wrap.PyArray, Any)
    pyconvert_add_rule(Wrap.pyconvert_rule_array_nocopy, "<buffer>", Wrap.PyArray, Any)
    pyconvert_add_rule(Wrap.pyconvert_rule_iterable, "collections.abc:Iterable", Wrap.PyIterable, Wrap.PyIterable)
    pyconvert_add_rule(Wrap.pyconvert_rule_sequence, "collections.abc:Sequence", Wrap.PyList, Wrap.PyList)
    pyconvert_add_rule(Wrap.pyconvert_rule_set, "collections.abc:Set", Wrap.PySet, Wrap.PySet)
    pyconvert_add_rule(Wrap.pyconvert_rule_mapping, "collections.abc:Mapping", Wrap.PyDict, Wrap.PyDict)
    pyconvert_add_rule(Wrap.pyconvert_rule_io, "io:IOBase", Wrap.PyIO, Wrap.PyIO)
    pyconvert_add_rule(Wrap.pyconvert_rule_io, "_io:_IOBase", Wrap.PyIO, Wrap.PyIO)
    pyconvert_add_rule(
        Wrap.pyconvert_rule_pandasdataframe,
        "pandas.core.frame:DataFrame",
        Wrap.PyPandasDataFrame,
        Wrap.PyPandasDataFrame,
    )
    pyconvert_add_rule(
        Wrap.pyconvert_rule_sequence,
        "pandas.core.arrays.base:ExtensionArray",
        Wrap.PyList,
        Wrap.PyList,
    )
    pyconvert_add_rule(Wrap.pyconvert_rule_array, "<arraystruct>", Array, Array)
    pyconvert_add_rule(Wrap.pyconvert_rule_array, "<arrayinterface>", Array, Array)
    pyconvert_add_rule(Wrap.pyconvert_rule_array, "<array>", Array, Array)
    pyconvert_add_rule(Wrap.pyconvert_rule_array, "<buffer>", Array, Array)
    pyconvert_add_rule(Wrap.pyconvert_rule_array, "<arraystruct>", AbstractArray, AbstractArray)
    pyconvert_add_rule(Wrap.pyconvert_rule_array, "<arrayinterface>", AbstractArray, AbstractArray)
    pyconvert_add_rule(Wrap.pyconvert_rule_array, "<array>", AbstractArray, AbstractArray)
    pyconvert_add_rule(Wrap.pyconvert_rule_array, "<buffer>", AbstractArray, AbstractArray)

    # JlWrap rules
    pyconvert_add_rule(JlWrap.pyconvert_rule_jlvalue, "juliacall:JlBase", Any, Any)

    # Fallback
    pyconvert_add_rule(Convert.pyconvert_rule_object, "builtins:object", Py, Any)
end

end
