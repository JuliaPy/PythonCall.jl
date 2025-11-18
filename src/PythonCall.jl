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
    for (t, T) in Convert.CTYPES_SIMPLE_TYPES
        name = "ctypes:c_$t"
        rule = Convert.pyconvert_rule_ctypessimplevalue{T,false}()
        saferule = Convert.pyconvert_rule_ctypessimplevalue{T,true}()
        isptr = endswith(t, "_p")
        isreal = !isptr
        isfloat = t in ("float", "double")
        isint = isreal && !isfloat
        isuint = isint && (startswith(t, "u") || t == "size_t")

        t == "char_p" && pyconvert_add_rule(saferule, name, Cstring, Cstring)
        t == "wchar_p" && pyconvert_add_rule(saferule, name, Cwstring, Cwstring)
        pyconvert_add_rule(saferule, name, T, T)
        isuint && pyconvert_add_rule(sizeof(T) ≤ sizeof(UInt) ? saferule : rule, name, UInt, UInt)
        isuint && pyconvert_add_rule(sizeof(T) < sizeof(Int) ? saferule : rule, name, Int, Int)
        isint && !isuint && pyconvert_add_rule(sizeof(T) ≤ sizeof(Int) ? saferule : rule, name, Int, Int)
        isint && pyconvert_add_rule(rule, name, Integer, Integer)
        isfloat && pyconvert_add_rule(saferule, name, Float64, Float64)
        isreal && pyconvert_add_rule(rule, name, Real, Real)
        isreal && pyconvert_add_rule(rule, name, Number, Number)
        isptr && pyconvert_add_rule(saferule, name, Ptr, Ptr)
    end
    # numpy rules
    for (t, T) in Convert.NUMPY_SIMPLE_TYPES
        name = "numpy:$t"
        rule = Convert.pyconvert_rule_numpysimplevalue{T,false}()
        saferule = Convert.pyconvert_rule_numpysimplevalue{T,true}()
        isbool = occursin("bool", t)
        isint = occursin("int", t) || isbool
        isuint = occursin("uint", t) || isbool
        isfloat = occursin("float", t)
        iscomplex = occursin("complex", t)
        isreal = isint || isfloat
        isnumber = isreal || iscomplex

        pyconvert_add_rule(saferule, name, T, Any)
        isuint && pyconvert_add_rule(sizeof(T) ≤ sizeof(UInt) ? saferule : rule, name, UInt, UInt)
        isuint && pyconvert_add_rule(sizeof(T) < sizeof(Int) ? saferule : rule, name, Int, Int)
        isint && !isuint && pyconvert_add_rule(sizeof(T) ≤ sizeof(Int) ? saferule : rule, name, Int, Int)
        isint && pyconvert_add_rule(rule, name, Integer, Integer)
        isfloat && pyconvert_add_rule(saferule, name, Float64, Float64)
        isreal && pyconvert_add_rule(rule, name, Real, Real)
        iscomplex && pyconvert_add_rule(saferule, name, ComplexF64, ComplexF64)
        iscomplex && pyconvert_add_rule(rule, name, Complex, Complex)
        isnumber && pyconvert_add_rule(rule, name, Number, Number)
    end
    pyconvert_add_rule(Convert.pyconvert_rule_datetime64, "numpy:datetime64", NumpyDates.DateTime64, Any)
    pyconvert_add_rule(
        Convert.pyconvert_rule_datetime64,
        "numpy:datetime64",
        NumpyDates.InlineDateTime64,
        NumpyDates.InlineDateTime64,
    )
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
