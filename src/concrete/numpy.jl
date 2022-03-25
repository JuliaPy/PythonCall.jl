struct pyconvert_rule_numpysimplevalue{R,S} <: Function end

function (::pyconvert_rule_numpysimplevalue{R,SAFE})(::Type{T}, x::Py) where {R,SAFE,T}
    ans = GC.@preserve x C.PySimpleObject_GetValue(R, getptr(x))
    if SAFE
        pyconvert_return(convert(T, ans))
    else
        pyconvert_tryconvert(T, ans)
    end
end

const NUMPY_SIMPLE_TYPES = [
    ("int8", Int8),
    ("int16", Int16),
    ("int32", Int32),
    ("int64", Int64),
    ("uint8", UInt8),
    ("uint16", UInt16),
    ("uint32", UInt32),
    ("uint64", UInt64),
    ("float16", Float16),
    ("float32", Float32),
    ("float64", Float64),
    ("complex32", ComplexF16),
    ("complex64", ComplexF32),
    ("complex128", ComplexF64),
]

function init_numpy()
    for (t,T) in NUMPY_SIMPLE_TYPES
        isint = occursin("int", t)
        isuint = occursin("uint", t)
        isfloat = occursin("float", t)
        iscomplex = occursin("complex", t)
        isreal = isint || isfloat
        isnumber = isreal || iscomplex

        name = "numpy:$t"
        rule = pyconvert_rule_numpysimplevalue{T, false}()
        saferule = pyconvert_rule_numpysimplevalue{T, true}()

        pyconvert_add_rule(name, T, saferule, PYCONVERT_PRIORITY_ARRAY)
        isuint && pyconvert_add_rule(name, UInt, sizeof(T) ≤ sizeof(UInt) ? saferule : rule)
        isuint && pyconvert_add_rule(name, Int, sizeof(T) < sizeof(Int) ? saferule : rule)
        isint && !isuint && pyconvert_add_rule(name, Int, sizeof(T) ≤ sizeof(Int) ? saferule : rule)
        isint && pyconvert_add_rule(name, Integer, rule)
        isfloat && pyconvert_add_rule(name, Float64, saferule)
        isreal && pyconvert_add_rule(name, Real, rule)
        iscomplex && pyconvert_add_rule(name, ComplexF64, saferule)
        iscomplex && pyconvert_add_rule(name, Complex, rule)
        isnumber && pyconvert_add_rule(name, Number, rule)
    end
end
