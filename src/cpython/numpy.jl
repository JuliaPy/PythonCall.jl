struct PyNumpySimpleData_TryConvert_value{R,tr} end

(::PyNumpySimpleData_TryConvert_value{R,tr})(o, ::Type{S}) where {S,R,tr} = begin
    val = PySimpleObject_GetValue(o, R)
    putresult(tr ? tryconvert(S, val) : convert(S, val))
end
