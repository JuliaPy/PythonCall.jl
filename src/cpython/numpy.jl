struct PyNumpySimpleData_TryConvert_value{R,tr} end

(::PyNumpySimpleData_TryConvert_value{R,tr})(o, ::Type{T}, ::Type{S}) where {T,S,R,tr} = begin
    val = PySimpleObject_GetValue(o, R)
    putresult(T, tr ? tryconvert(S, val) : convert(S, val))
end
