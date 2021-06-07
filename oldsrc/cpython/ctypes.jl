struct PySimpleCData_TryConvert_value{R,tr} end

(::PySimpleCData_TryConvert_value{R,tr})(o, ::Type{S}) where {S,R,tr} = begin
    ptr = PySimpleObject_GetValue(o, Ptr{R})
    val = unsafe_load(ptr)
    putresult(tr ? tryconvert(S, val) : convert(S, val))
end
