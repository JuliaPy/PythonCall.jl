struct PySimpleCData_TryConvert_value{R,tr} end

(::PySimpleCData_TryConvert_value{R,tr})(o, ::Type{T}, ::Type{S}) where {T,S,R,tr} = begin
    ptr = PySimpleObject_GetValue(o, Ptr{R})
    val = unsafe_load(ptr)
    putresult(T, tr ? tryconvert(S, val) : convert(S, val))
end
