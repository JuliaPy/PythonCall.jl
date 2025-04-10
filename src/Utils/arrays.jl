# TODO: what is the best way?
ismutablearray(x::Array) = true
ismutablearray(x::AbstractArray) = begin
    p = parent(x)
    p === x ? false : ismutablearray(p)
end

size_to_fstrides(elsz::Integer, sz::Tuple{Vararg{Integer}}) =
    isempty(sz) ? () : (elsz, size_to_fstrides(elsz * sz[1], sz[2:end])...)

size_to_cstrides(elsz::Integer, sz::Tuple{Vararg{Integer}}) =
    isempty(sz) ? () : (size_to_cstrides(elsz * sz[end], sz[1:end-1])..., elsz)
