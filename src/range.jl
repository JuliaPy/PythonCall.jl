pyrange(x, y, z) = pybuiltins.rangetype(x, y, z)
pyrange(x, y) = pybuiltins.rangetype(x, y)
pyrange(y) = pybuiltins.rangetype(y)
export pyrange

pyrange_fromrange(x::AbstractRange) = pyrange(first(x), last(x) + sign(step(x)), step(x))
