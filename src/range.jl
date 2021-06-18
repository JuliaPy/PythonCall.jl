pyrange(x, y, z) = pyrangetype(x, y, z)
pyrange(x, y) = pyrangetype(x, y)
pyrange(y) = pyrangetype(y)
export pyrange

pyrange_fromrange(x::AbstractRange) = pyrange(first(x), last(x) + sign(step(x)), step(x))
