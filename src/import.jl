pyimport(m::AbstractString) = check(C.PyImport_ImportModule(m))
pyimport(m) = check(C.PyImport_Import(pyobject(m)))
export pyimport
