pyimport(m::AbstractString) = cpycall_obj(Val(:PyImport_ImportModule), m)
pyimport(m) = cpycall_obj(Val(:PyImport_Import), pyobject(m))
export pyimport
