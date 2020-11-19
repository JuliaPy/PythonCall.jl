pycontains(o, v) = check(Bool, C.PySequence_Contains(pyobject(o), pyobject(v)))
export pycontains
