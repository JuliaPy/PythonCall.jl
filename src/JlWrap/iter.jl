mutable struct Iterator
    val::Any
    st::Any
end
Iterator(x) = Iterator(x, nothing)
Base.length(x::Iterator) = length(x.val)

const pyjlitertype = pynew()

function pyjliter_next(self::Iterator)
    val = self.val
    st = self.st
    if st === nothing
        z = iterate(val)
    else
        z = iterate(val, something(st))
    end
    if z === nothing
        errset(pybuiltins.StopIteration)
        PyNULL
    else
        r, newst = z
        self.st = Some(newst)
        Py(r)
    end
end

function init_iter()
    jl = pyjuliacallmodule
    m = pyimport("juliacall._jlwrap.iter")
    methods = m._jl_methods
    methods["__next__"] = pyjl_methodnum(pyjliter_next)
    pycopy!(pyjlitertype, jl.IteratorValue)
end

pyjltype(::Iterator) = pyjlitertype
