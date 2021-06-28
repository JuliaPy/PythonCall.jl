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
        pynew()
    else
        r, newst = z
        self.st = Some(newst)
        Py(r)
    end
end

function init_jlwrap_iter()
    jl = pyjuliacallmodule
    filename = "$(@__FILE__):$(1+@__LINE__)"
    pybuiltins.exec(pybuiltins.compile("""
    class IteratorValue(ValueBase):
        __slots__ = ()
        __module__ = "juliacall"
        def __iter__(self):
            return self
        def __next__(self):
            return self._jl_callmethod($(pyjl_methodnum(pyjliter_next)))
    """, filename, "exec"), jl.__dict__)
    pycopy!(pyjlitertype, jl.IteratorValue)
end

pyjl(v::Iterator) = pyjl(pyjlitertype, v)
