const pyjltypetype = pynew()

function pyjltype_getitem(self::Type, k_)
    if pyistuple(k_)
        k = pyconvert(Vector{Any}, k_)
        pydel!(k_)
        Py(self{k...})
    else
        k = pyconvert(Any, k_)
        Py(self{k})
    end
end

const PYNUMPYDTYPE = IdDict{Type,Py}()

function pyjltype_numpy_dtype(self::Type)
    ans = get!(PYNUMPYDTYPE, self) do
        typestr, descr = pytypestrdescr(self)
        # unsupported type
        if typestr == ""
            return PyNULL
        end
        np = pyimport("numpy")
        # simple scalar type
        if pyisnull(descr)
            return np.dtype(typestr)
        end
        # We could juse use np.dtype(descr), but when there is padding, np.dtype(descr)
        # changes the names of the padding fields from "" to "f{N}". Using this other
        # dtype constructor avoids this issue and preserves the invariant:
        #   np.dtype(eltype(array)) == np.array(array).dtype
        names = []
        formats = []
        offsets = []
        for i = 1:fieldcount(self)
            nm = fieldname(self, i)
            push!(names, nm isa Integer ? "f$(nm-1)" : String(nm))
            ts, ds = pytypestrdescr(fieldtype(self, i))
            push!(formats, pyisnull(ds) ? ts : ds)
            push!(offsets, fieldoffset(self, i))
        end
        return np.dtype(
            pydict(
                names = pylist(names),
                formats = pylist(formats),
                offsets = pylist(offsets),
                itemsize = sizeof(self),
            ),
        )
    end
    if pyisnull(ans)
        errset(pybuiltins.AttributeError, "__numpy_dtype__")
    end
    return ans
end

function init_type()
    jl = pyjuliacallmodule
    pybuiltins.exec(
        pybuiltins.compile(
            """
$("\n"^(@__LINE__()-1))
class TypeValue(AnyValue):
    __slots__ = ()
    def __getitem__(self, k):
        return self._jl_callmethod($(pyjl_methodnum(pyjltype_getitem)), k)
    def __setitem__(self, k, v):
        raise TypeError("not supported")
    def __delitem__(self, k):
        raise TypeError("not supported")
    @property
    def __numpy_dtype__(self):
        return self._jl_callmethod($(pyjl_methodnum(pyjltype_numpy_dtype)))
""",
            @__FILE__(),
            "exec",
        ),
        jl.__dict__,
    )
    pycopy!(pyjltypetype, jl.TypeValue)
end

pyjltype(::Type) = pyjltypetype
