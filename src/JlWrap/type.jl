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
    m = pyimport("juliacall._jlwrap.type")
    methods = m._jl_methods
    methods["__getitem__"] = pyjl_methodnum(pyjltype_getitem)
    methods["__numpy_dtype__"] = pyjl_methodnum(pyjltype_numpy_dtype)
    pycopy!(pyjltypetype, jl.TypeValue)
end

pyjltype(::Type) = pyjltypetype
