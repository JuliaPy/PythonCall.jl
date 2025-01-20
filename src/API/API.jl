"The version of PythonCall."
const VERSION = v"0.9.23"

# types
include("types.jl")

# submodules
include("GIL.jl")
include("GC.jl")

# functions
function python_executable_path end
function python_library_handle end
function python_library_path end
function python_version end
function ispy end
function pynew end
function pyisnull end
function pycopy! end
function getptr end
function pydel! end
function unsafe_pynext end
function pyconvert_add_rule end
function pyconvert_return end
function pyconvert_unconverted end
function event_loop_on end
function event_loop_off end
function fix_qt_plugin_path end
function pyis end
function pyrepr end
function pyascii end
function pyhasattr end
function pygetattr end
function pysetattr end
function pydelattr end
function pyissubclass end
function pyisinstance end
function pyhash end
function pytruth end
function pynot end
function pylen end
function pyhasitem end
function pygetitem end
function pysetitem end
function pydelitem end
function pydir end
function pycall end
function pyeq end
function pyne end
function pyle end
function pylt end
function pyge end
function pygt end
function pycontains end
function pyin end
function pyneg end
function pypos end
function pyabs end
function pyinv end
function pyindex end
function pyadd end
function pysub end
function pymul end
function pymatmul end
function pyfloordiv end
function pytruediv end
function pymod end
function pydivmod end
function pylshift end
function pyrshift end
function pyand end
function pyxor end
function pyor end
function pyiadd end
function pyisub end
function pyimul end
function pyimatmul end
function pyifloordiv end
function pyitruediv end
function pyimod end
function pyilshift end
function pyirshift end
function pyiand end
function pyixor end
function pyior end
function pypow end
function pyipow end
function pyiter end
function pynext end
function pybool end
function pystr end
function pybytes end
function pyint end
function pyfloat end
function pycomplex end
function pytype end
function pyslice end
function pyrange end
function pytuple end
function pylist end
function pycollist end
function pyrowlist end
function pyset end
function pyfrozenset end
function pydict end
function pydate end
function pytime end
function pydatetime end
function pyfraction end
function pyeval end
function pyexec end
function pywith end
function pyimport end
function pyprint end
function pyhelp end
function pyall end
function pyany end
function pycallable end
function pycompile end

# macros
macro pyeval end
macro pyexec end
macro pyconst end

# exports
export Py
export PyException
export ispy
export pyis
export pyrepr
export pyascii
export pyhasattr
export pygetattr
export pysetattr
export pydelattr
export pyissubclass
export pyisinstance
export pyhash
export pytruth
export pynot
export pylen
export pyhasitem
export pygetitem
export pysetitem
export pydelitem
export pydir
export pycall
export pyeq
export pyne
export pyle
export pylt
export pyge
export pygt
export pycontains
export pyin
export pyneg
export pypos
export pyabs
export pyinv
export pyindex
export pyadd
export pysub
export pymul
export pymatmul
export pyfloordiv
export pytruediv
export pymod
export pydivmod
export pylshift
export pyrshift
export pyand
export pyxor
export pyor
export pyiadd
export pyisub
export pyimul
export pyimatmul
export pyifloordiv
export pyitruediv
export pyimod
export pyilshift
export pyirshift
export pyiand
export pyixor
export pyior
export pypow
export pyipow
export pyiter
export pynext
export pybool
export pystr
export pybytes
export pyint
export pyfloat
export pycomplex
export pytype
export pyslice
export pyrange
export pytuple
export pylist
export pycollist
export pyrowlist
export pyset
export pyfrozenset
export pydict
export pydate
export pytime
export pydatetime
export pyfraction
export pyeval
export pyexec
export @pyeval
export @pyexec
export pywith
export pyimport
export pyprint
export pyhelp
export pyall
export pyany
export pycallable
export pycompile
export pybuiltins
export @pyconst

# public bindings
if Base.VERSION ≥ v"1.11"
    eval(
        Expr(
            :public,
            :VERSION,
            :GIL,
            :GC,
            :CONFIG,
            # :PyNULL,
            # :PYCONVERT_PRIORITY_WRAP,
            # :PYCONVERT_PRIORITY_ARRAY,
            # :PYCONVERT_PRIORITY_CANONICAL,
            # :PYCONVERT_PRIORITY_NORMAL,
            # :PYCONVERT_PRIORITY_FALLBACK,
            :python_executable_path,
            :python_library_handle,
            :python_library_path,
            :python_version,
            :pynew,
            :pyisnull,
            :pycopy!,
            :getptr,
            :pydel!,
            :unsafe_pynext,
            :pyconvert_add_rule,
            :pyconvert_return,
            :pyconvert_unconverted,
            :event_loop_on,
            :event_loop_off,
            :fix_qt_plugin_path,
        ),
    )
end
