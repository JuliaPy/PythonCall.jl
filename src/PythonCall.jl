module PythonCall

using MacroTools

include("utils.jl")

include("cpython/CPython.jl")

include("Py.jl")
include("err.jl")
include("config.jl")
include("convert.jl")
# abstract interfaces
include("object.jl")
include("iter.jl")
include("import.jl")
include("number.jl")
# concrete types
include("consts.jl")
include("str.jl")
include("bytes.jl")
include("tuple.jl")
include("list.jl")
include("dict.jl")
include("bool.jl")
include("int.jl")
include("float.jl")
include("complex.jl")
include("set.jl")
include("slice.jl")
include("range.jl")
include("none.jl")
include("type.jl")
# misc
include("with.jl")
include("multimedia.jl")
include("pyconst_macro.jl")
include("py_macro.jl")

function __init__()
    init_consts()
    init_pyconvert()
end

end
