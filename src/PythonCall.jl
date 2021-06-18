module PythonCall

include("cpython/CPython.jl")
include("Py.jl")
include("err.jl")
include("config.jl")
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
# misc
include("with.jl")
include("multimedia.jl")

function __init__()
    init_consts()
end

end
