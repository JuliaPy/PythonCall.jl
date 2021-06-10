"""
    module HPython

This provides a simplified interface to the Python C API.

It is still unsafe in that reference counting is up to the user, but has some features to
make this less error-prone and easier to debug.

It is very loosely based on HPy, where the H stands for "handle".

Where the C API would return a Cint, we return its logical interpretation as Hvoid (nothing
or error) or Hbool (bool or error) or Cint (int or error) or ... This simplifies error
checking.
"""
module HPython

import ..CPython
const C = CPython

const DEBUG = true # get(ENV, "JULIA_PYTHONCALL_DEBUG", "no") == "yes"

include("forwarddefs.jl")
include("context.jl")
include("handle.jl")
include("autohandle.jl")
include("orerr.jl")
include("err.jl")
# abstract interfaces
include("object.jl")
include("iter.jl")
include("builtins.jl")
# concrete types
include("str.jl")
include("bytes.jl")
include("tuple.jl")

end
