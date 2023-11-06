"""
Garbage collection of Python objects.

See [`enable`](@ref) and [`gc`](@ref).
"""
module GC

using .._Py: C

const ENABLED = Ref(true)
const QUEUE = C.PyPtr[]

"""
    PythonCall.GC.enable(on::Bool)

Control whether garbage collection of Python objects is turned on or off.

Return the previous GC state.

Disabling the GC means that whenever a Python object owned by Julia is finalized, it is not
immediately freed but is instead added to a queue of objects to free later when GC is
re-enabled.

Like most PythonCall functions, you must only call this from the main thread.
"""
function enable(on::Bool)
    was_on = ENABLED[]
    if on
        ENABLED[] = true
        if !was_on
            gc()
        end
    else
        ENABLED[] = false
    end
    return ans
end

"""
    PythonCall.GC.gc()

Perform garbage collection of Python objects.

Like most PythonCall functions, you must only call this from the main thread.
"""
function gc()
    if !isempty(QUEUE)
        C.with_gil(false) do
            for ptr in QUEUE
                if ptr != C.PyNULL
                    C.Py_DecRef(ptr)
                end
            end
        end
        empty!(QUEUE)            
    end
end

function enqueue(ptr::C.PyPtr)
    if ptr != C.PyNULL && C.CTX.is_initialized
        if ENABLED[]
            C.with_gil(false) do
                C.Py_DecRef(ptr)
            end
        else
            push!(QUEUE, ptr)
        end
    end
    return
end

function enqueue_all(ptrs)
    if C.CTX.is_initialized
        if ENABLED[]
            C.with_gil(false) do
                for ptr in ptrs
                    if ptr != C.PyNULL
                        C.Py_DecRef(ptr)
                    end
                end
            end
        else
            append!(QUEUE, ptrs)
        end
    end
    return
end

end # module GC
