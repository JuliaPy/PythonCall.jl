const GC_ENABLED = Ref(true)
const GC_QUEUE = PyPtr[]

function gc_disable()
    GC_ENABLED[] = false
    return
end

function gc_enable()
    GC_ENABLED[] = true
    if !isempty(GC_QUEUE)
        with_gil(false) do
            for ptr in GC_QUEUE
                if ptr != PyNULL
                    Py_DecRef(ptr)
                end
            end
        end
    end
    empty!(GC_QUEUE)
    return
end

function gc_enqueue(ptr::PyPtr)
    if ptr != PyNULL && CTX.is_initialized
        if GC_ENABLED[]
            with_gil(false) do
                Py_DecRef(ptr)
            end
        else
            push!(GC_QUEUE, ptr)
        end
    end
    return
end

function gc_enqueue_all(ptrs)
    if CTX.is_initialized
        if GC_ENABLED[]
            with_gil(false) do
                for ptr in ptrs
                    if ptr != PyNULL
                        Py_DecRef(ptr)
                    end
                end
            end
        else
            append!(GC_QUEUE, ptrs)
        end
    end
    return
end
