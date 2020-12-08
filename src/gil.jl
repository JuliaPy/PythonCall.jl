function with_gil(f)
    if CONFIG.isembedded
        g = C.PyGILState_Ensure()
        try
            f()
        finally
            C.PyGILState_Release(g)
        end
    else
        f()
    end
end
