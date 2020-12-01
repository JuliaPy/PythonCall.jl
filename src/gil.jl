"""
    gil_is_on()

True if we own the GIL. If so, we can safely interact with Python.
"""
gil_is_on() = CONFIG.gilstate == C_NULL

"""
    gil_on()

Ensure the GIL is on, so that we can safely interact with Python. Return true if the GIL was already on.
"""
gil_on() =
    if gil_is_on()
        true
    else
        C.PyEval_RestoreThread(CONFIG.gilstate)
        CONFIG.gilstate = C_NULL
        false
    end

"""
    gil_off()

Ensure the GIL is off. Return true if the GIL was on.

After calling this, you must not call any Python functions until `gil_on()` is called. (It is OK if the garbage collector frees Python objects with the GIL off: the finalizer will temporarily acquire it.)
"""
gil_off() =
    if gil_is_on()
        CONFIG.gilstate = C.PyEval_SaveThread()
        true
    else
        false
    end
