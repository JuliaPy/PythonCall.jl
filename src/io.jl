const pyiomodule = PyLazyObject(() -> pyimport("io"))
const pyiobasetype = PyLazyObject(() -> pyiomodule.IOBase)
const pyrawiobasetype = PyLazyObject(() -> pyiomodule.RawIOBase)
const pybufferediobasetype = PyLazyObject(() -> pyiomodule.BufferedIOBase)
const pytextiobasetype = PyLazyObject(() -> pyiomodule.TextIOBase)
const pytextiowrappertype = PyLazyObject(() -> pyiomodule.TextIOWrapper)
const pyiounsupportedoperation = PyLazyObject(() -> pyiomodule.UnsupportedOperation)

pybufferedio(io::IO) = pyjuliabufferedio(io)
pytextio(io::IO) = pyjuliatextio(io)
export pybufferedio, pytextio
