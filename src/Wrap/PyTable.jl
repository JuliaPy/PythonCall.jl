module PyTables

using ...PythonCall

import ...PythonCall: PyTable

PyTable(x) = pyconvert(PyTable, x)

# Tables.istable(x::Py) = Tables.istable(@pyconvert(PyTable, x, return false))
# Tables.rowaccess(x::Py) = Tables.rowaccess(@pyconvert(PyTable, x, return false))
# Tables.rows(x::Py) = Tables.rows(pyconvert(PyTable, x))
# Tables.columnaccess(x::Py) = Tables.columnaccess(@pyconvert(PyTable, x, return false))
# Tables.columns(x::Py) = Tables.columns(pyconvert(PyTable, x))
# Tables.materializer(x::Py) = Tables.materializer(pyconvert(PyTable, x))

end
