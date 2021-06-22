pyimport(m) = pynew(errcheck(@autopy m C.PyImport_Import(getptr(m_))))
export pyimport
