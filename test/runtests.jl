using TestItemRunner
using CondaPkg
CondaPkg.add("pandas")
@run_package_tests verbose=true
