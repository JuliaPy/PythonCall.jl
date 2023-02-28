# FAQ & Troubleshooting

## Heap corruption when using PyTorch ([issue 215](https://github.com/cjdoris/PythonCall.jl/issues/215))

On some systems, you may see an error like the following when using `torch` and `juliacall`:
```text
Python(65251,0x104cf8580) malloc: Heap corruption detected, free list is damaged at 0x600001c17280
*** Incorrect guard value: 1903002876
Python(65251,0x104cf8580) malloc: *** set a breakpoint in malloc_error_break to debug
[1]    65251 abort      ipython
```

A solution is to ensure that `juliacall` is imported before `torch`.

## `ccall requries the compiler` error when importing some Python libraries
On some systems, you may see an error like the following when import e.g. `matplotlib` before `juliacall`:

```
ERROR: `ccall` requires the compilerTraceback (most recent call last):
  File "/home/dingraha/projects/pythoncall_import_error/run.py", line 2, in <module>
    from foo import Foo
  File "/home/dingraha/projects/pythoncall_import_error/foo.py", line 4, in <module>
    import juliacall; jl = juliacall.newmodule("FooModule")
  File "/home/dingraha/projects/pythoncall_import_error/venv/lib/python3.9/site-packages/juliacall/__init__.py", line 218, in <module>
    init()
  File "/home/dingraha/projects/pythoncall_import_error/venv/lib/python3.9/site-packages/juliacall/__init__.py", line 214, in init
    raise Exception('PythonCall.jl did not start properly')
Exception: PythonCall.jl did not start properly
```

As described in [issue 255](https://github.com/cjdoris/PythonCall.jl/issues/255), the likely problem is that the "other" Python library (`matplotlib`, whatever) is loading the system `libstdc++.so`, which isn't compatible with the `libstdc++.so` that Julia ships with.
Linux distributions with older `libstdc++` versions seem more likely to suffer from this issue.
The solution is to either:

  * use a Linux distribution with a more recent `libstdc++`
  * import `juliacall` before the other Python library, so that Julia's `libstdc++` is loaded
  * use a Python from a conda environment, which will have a newer `libstdc++` that is compatible with Julia's
