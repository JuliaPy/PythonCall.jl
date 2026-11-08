# Minimal rule-result wrapping

The optimization succeeds: ordinary built-in rules now return their values
directly, wrapping only successful nothing/Some values. Public pytryconvert
continues to wrap every success. No registry, ordering, scope, preference or
failure/exception policy changed, and no additional fast path was introduced.

This report supersedes affected allocation/timing results in the earlier full
benchmark refresh. Its baseline is the implementation immediately before this
optimization, **not original v1**.

## Change and validation

The internal success helper returns ordinary values directly; its Nothing/Some
method wraps them once. Internal result readers accept both forms. Checked Julia
conversion uses this encoding, and vector/dictionary/integer rules no longer
construct unconditional Some results. Existing array builders/rules already use
these helpers, so they benefit without an array-specific implementation.

Only src/Convert/pyconvert.jl and rules.jl change at runtime. The isolated diff
is saved as convert-unwrapped.patch. Source-author documentation now recommends
ordinary success values. Tests exercise successful nothing, nested Some,
wrapped Julia values and nested native containers, preserving exact values.

Focused conversion, Python-wrapper and Julia-wrapper tests: **1,765 passed,
one existing optional-dependency skip** (the NumPy copy=False test is gated by
Setup.devdeps with the Null backend). Full suites/version coverage remain for CI.

Inferred public try-conversion bounds remain unchanged for Int, Integer,
Union{Int,Nothing} and Any. The generic array rule now infers
Union{Nothing,Array}, rather than an existential Some-wrapped array. Arrays
still have runtime-discovered element/rank types, but their existing object
references can cross that boundary directly.

## Method

Three sequential before/after rounds on Julia 1.13.2, Python 3.15.0, NumPy 2.5.4
and BenchmarkTools 1.8.0, with no concurrent tests. The saved pre-change source
was developed into a separate environment. Round order: before/after,
after/before, before/after. Both use the same Python executable/Null backend.

Times are microseconds: median of three trial minima, using up to 5,000 samples /
one second per warm trial. Julia bytes/counts exclude Python allocations.
Source paths/fingerprints, trial medians and round ranges are retained. These
short runs do not establish confidence intervals. The unchanged DAG/cold-path
mechanism was not remeasured in this follow-up.

## Arrays

| Case | Before us | After us | Change | Before B / allocs | After B / allocs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Buffer-only native | 7.325 | 7.150 | -2.4% | 8856 / 26 | 8712 / 23 |
| Buffer-only wrapper | 5.583 | 5.283 | -5.4% | 768 / 22 | 640 / 20 |
| Interface-only native | 22.200 | 21.700 | -2.3% | 9368 / 55 | 9224 / 52 |
| Array-method-only native | 8.133 | 7.633 | -6.1% | 9016 / 30 | 8872 / 27 |
| ndarray native | 4.867 | 4.686 | -3.7% | 8888 / 23 | 8744 / 20 |
| ndarray wrapper | 3.388 | 3.288 | -3.0% | 800 / 19 | 672 / 17 |

All six cases remove the predicted 128 bytes / two objects for wrappers,
144 bytes / three objects for native arrays. Their allocation counts/bytes
now match the original-v1 protocol fixtures. This does not establish timing
equivalence to original v1; this experiment isolates the newer wrapper change.

Profile.Allocs confirms the disappearance of Some{PyArray{...}}, the additional
boxed PyArray during unwrapping, and Some{Vector{T}} for native copying. The
post-change profile contains no Some objects for the four profiled array cases.
See convert-array-allocations-unwrapped.txt; the earlier pre-change stacks are
in convert-array-allocations-after.txt.

## Other containers and scalars

| Case | Before us | After us | Change | Before B / allocs | After B / allocs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Vector, 10 integers | 7.120 | 6.560 | -7.9% | 720 / 18 | 704 / 17 |
| Vector, 10,000 integers | 4714.836 | 4624.435 | -1.9% | 513880 / 19508 | 513864 / 19507 |
| Vector{Int}, 10,000 integers | 4084.732 | 4036.031 | -1.2% | 362024 / 10018 | 362008 / 10017 |
| Nested concrete vectors | 6.950 | 6.920 | -0.4% | 768 / 29 | 720 / 26 |
| Nested Any conversion | 10.800 | 9.800 | -9.3% | 864 / 32 | 816 / 29 |
| Dict, inferred values | 6.480 | 6.540 | +0.9% | 1280 / 24 | 1264 / 23 |
| Dict{String,Float64} | 3.675 | 3.613 | -1.7% | 672 / 16 | 656 / 15 |
| Empty Vector | 1.240 | 1.130 | -8.9% | 112 / 5 | 96 / 4 |
| Empty Dict | 1.040 | 1.036 | -0.3% | 208 / 5 | 192 / 4 |

Each native container loses its outer 16-byte result wrapper; nested vectors
lose one per converted vector. Timing changes vary, including the slightly
slower inferred dictionary despite fewer allocations. Do not infer a fixed
speedup from removing one allocation.

Cached Int/Float64/Nothing remain allocation-free; BigInt remains 216 bytes /
seven allocations. Their typed minima before/after are 241/240 ns, 240/237 ns,
232/232 ns and 935/980 ns respectively. BigInt's +4.8% and incompatible-string
failure's +3.4% remain reported in the summary; round ranges overlap and no
additional allocations or inferred-type regressions were found. Other small
scalar timing changes are similarly not treated as established improvements.

## Artifacts and reproduction

Current data: convert-final-unwrapped-{before,after}-r{1,2,3}.tsv and
convert-expanded-unwrapped-{before,after}-r{1,2,3}.tsv, each with an environment
record. The aggregate is convert-unwrapped-summary.tsv (43 cases/modes).
Both revisions passed all benchmark fixture assertions. Allocation differences
are constant across all three rounds.

Reconstruct the pre-change source by copying the current checkout and reversing
convert-unwrapped.patch **inside that copy**. Develop it into an isolated Julia
environment with BenchmarkTools 1.8.0. For each revision/round, sequentially run:

~~~sh
julia --project=/path/to/revision-env benchmark/convert-final.jl unwrapped-before-r1
julia --project=/path/to/revision-env benchmark/convert-expanded.jl unwrapped-before-r1 warm
~~~

Use labels before/after and rounds 1/2/3, with the order described above; keep
the same installed Python/backend for both revisions. Then aggregate/profile:

~~~sh
python3 benchmark/summarize-convert-final.py unwrapped
julia --project=benchmark benchmark/convert-array-allocations.jl unwrapped
~~~
