# Final scalar, result-representation and inference findings

**Current update:** [convert-unwrapped-report.md](convert-unwrapped-report.md)
records minimal built-in result wrapping. Scalar allocation counts and public
inferred bounds remain unchanged; arrays/native containers lose their unnecessary
outer wrappers. Its focused validation passed 1,765 checks with one existing skip.

This report uses the final working tree, including source aliases/shorthand and
the agreed numeric-source policy. It supersedes the earlier scalar-optimization
snapshots. [convert-report.md](convert-report.md) contains implementation details,
validation status, environment and reproduction.

## Before/after cached conversion

Times are nanoseconds: median of three serial trial minima from the shared
convert-final.jl harness. Trial order was reversed in the middle round.
There were no concurrent tests. Both versions used Julia 1.13.2 and the same
Python 3.15.0 executable. Allocation figures are Julia bytes/counts.

| Target | Typed before ns | Typed after ns | Typed B / allocs before → after | Runtime-target before ns | Runtime-target after ns |
| --- | ---: | ---: | --- | ---: | ---: |
| Int | 223 | 240 | 0 / 0 → 0 / 0 | 341 | 360 |
| Integer | 590 | 242 | 0 / 0 → 0 / 0 | 705 | 361 |
| BigInt | 977 | 970 | 216 / 7 → 216 / 7 | 1140 | 1170 |
| Float64 | 240 | 240 | 0 / 0 → 0 / 0 | 352 | 362 |
| Bool | 232 | 254 | 0 / 0 → 0 / 0 | 349 | 369 |
| String | 297 | 311 | 48 / 2 → 48 / 2 | 406 | 429 |
| Nothing | 211 | 230 | 0 / 0 → 0 / 0 | 318 | 356 |
| Failure | 239 | 232 | 0 / 0 → 0 / 0 | 352 | 330 |

The typed caller carries the requested type in Val. The runtime-target caller
interpolates a Julia Type object, which can hide its identity from inference.
In the latter, both revisions allocate an additional 16-byte box for Float64
and for Int=10000; the value is outside Julia's small boxed-integer cache.
These caller effects must not be presented as conversion-only allocations.

The final implementation's cached Int, Float64 and Nothing paths allocate zero.
BigInt has returned to the baseline 216 bytes / seven allocations. The initial
redesign's Float64 dispatch box was removed by checking the type before crossing
the dispatch return boundary; the integer shortcut uses the shared raw kernel
rather than allowing Some{BigInt} to escape.

None is not slower than Int in the final paired measurements. The additional
interleaved scalar-only run produced typed minima of approximately 230–242 ns
for None, 243–250 ns for Int, and 250–251 ns for Float64. This run confirms
allocation behavior and variability; its timings are not mixed into the paired
before/after aggregate.

## Public try-conversion and inference

The final scalar-only run measured pytryconvert at zero allocations for Int,
Float64 and Nothing. BigInt try-conversion uses 232 bytes / eight allocations:
the additional 16-byte Some is part of the public success representation.
Ordinary pyconvert(BigInt, ...) uses 216 / seven.

Final inferred public return bounds in convert-inference-final.txt:

| Requested type | Return bound |
| --- | --- |
| Int | Union{Nothing,Some{Int}} |
| Integer | Union{Nothing,Some{T} where T<:Integer} |
| Union{Int,Nothing} | Union{Nothing,Some{Nothing},Some{Int}} |
| Any | Union{Nothing,Some} |

Raw scalar dispatch remains bounded by the concrete value type and the private
failure sentinel. The Float64 allocation in the profiling artifact is at the
unspecialized profiling caller, not inside scalar dispatch. There is no
Some{BigInt} object in the ordinary conversion profile.

## Result representation comparison

convert-results.jl models the original Union{T,Unconverted} representation and
Some/nothing with specialized result bounds. It covers successful and failed
calls, direct results, results escaping through Ref{Any}, and typed loops.
These deliberately dynamic/escaping harnesses test return boundaries; their
figures are not lower bounds on optimized Some construction.

Successful escaping model results, including the common Ref{Any} allocation:

| Target | Raw B / allocs | Some B / allocs |
| --- | ---: | ---: |
| Int64 | 16 / 1 | 32 / 2 |
| Float64 | 48 / 3 | 48 / 3 |
| String | 16 / 1 | 48 / 3 |
| Vector{Int} | 16 / 1 | 48 / 3 |
| Integer | 16 / 1 | 32 / 2 |
| AbstractArray | 16 / 1 | 48 / 3 |
| Union{Int64,Nothing} | 16 / 1 | 32 / 2 |
| Any | 16 / 1 | 32 / 2 |

Both 100-element typed loops allocate zero (roughly 288 ns raw versus 268 ns
Some in this short run). Do not interpret that small timing difference as a
general representation advantage. Full successful/failed and escaping/direct
measurements are in convert-results-final.tsv.

The measurements justify the ergonomic Some/nothing API with optimized raw
internal dispatch. The earlier all-Some internal experiment introduced
per-element allocations and slowed large-vector conversion; its historical
artifacts are retained, while final container results are in the main report.

## Numeric-source policy and validation

Generic numbers.Real/Complex fallback rules now apply only to Python
float/complex sources. Integral and Rational sources retain exact intermediates;
supported NumPy scalars retain dedicated rules. Other numeric types require
explicit rules. An out-of-range integer can no longer decline exact conversion
and then succeed after float rounding.

Tests include throwing __float__/__complex__ methods on integer/rational/custom
numeric sources, nonintegral fractions, float/complex subclasses, explicit rule
opt-in, NumPy scalars, singleton-rule mutations and inferred scalar returns.
Latest local validation: 605 conversion checks and 15 ordinary Python tests
passed, with no broken conversion checks. Final full suites and supported
version coverage are assigned to CI.

## Artifacts and reproduction

Use benchmark/run-convert-final.sh and summarize-convert-final.py as described
in the main report. The paired results live in convert-final-{before,after}-r*,
with source/environment records. The scalar-only diagnostic run uses
convert-scalars-final.tsv, -repeat.tsv, -inference.txt and -allocations.txt.
Representation and public inference outputs are convert-results-final.tsv and
convert-inference-final.txt. Earlier before/optimized/source-policy files are
historical snapshots, not current failures or current comparison tables.
