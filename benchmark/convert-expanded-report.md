# Final rule dispatch, protocols and preferences

**Current update:** [convert-unwrapped-report.md](convert-unwrapped-report.md)
records the subsequent optimization: all extra Some/reboxing allocations below
are removed, with approximately 2–6% faster array conversions in its direct
before/after experiment. The tables below retain the earlier full original-v1
comparison, including cold-path measurements.

These results cover the final working tree, including public source aliases,
built-in shorthand and the numeric-source policy. They supersede the earlier
single-run table. See [convert-report.md](convert-report.md) for environment,
implementation, final validation status and reproduction.

## Fixtures and method

The same harness ran on original v1 commit
25f82cc8effbd6c487ddceaffc34aa7cbe86505b and current v1. Three before/after rounds
ran sequentially with the middle round reversed. No tests ran concurrently.
Each process records loaded source paths/hashes and environment versions.

The buffer-only fixture is array.array('d', ...): it has the C buffer protocol
and none of the array attributes. Assertions verify the first selected converter
is the protocol array converter on both revisions. Interface-only and
__array__-only fixtures lack buffers. Each array has 1,000 elements, and shape,
element type and endpoint values are checked outside timing.

Synthetic classes exercise one rule, decline/success, three declines/success,
destination scope filtering and virtual collections.abc.Mapping membership.
Eligible counts are checked: 1/2/4 on both revisions; scoped filtering changes
17 callbacks to one eligible callback. There are 27 synthetic/scoped registrations.
Every input is created before timing; requested types remain visible through Val.

All times are microseconds: median of three trial minima. Warm trials use up to
5,000 samples / one second. Julia allocations exclude Python allocations. Raw
minimum/median data and round ranges are in convert-final-summary.tsv and
convert-expanded-final-{before,after}-r{1,2,3}.tsv. No confidence intervals are claimed.

## Warm conversion

| Case | Before us | After us | Change | Before B / allocs | After B / allocs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Single rule | 0.393 | 0.378 | -4.0% | 0 / 0 | 0 / 0 |
| Decline then success | 0.535 | 0.540 | +1.0% | 0 / 0 | 0 / 0 |
| Three declines then success | 0.979 | 0.936 | -4.4% | 0 / 0 | 0 / 0 |
| Sixteen scoped decliners | 4.071 | 0.385 | -90.6% | 0 / 0 | 0 / 0 |
| Virtual Mapping ABC | 0.394 | 0.379 | -3.7% | 0 / 0 | 0 / 0 |
| Buffer-only native | 7.000 | 7.475 | +6.8% | 8712 / 23 | 8856 / 26 |
| Buffer-only wrapper | 5.433 | 5.620 | +3.4% | 640 / 20 | 768 / 22 |
| Interface-only native | 21.199 | 22.600 | +6.6% | 9224 / 52 | 9368 / 55 |
| Array-method-only native | 7.633 | 8.000 | +4.8% | 8872 / 27 | 9016 / 30 |
| ndarray native | 4.471 | 4.816 | +7.7% | 8744 / 20 | 8888 / 23 |
| ndarray wrapper | 3.200 | 3.612 | +12.9% | 672 / 17 | 800 / 19 |

Scope filtering measures the intended exclusion of sixteen callbacks, rather
than faster execution of the same callback work.

### Do extra array allocations translate to slower conversion?

The final end-to-end measurements show modest regressions, with important
variation between rounds/statistics. The following ranges are the lowest/highest
trial minima, while the last columns are medians of the three trial medians:

| Case | Before minimum range us | After minimum range us | Before p50 us | After p50 us |
| --- | ---: | ---: | ---: | ---: |
| Buffer-only native | 6.925–7.225 | 7.400–7.675 | 7.625 | 8.300 |
| Buffer-only wrapper | 5.133–5.483 | 5.550–5.633 | 6.500 | 6.433 |
| Interface-only native | 20.800–21.500 | 21.800–22.999 | 22.700 | 24.100 |
| Array-method-only native | 7.266–7.900 | 7.933–8.233 | 8.400 | 9.100 |
| ndarray native | 4.257–4.800 | 4.750–4.950 | 5.157 | 5.167 |
| ndarray wrapper | 3.125–3.312 | 3.288–3.725 | 3.538 | 3.925 |

ndarray wrapping's median latency also rises about 11%; native ndarray median
latency is essentially unchanged despite the worse minimum. Buffer wrapping's
median also does not regress. Therefore the extra allocations do not imply a
fixed slowdown across every workload or statistic. These experiments compare
the whole old/new conversion paths; they cannot assign all timing changes to
allocation alone.

### Allocation attribution

Separate warmed Profile.Allocs runs on both revisions identify two extra objects
for wrappers: Some{PyArray{...}} created in pyarray_make/pyconvert_return, and a
boxed PyArray when the result is unwrapped in pyconvert_raw. Native conversion
also creates Some{Vector{...}} in pyconvert_tryconvert, and unwraps the temporary
PyArray through pyconvert_result. These account for the additional object counts.

The profiles report requested object sizes; BenchmarkTools reports allocator
byte totals. Use the stable measured differences of 128 bytes / two allocations
for wrappers and 144 bytes / three allocations for native arrays. Raw stacks
are in convert-array-allocations-before.txt and -after.txt. No array runtime
optimization was made in this refresh.

## Selection misses and invalidation

Cache clearing is BenchmarkTools setup, outside timing, with evals=1.
A selection miss clears only target/context selections and preserves the global
order. Full invalidation also dirties the global order. Old v1 has no equivalent
global DAG rebuild. Timings include converting the fixture.

Selection-miss trials use up to 200 samples / 0.4 seconds; invalidation trials
use up to 40. Aggregate values again use the median of three trial minima.

| Case | Selection before us | Selection after us | Invalidated before us | Invalidated after us |
| --- | ---: | ---: | ---: | ---: |
| Single rule | 26.899 | 71.199 | 27.599 | 2588.880 |
| Decline then success | 28.100 | 67.899 | 29.600 | 2602.380 |
| Three declines then success | 32.599 | 62.499 | 34.599 | 2588.081 |
| Sixteen scoped decliners | 65.500 | 70.199 | 75.699 | 2587.680 |
| Virtual Mapping ABC | 57.399 | 71.499 | 65.300 | 2600.080 |
| Buffer-only native | 84.900 | 43.600 | 104.499 | 2608.181 |
| Buffer-only wrapper | 75.499 | 35.000 | 96.500 | 2528.881 |
| Interface-only native | 66.700 | 58.100 | 246.598 | 2890.477 |
| Array-method-only native | 45.099 | 38.999 | 51.099 | 2648.979 |
| ndarray native | 71.399 | 39.000 | 85.699 | 2582.780 |
| ndarray wrapper | 70.500 | 34.300 | 80.999 | 2870.077 |

Full rebuilding is a real rare-path cost: approximately 1.47 MB / 21,262
allocations for the single-rule fixture, compared with 4,976 B / 149 before.
Ordinary protocol selection misses generally improve, while simple synthetic
classes regress. No registry-size scaling study was performed.

## Multiple matching preferences

Both supplied source names match one controlled class. Its String rule declines
and its Int rule succeeds with 42. The two preference attempts always have
nonempty intersections with Any. Context construction is included in timing.

| Ordered attempts | After us | B / allocs |
| --- | ---: | ---: |
| Int succeeds before String | 3.375 | 560 / 18 |
| String declines, then Int succeeds | 4.128 | 720 / 24 |
| String/Float64 decline, then ordinary conversion succeeds | 5.143 | 752 / 25 |

Old v1 is explicitly recorded as unsupported. These scalar fixtures make
fallthrough controllable; they do not recommend narrow scalar preferences.
Additional unmatched/wrapper/nested cases are in convert-dispatch-final.tsv.
Those include deliberate representation/work differences: matching wrapper
preferences avoid copying, whereas an unmatched preference is checked again
by context-propagating child conversions.

## Reproduction

Run benchmark/run-convert-final.sh as documented in the main report, then
python3 benchmark/summarize-convert-final.py. For allocation profiles, run
convert-array-allocations.jl with labels before/after in their respective
Julia environments, using the same installed Python. Profiling runs occur
after timing and do not overlap it.
