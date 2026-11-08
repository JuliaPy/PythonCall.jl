# v1 pyconvert final implementation report

The redesign is implemented on v1. The latest minimal-wrapping optimization and
current performance totals are in [convert-unwrapped-report.md](convert-unwrapped-report.md).
Latest local checks: **1,765 conversion/wrapper/Julia-wrapper checks passed**, with
one existing optional-dependency skip. The most recent 15-test ordinary Python
run predates that optimization. The
maintainer assigned final full-suite/notebook/docs and version-matrix validation
to CI; those checks were not rerun for this benchmark refresh. Work is uncommitted.

## Design implemented

- Rules store source, advertised output, scope, function and increasing ID. Adds
  always create entries; replacement preserves identity/order. Mutations invalidate
  all selections. Pending sources resolve without importing optional modules.
- A global strict-subclass DAG includes ABC relationships and independent array/
  buffer matchers. Inherited priorities are monotonic; stable topological sorting
  uses source registration order, then rule creation order. Aliases share ordering
  and priority. Per-call selection filters without reordering.
- Output intersection and destination scope determine eligibility, including
  normalized union arms. Each rule receives non-union intersections, rule-first;
  union-arm order is unspecified. The final Py fallback is outside the registry.
- Source names resolve loaded-module attributes, including public re-exports and
  dotted paths. Bare names abbreviate builtins. Legacy exec-defined ancestors
  can also resolve from an encountered class hierarchy. Cached Python type
  references prevent pointer reuse.
- Cached invocations select context-aware/legacy signatures once per target and
  context type. Scalar shortcuts require a matching cached first rule/target.
  Explicit contexts carry ordered source-specific preferences and propagate
  recursively when requested, with no task-local state.
- Canonical materialized containers become native Julia containers. Vector/dict
  types widen with Base.promote_typejoin within requested bounds; unconstrained
  empties use Any. Generic iterables remain lazy PyIterable values.
- Rules accept values, nothing for decline, and Some for explicit success.
  Public pytryconvert always returns Some/nothing. Internal raw values, a private
  failure sentinel and an escape avoid per-element Some allocations. Exceptions
  propagate; preferred-type introspection returns a candidate, not a guarantee.
  Built-in success helpers wrap only actual nothing/Some values; array and
  native-container results cross dispatch boundaries directly.

Public additions: PyConvertContext, pytryconvert, pyconvert_remove_rule,
pyconvert_replace_rule, pyconvert_rules, scope and prefer. Public rule priorities
and result helpers were removed. Main changes are in src/Convert/pyconvert.jl,
rules.jl, NumPy/pandas registrations, Wrap/PyArray, JlWrap, API declarations,
conversion documentation, migration guide and tests.

## Agreed decisions and limitations

The maintainer accepted pyconvert(Py, x::Py) as an identity shortcut. Generic
numbers.Real/Complex coercions were replaced by float/complex source rules:
integral/rational sources retain exact intermediates, supported NumPy scalars
retain dedicated rules, and other numeric types require explicit rules. Python
float-to-Int remains available when exact and in range. This fixes integer
overflow being hidden by an intermediate rounded Float64.

The optimized internal result representation is explicitly allowed by the
specification; the public contract is unchanged. v1 calls its iterator wrapper
PyIterable rather than PyIterator. Ownership remains an API contract.
Cycles/identity preservation, dynamic ABC mutations, arbitrary module/type
replacement and method changes without invalidation remain unsupported.
Copy-policy frameworks and public priority controls were deliberately omitted.

## Final benchmark method

The full original-v1 comparison below measures the implementation immediately
before minimal result wrapping. The linked follow-up supersedes affected
allocation/timing totals and compares directly against that pre-change snapshot.

Baseline: original v1 commit 25f82cc8effbd6c487ddceaffc34aa7cbe86505b, extracted
into an isolated checkout/environment. After: the final working tree, including
the alias/shorthand and numeric-source changes. Environment records include
source paths and SHA256 fingerprints.

Linux x86_64, Intel Core i7-8750H, Julia 1.13.2, Python 3.15.0, NumPy 2.5.4,
BenchmarkTools 1.8.0. Both revisions use the same Python executable and Null
CondaPkg backend. All processes ran sequentially without concurrent tests.
Round order: before/after, after/before, before/after.

Tables use microseconds and the **median of three trial minima**. Warm trials
use up to 5,000 samples / one second. Julia bytes/allocation counts exclude
Python allocations. Trial medians and round ranges remain in
convert-final-summary.tsv. These are short local trials, not confidence intervals.

## Cached scalars

The requested type is visible to inference through Val. Runtime-target variants
are separately recorded; those can box values in the benchmark caller.

| Target/source | Before us | After us | Change | Before B / allocs | After B / allocs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Int / int | 0.223 | 0.240 | +7.9% | 0 / 0 | 0 / 0 |
| Integer / int | 0.590 | 0.242 | -59.1% | 0 / 0 | 0 / 0 |
| BigInt / large int | 0.977 | 0.970 | -0.7% | 216 / 7 | 216 / 7 |
| Float64 / float | 0.240 | 0.240 | -0.1% | 0 / 0 | 0 / 0 |
| Bool / bool | 0.232 | 0.254 | +9.3% | 0 / 0 | 0 / 0 |
| String / str | 0.297 | 0.311 | +5.0% | 48 / 2 | 48 / 2 |
| Nothing / None | 0.211 | 0.230 | +9.3% | 0 / 0 | 0 / 0 |
| Int / incompatible str | 0.239 | 0.232 | -2.8% | 0 / 0 | 0 / 0 |

Positive scalar differences are tens of nanoseconds and remain visible in the
report. None is not slower than Int in the final implementation.
Float64/BigInt no longer retain the initial redesign's extra allocation.

## Containers

| Case | Before us | After us | Change | Before B / allocs | After B / allocs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Vector, 10 integers | 9.899 | 6.760 | -31.7% | 704 / 17 | 720 / 18 |
| Vector, 10,000 integers | 7243.345 | 4821.463 | -33.4% | 513864 / 19507 | 513880 / 19508 |
| Vector{Int}, 10,000 integers | 3918.970 | 4122.068 | +5.2% | 362008 / 10017 | 362024 / 10018 |
| Vector, mixed numeric | 4.471 | 3.800 | -15.0% | 320 / 11 | 352 / 13 |
| Vector, heterogeneous | 5.466 | 4.786 | -12.5% | 368 / 13 | 384 / 14 |
| Vector{Vector{Int}}, nested | 7.975 | 7.060 | -11.5% | 720 / 26 | 768 / 29 |
| Dict, heterogeneous values | 7.425 | 6.700 | -9.8% | 1248 / 22 | 1280 / 24 |
| Dict{String,Float64} | 3.237 | 3.750 | +15.8% | 656 / 15 | 672 / 16 |
| Empty Vector | 1.290 | 1.140 | -11.6% | 64 / 3 | 112 / 5 |
| Empty Dict | 1.054 | 1.100 | +4.4% | 112 / 3 | 208 / 5 |

Mixed numeric outputs intentionally change from promoted Float64 to Real while
preserving individual values; dictionary value types change similarly. Empty
types change from bottom to Any. Nested Any changes from wrapping (0.390 us,
16 B / 1 allocation) to recursive copying (9.500 us, 864 B / 32 allocations);
those perform different work. Explicit PyList and iterator wrapping remain
about 0.39 us / 16 B / 1 allocation.

## Arrays, dispatch and preferences

In this pre-optimization snapshot, array wrappers retain 128 extra bytes / two allocations, native arrays 144 bytes /
three allocations. Median trial minima rise 3.4% for buffer wrapping, 12.9% for
ndarray wrapping, and 4.8–7.7% for native arrays. Native ndarray trial medians
are essentially unchanged, while ndarray wrapper trial medians rise about 11%.
These end-to-end timings do not isolate allocation cost from other dispatch
changes. See [convert-expanded-report.md](convert-expanded-report.md) for ranges,
complete protocol/cold-path tables and allocation attribution.

Cached single-rule/decline-chain dispatch stays allocation-free at roughly
0.38/0.54/0.94 us. Scope filtering excludes sixteen callbacks (4.07 to 0.38 us),
an intentional work difference. Selection misses are mixed: the simple fixture
rises from 26.9 to 71.2 us, while most protocol cases improve.

Full invalidation/global rebuilding costs about 2.5–2.9 ms and 1.47–1.51 MB
with builtins plus 27 synthetic registrations, versus tens/hundreds of
microseconds in the old per-type scheme. Warm calls never rebuild the DAG.
This remains a measured cold-path regression, not a registry scaling study.

New-API matching-preference cases cost 3.38/4.13/5.14 us for first success,
first decline then success, and all preferences declining before ordinary
conversion. Context-construction/source checks are included. Old v1 is marked
unsupported. Broader nested preferences are in convert-dispatch-final.tsv.

## Result representation and inference

Final raw-Unconverted versus Some/nothing models cover successful/failed Int64,
Float64, String, Vector{Int}, Integer, AbstractArray, Union{Int64,Nothing}, Any,
escaping results and typed 100-element loops. Both loops allocate zero;
escaping Some values can allocate, including an extra 16 bytes in the Int case.
This supports optimized internal dispatch behind the ergonomic public API.

Inference preserves Some{Int} for Int, existential Some subtypes for Integer,
separate Some alternatives for Union{Int,Nothing}, and Some for Any. Scalar
profiles confirm no internal Float64 box or Some{BigInt} allocation. See
convert-results-final.tsv, convert-inference-final.txt, convert-scalars-final-*
and [convert-scalars-report.md](convert-scalars-report.md).

## Reproduction and artifacts

Use the same Julia executable and installed Python/backend in all runs. Prepare
an isolated baseline environment without changing the working tree:

~~~sh
convert_baseline_src=$(mktemp -d)
convert_baseline_env=$(mktemp -d)
git archive 25f82cc8effbd6c487ddceaffc34aa7cbe86505b | tar -x -C "$convert_baseline_src"
/absolute/path/to/julia --project="$convert_baseline_env" -e \
  'using Pkg; Pkg.develop(path=ARGS[1]); Pkg.add(PackageSpec(name="BenchmarkTools",version="1.8.0"))' \
  "$convert_baseline_src"
~~~

With the checkout's benchmark dependencies installed, run:

~~~sh
JULIA_CONDAPKG_BACKEND=Null JULIA_PYTHONCALL_EXE=/absolute/path/to/python \
  bash benchmark/run-convert-final.sh /absolute/path/to/julia "$convert_baseline_env"
python3 benchmark/summarize-convert-final.py
~~~

The runner serializes three paired common/expanded rounds, then final scalar,
result/inference and preference suites. Fixture assertions passed on both
revisions. Allocation attribution can be reproduced with
convert-array-allocations.jl before/after in the respective environments.

Authoritative paired data use convert-final-{before,after}-r{1,2,3} and
convert-expanded-final-{before,after}-r{1,2,3}; each includes an environment record.
The summarizer validates constant source snapshots and allocation counts across
rounds. Older TSV/profile files are historical snapshots, superseded by these
final measurements. No runtime policy or conversion optimization changed during
this refresh.
