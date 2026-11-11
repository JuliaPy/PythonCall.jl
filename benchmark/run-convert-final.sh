#!/usr/bin/env bash
# Sequential runs: no overlapping benchmark/test processes launched by this script.
# Supply the absolute Julia executable and an environment developed against the
# original v1 revision, with BenchmarkTools installed. The installed Python
# executable/backend are inherited from the environment and shared by all runs.
set -euo pipefail
if [[ $# != 2 ]]; then
    printf 'Usage: bash benchmark/run-convert-final.sh JULIA_EXECUTABLE BASELINE_ENV\n' >&2
    exit 2
fi
julia_exe=$1
baseline_env=$2
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
for round in 1 2 3; do
    if [[ $round == 2 ]]; then
        revisions=(after before)
    else
        revisions=(before after)
    fi
    for revision in "${revisions[@]}"; do
        project=$baseline_env
        if [[ $revision == after ]]; then
            project=$repo_dir/benchmark
        fi
        printf 'Measuring %s round %s\n' "$revision" "$round"
        "$julia_exe" --project="$project" "$repo_dir/benchmark/convert-final.jl" "$revision-r$round"
        "$julia_exe" --project="$project" "$repo_dir/benchmark/convert-expanded.jl" "final-$revision-r$round"
    done
done
"$julia_exe" --project="$repo_dir/benchmark" "$repo_dir/benchmark/convert-scalars.jl" final
"$julia_exe" --project="$repo_dir/benchmark" "$repo_dir/benchmark/convert-results.jl" final
"$julia_exe" --project="$repo_dir/benchmark" "$repo_dir/benchmark/convert-dispatch.jl" final
printf 'Final conversion benchmarks complete\n'
