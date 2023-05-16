#!/usr/bin/env python3
import pyperf

from suites import SUITE

def main():
    runner = pyperf.Runner()

    for (name, func, args) in SUITE:
        runner.bench_func(name, func, *args)

    return None

if __name__ == '__main__':
    main()
