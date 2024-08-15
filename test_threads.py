from juliacall import Main as jl
from concurrent.futures import ThreadPoolExecutor
from time import sleep

pool = ThreadPoolExecutor(4)

jl_yield = jl.seval("yield")
jl_sleep = jl.seval("sleep")

ids = []


def target():
    print("starting")
    jl.seval("PythonCall.GIL.@unlock sleep(3)")
    print("stopping")
    ids.append(jl.Threads.threadid())


print("submitting...")
tasks = [pool.submit(target) for _ in range(8)]

while True:
    done = [task.done() for task in tasks]
    print("done:", done)
    if all(done):
        break
    print("sleeping...")
    jl_sleep(1)

print("threadids:", ids)
