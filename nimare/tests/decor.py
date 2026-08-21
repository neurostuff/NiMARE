from time import time
from time import sleep


def timeit(func):
    """Decorator to measure the execution time of a function."""

    def wrapper(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()
        print(f"Function '{func.__name__}' took {end_time - start_time:.2f} seconds to execute.")
        return result

    return wrapper


@timeit
def big_computation():
    """Simulate a big computation by sleeping for 1 second."""
    sleep(1)


timeit_big_computation = timeit(big_computation)


def big_computation_2():
    """Simulate a big computation by sleeping for 2 seconds."""
    sleep(2)


print(big_computation())
