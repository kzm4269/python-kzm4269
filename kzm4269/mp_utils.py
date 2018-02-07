"""Utilities for multiprocessing"""
import multiprocessing as mp
from collections import namedtuple
from contextlib import contextmanager

PmapResult = namedtuple('PmapResult', 'value, exc')


@contextmanager
def pmap(func, sequence, processes=None):
    """Parallel map() based on multiprocessing."""
    if processes is None:
        processes = mp.cpu_count()

    publish_queue = mp.Queue(1)
    subscribe_queue = mp.Queue()

    def publish():
        """Put values in sequence into the publish queue."""
        try:
            for data in sequence:
                publish_queue.put((data,))
        finally:
            for _ in range(processes):
                publish_queue.put(None)

    def convert():
        """Apply func to values in the publish queue and put these into the
        subscribe queue.
        """
        try:
            for data, in iter(publish_queue.get, None):
                try:
                    result = PmapResult(value=func(data), exc=None)
                except BaseException as exc:
                    result = PmapResult(value=None, exc=exc)
                subscribe_queue.put(result)
        finally:
            subscribe_queue.put(None)

    def subscribe():
        """Get values in the subscribe queue."""
        n = 0
        while n < processes:
            result = subscribe_queue.get()
            if result is None:
                n += 1
            else:
                yield result

    procs = [mp.Process(target=publish)]
    procs += [mp.Process(target=convert) for _ in range(processes)]
    for p in procs:
        p.start()
    try:
        yield subscribe()
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()


def _main():
    from time import sleep
    from random import random

    def func(x):
        sleep((1 + random()) / 2)
        if random() < 0.2:
            raise Exception()
        return str(x)

    with pmap(func, range(10)) as results:
        for result in results:
            print(result)


if __name__ == '__main__':
    _main()
