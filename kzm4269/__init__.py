def _main():
    from contextlib import contextmanager

    @contextmanager
    def f():
        print('enter')
        yield
        print('exit')

    @f()
    def g():
        print('hello')

    g()


if __name__ == '__main__':
    _main()
