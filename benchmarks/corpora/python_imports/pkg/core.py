import functools


def helper():
    return 1


def traced(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    return wrapper


@traced
def decorated():
    return helper()


class Engine:
    def run(self):
        return decorated()
