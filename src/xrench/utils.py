from __future__ import annotations


class Singleton:
    """
    Singleton class.
    See: https://stackoverflow.com/questions/6760685/creating-a-singleton-in-python.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):  # noqa: D102
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        else:
            cls._instances[cls].__init__(*args, **kwargs)
        return cls._instances[cls]
