import pytest

from xrench import utils


@pytest.fixture(autouse=True)
def clear_singleton_instances() -> None:
    utils.Singleton._instances.clear()
    yield
    utils.Singleton._instances.clear()


class DummySingleton(metaclass=utils.Singleton):
    def __init__(self, value: int) -> None:
        self.value = value
        self.init_count = getattr(self, "init_count", 0) + 1


class AnotherDummySingleton(metaclass=utils.Singleton):
    def __init__(self, value: str) -> None:
        self.value = value


def test_singleton_returns_same_instance_on_multiple_calls() -> None:
    first = DummySingleton(1)
    second = DummySingleton(2)

    assert first is second
    assert first.value == 2
    assert second.value == 2
    assert first.init_count == 2


def test_singleton_maintains_separate_instances_per_class() -> None:
    first = DummySingleton(1)
    second = AnotherDummySingleton("hello")

    assert first is not second
    assert first.value == 1
    assert second.value == "hello"


def test_singleton_reinitializes_existing_instance_on_subsequent_calls() -> None:
    instance = DummySingleton(5)
    assert instance.init_count == 1

    second_call = DummySingleton(10)
    assert second_call is instance
    assert instance.value == 10
    assert instance.init_count == 2
