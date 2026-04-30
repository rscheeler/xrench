import io

import pytest

from xrench import logcontrol


def test_init_adds_filtered_handler_and_defaults_to_muted(monkeypatch) -> None:
    add_calls = []
    remove_calls = []
    disable_calls = []
    fake_stderr = io.StringIO()

    def fake_remove(handler_id=0) -> None:
        remove_calls.append(handler_id)

    def fake_add(sink, *args, **kwargs) -> int:
        add_calls.append((sink, args, kwargs))
        return 1

    def fake_disable(module_name) -> None:
        disable_calls.append(module_name)

    monkeypatch.setattr(logcontrol.logger, "remove", fake_remove)
    monkeypatch.setattr(logcontrol.logger, "add", fake_add)
    monkeypatch.setattr(logcontrol.logger, "disable", fake_disable)
    monkeypatch.setattr(logcontrol.sys, "stderr", fake_stderr)

    controller = logcontrol.LOGCONTROLLER(module_name="xrench", level="warning")

    assert controller.level == "WARNING"
    assert remove_calls == [0]
    assert len(add_calls) == 1
    assert add_calls[0][0] is fake_stderr
    assert add_calls[0][2]["level"] == "WARNING"
    assert callable(add_calls[0][2]["filter"])
    assert add_calls[0][2]["filter"]({"name": "xrench.foo"}) is True
    assert add_calls[0][2]["filter"]({"name": "other"}) is False
    assert disable_calls == ["xrench"]


def test_level_setter_replaces_existing_handler(monkeypatch):
    add_calls = []
    remove_calls = []
    fake_stderr = io.StringIO()
    ids = [1, 2]

    def fake_remove(handler_id=0):
        remove_calls.append(handler_id)

    def fake_add(sink, *args, **kwargs):
        add_calls.append((sink, args, kwargs))
        return ids.pop(0)

    monkeypatch.setattr(logcontrol.logger, "remove", fake_remove)
    monkeypatch.setattr(logcontrol.logger, "add", fake_add)
    monkeypatch.setattr(logcontrol.logger, "disable", lambda name: None)
    monkeypatch.setattr(logcontrol.sys, "stderr", fake_stderr)

    controller = logcontrol.LOGCONTROLLER(module_name="xrench")
    assert controller._handler_id == 1

    controller.level = "debug"

    assert controller.level == "DEBUG"
    assert controller._handler_id == 2
    assert remove_calls[-1] == 1
    assert add_calls[-1][2]["level"] == "DEBUG"


def test_mute_and_unmute_call_logger_enable_disable(monkeypatch):
    enable_calls = []
    disable_calls = []

    monkeypatch.setattr(logcontrol.logger, "remove", lambda handler_id=0: None)
    monkeypatch.setattr(logcontrol.logger, "add", lambda sink, *args, **kwargs: 1)
    monkeypatch.setattr(
        logcontrol.logger,
        "enable",
        lambda module_name: enable_calls.append(module_name),
    )
    monkeypatch.setattr(
        logcontrol.logger,
        "disable",
        lambda module_name: disable_calls.append(module_name),
    )
    monkeypatch.setattr(logcontrol.sys, "stderr", io.StringIO())

    controller = logcontrol.LOGCONTROLLER(module_name="xrench")
    controller.unmute()
    controller.mute()

    assert enable_calls == ["xrench"]
    assert disable_calls == ["xrench", "xrench"]
