import importlib

import pint

from xrench import units


def test_get_ureg_uses_existing_application_registry(monkeypatch) -> None:
    original_registry_class = pint.UnitRegistry
    existing_registry = original_registry_class(cache_folder=":auto:")

    monkeypatch.setattr(units.pint, "get_application_registry", lambda: existing_registry)
    monkeypatch.setattr(units.pint, "UnitRegistry", original_registry_class)
    monkeypatch.setattr(units.pint, "set_application_registry", lambda registry: None)

    reloaded = importlib.reload(units)

    assert reloaded.get_ureg() is existing_registry
    assert reloaded.ureg is existing_registry
    assert existing_registry.autoconvert_offset_to_baseunit is True
    assert existing_registry.force_ndarray_like is True
    assert str(reloaded.ureg("1 dBK").units) == "decibelkelvin"


def test_get_ureg_creates_and_sets_registry_when_internal(monkeypatch) -> None:
    original_registry_class = pint.UnitRegistry
    internal_registry = original_registry_class(cache_folder=None)

    created_registry: pint.UnitRegistry | None = None
    set_calls: list[pint.UnitRegistry] = []

    def fake_unit_registry(*args, **kwargs):
        nonlocal created_registry
        created_registry = original_registry_class(*args, **kwargs)
        return created_registry

    monkeypatch.setattr(units.pint, "get_application_registry", lambda: internal_registry)
    monkeypatch.setattr(units.pint, "UnitRegistry", fake_unit_registry)
    monkeypatch.setattr(
        units.pint,
        "set_application_registry",
        lambda registry: set_calls.append(registry),
    )

    reloaded = importlib.reload(units)

    assert created_registry is not None
    assert set_calls == [created_registry]
    assert reloaded.ureg is created_registry
    assert created_registry.cache_folder is not None
    assert created_registry.autoconvert_offset_to_baseunit is True
    assert created_registry.force_ndarray_like is True
    assert str(reloaded.ureg("1 dBHz").units) == "decibelhertz"
