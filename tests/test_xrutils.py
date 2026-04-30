import numpy as np
import pytest
import xarray as xr
from scipy.spatial.transform import Rotation

from xrench import xrutils
from xrench.units import ureg


class TestKw2da:
    def test_converts_scalar_to_dataarray(self) -> None:
        result = xrutils.kw2da(x=5)
        assert isinstance(result["x"], xr.DataArray)
        assert result["x"].dims == ("x",)
        assert np.array_equal(result["x"].values, [5])

    def test_converts_array_to_dataarray(self) -> None:
        arr = np.array([1, 2, 3])
        result = xrutils.kw2da(x=arr)
        assert isinstance(result["x"], xr.DataArray)
        assert result["x"].dims == ("x",)
        assert np.array_equal(result["x"].values, arr)

    def test_converts_dataarray_with_quantity_to_base_units(self) -> None:
        da = xr.DataArray(np.array([5.0, 10.0]) * ureg("km"), dims=("distance",))
        result = xrutils.kw2da(distance=da)
        assert isinstance(result["distance"], xr.DataArray)
        assert result["distance"].dims == ("distance",)
        assert np.allclose(result["distance"].values, [5000, 10000])

    def test_preserves_existing_dataarray(self) -> None:
        da = xr.DataArray([1, 2, 3], dims=("x",), coords={"x": [1, 2, 3]})
        result = xrutils.kw2da(data=da)
        assert result["data"] is da

    def test_handles_multiple_kwargs(self) -> None:
        result = xrutils.kw2da(x=1, y=2, z=3)
        assert len(result) == 3
        assert all(isinstance(v, xr.DataArray) for v in result.values())


class TestVectorNorm:
    def test_computes_norm_on_single_dimension(self) -> None:
        da = xr.DataArray([[3, 4]], dims=("batch", "component"))
        result = xrutils.vector_norm(da, dim="component")
        assert np.isclose(result.values[0], 5.0)

    def test_default_norm_order(self) -> None:
        da = xr.DataArray([[1, 2, 2]], dims=("batch", "component"))
        result = xrutils.vector_norm(da, dim="component")
        assert np.isclose(result.values[0], 3.0)

    def test_custom_norm_order(self) -> None:
        da = xr.DataArray([[1, 2, 3]], dims=("batch", "component"))
        result = xrutils.vector_norm(da, dim="component", ord=1)
        assert np.isclose(result.values[0], 6.0)


class TestComputeIfDask:
    def test_returns_numpy_array_unchanged(self) -> None:
        arr = np.array([1, 2, 3])
        result = xrutils.compute_if_dask(arr)
        assert result is arr

    def test_returns_scalar_unchanged(self) -> None:
        result = xrutils.compute_if_dask(42)
        assert result == 42


class TestWrapsXr:
    def test_strips_units_on_entry_and_reapplies_on_exit(self) -> None:
        @xrutils.wraps_xr(ret_units="meter")
        def process(x: xr.DataArray) -> xr.DataArray:
            return x * 2

        da_with_units = xr.DataArray([5.0], dims=("x",))
        result = process(da_with_units * ureg("meter"))
        assert isinstance(result.data, type(ureg("1 meter")))
        assert np.isclose(result.data.magnitude[0], 10.0)

    def test_with_multiple_arg_units(self) -> None:
        @xrutils.wraps_xr(ret_units="meter", arg_units=["meter", None])
        def add_arrays(x: xr.DataArray, y: xr.DataArray) -> xr.DataArray:
            return x + y

        da1 = xr.DataArray([5.0], dims=("x",)) * ureg("meter")
        da2 = xr.DataArray([3.0], dims=("x",))
        result = add_arrays(da1, da2)
        assert isinstance(result.data, type(ureg("1 meter")))
        assert np.isclose(result.data.magnitude[0], 8.0)

    def test_with_none_units(self) -> None:
        @xrutils.wraps_xr(ret_units=None)
        def get_magnitude(x: xr.DataArray) -> xr.DataArray:
            return x

        da = xr.DataArray([5.0], dims=("x",))
        result = get_magnitude(da)
        assert result is da


class TestApplyRotation:
    def test_applies_identity_rotation(self) -> None:
        da = xr.DataArray(
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            dims=("batch", "position"),
            coords={"position": ["x", "y", "z"]},
        )
        rotation = Rotation.identity()
        result = xrutils.apply_rotation(rotation, da, rotation_dim="position")
        assert np.allclose(result.values, da.values)

    def test_applies_90_degree_rotation(self) -> None:
        da = xr.DataArray(
            np.array([[1.0, 0.0, 0.0]]),
            dims=("batch", "position"),
            coords={"position": ["x", "y", "z"]},
        )
        rotation = Rotation.from_rotvec([0, 0, np.pi / 2])
        result = xrutils.apply_rotation(rotation, da, rotation_dim="position")
        assert np.allclose(result.values[0, 0], 0.0, atol=1e-10)
        assert np.allclose(result.values[0, 1], 1.0, atol=1e-10)

    def test_applies_inverse_rotation(self) -> None:
        da = xr.DataArray(
            np.array([[1.0, 2.0, 3.0]]),
            dims=("batch", "position"),
            coords={"position": ["x", "y", "z"]},
        )
        rotation = Rotation.from_rotvec([0.1, 0.2, 0.3])
        result_forward = xrutils.apply_rotation(rotation, da.copy(), rotation_dim="position")
        result_inverse = xrutils.apply_rotation(
            rotation,
            result_forward.copy(),
            rotation_dim="position",
            inverse=True,
        )
        assert np.allclose(result_inverse.values, da.values)

    def test_preserves_units_after_rotation(self) -> None:
        da = xr.DataArray(
            np.array([[1.0, 0.0, 0.0]]) * ureg("meter"),
            dims=("batch", "position"),
            coords={"position": ["x", "y", "z"]},
        )
        rotation = Rotation.identity()
        result = xrutils.apply_rotation(rotation, da, rotation_dim="position")
        assert isinstance(result.data, type(ureg("1 meter")))
