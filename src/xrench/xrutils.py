from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from copy import deepcopy
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
import xarray as xr
from pint import Quantity
from scipy.spatial.transform import Rotation

from .units import ureg

if TYPE_CHECKING:
    from typing import ParamSpec

    P = ParamSpec("P")
    R = TypeVar("R")
    # Define types for the units input (can be string, unit object, or sequence)
    UnitLike = Any  # pint.Unit, str, or None
    UnitsIn = Sequence[UnitLike]
    UnitsOut = UnitLike | Sequence[UnitLike]


def kw2da(**kwargs: Any) -> dict[str, xr.DataArray]:
    """
    Convert kwargs of one-dimensional data into xarray DataArray objects.
    Handles floats, lists, pint Quantities, numpy arrays, and existing DataArrays.
    """
    out = {}

    for k, v in kwargs.items():
        # 1. Handle existing DataArrays
        if isinstance(v, xr.DataArray):
            # Check if underlying data is a pint Quantity
            if hasattr(v.data, "to_base_units"):
                v = v.copy(deep=True)  # Avoid mutating original
                v.data = v.data.to_base_units()
            out[k] = v
            continue

        # 2. Handle Pint Quantities (convert to base units and strip magnitude for coords)
        if hasattr(v, "to_base_units"):
            v = v.to_base_units().magnitude

        # 3. Normalize to a 1D numpy array
        # np.atleast_1d handles scalars (0D), lists, and arrays consistently
        v_arr = np.atleast_1d(v)

        # 4. Create the DataArray
        # Using k for both the dimension name and the coordinate values
        out[k] = xr.DataArray(
            v_arr,
            dims=(k,),
            coords={k: v_arr},
            name=k,
        )

    return out


def vector_norm(x: xr.DataArray, dim: str, ord: Any | None = None) -> xr.DataArray:
    """
    Wrapper to perform np.linalg.norm on a xr.DataArray.

    Parameters
    ----------
    x : xr.DataArray
        Array to dermine norm on
    dim : str
        Dimension(s) to calculate norm
    ord : {non-zero int, inf, -inf, 'fro', 'nuc'}, optional
        Order of the norm. Default is None.

    References:
    ----------
    https://www.programcreek.com/python/example/123575/xarray.apply_ufunc
    """
    norm = xr.apply_ufunc(
        np.linalg.norm,
        x,
        input_core_dims=[[dim]],
        kwargs=dict(ord=ord, axis=-1),
    )

    return norm


def compute_if_dask(x: Any) -> Any:
    """Computes a dask array/quantity, otherwise returns the input."""
    # Check if the object itself is a pint Quantity with dask data
    if hasattr(x, "compute"):
        return x.compute()
    # It's a numpy array, a pint Quantity with numpy data, etc.
    return x


def wraps_xr(
    ret_units: Any,
    arg_units: Sequence[Any] | None = None,
    kwarg_units: dict[str, Any] | None = None,
) -> Callable:
    """
    Decorator that strips units on entry and adds them back on exit.
    Preserves xarray containers to maintain coordinate/dimension integrity.

    Parameters
    ----------
    ret_units : Any
        Units to apply to the returned result.
    arg_units : Sequence[Any] | None, optional
        Units for positional arguments.
    kwarg_units : dict[str, Any] | None, optional
        Units for keyword arguments.
    """
    actual_arg_units = arg_units if arg_units is not None else ()
    actual_kwarg_units = kwarg_units if kwarg_units is not None else {}

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # 1. STRIP: Convert Quantities to Magnitudes inside the containers
            new_args = list(args)
            for i, unit in enumerate(actual_arg_units):
                if i < len(new_args):
                    new_args[i] = _to_mag(new_args[i], unit)

            for key, unit in actual_kwarg_units.items():
                if key in kwargs:
                    kwargs[key] = _to_mag(kwargs[key], unit)

            # 2. EXECUTE: Function works with unit-less DataArrays
            result = func(*new_args, **kwargs)

            # 3. ADD: Re-apply units to the output
            return _to_unit(result, ret_units)

        return wrapper

    return decorator


def _to_mag(obj: Any, unit: Any) -> Any:
    """Strip units while preserving the xarray shell."""
    if unit is None:
        return obj

    # Handle list of items
    if isinstance(obj, (list, tuple)) and not hasattr(obj, "to"):
        return type(obj)(_to_mag(item, unit) for item in obj)

    target_unit = ureg(unit) if isinstance(unit, str) else unit

    if isinstance(obj, xr.DataArray):
        if hasattr(obj.data, "to"):
            # Return a copy of the DataArray but with raw magnitude data
            return obj.copy(data=obj.data.to(target_unit).magnitude)
        return obj

    if hasattr(obj, "to"):
        return obj.to(target_unit).magnitude

    return obj


def _to_unit(obj: Any, unit: Any) -> Any:
    """Apply units back to the data."""
    if isinstance(unit, (tuple, list)):
        if not isinstance(obj, (tuple, list)) and len(unit) == 1:
            return _to_unit(obj, unit[0])
        return type(obj)(_to_unit(o, u) for o, u in zip(obj, unit))

    if unit is None:
        return obj

    q = ureg(unit) if isinstance(unit, str) else unit

    if isinstance(obj, xr.DataArray):
        # Prevent double-wrapping if it already has units
        if hasattr(obj.data, "to"):
            return obj.copy(data=obj.data.to(q))

        # Add units to the raw data (numpy/dask)
        new_obj = obj.copy(deep=False)
        new_obj.data = obj.data * q
        return new_obj

    return obj * q


def apply_rotation(
    rotation: Rotation | xr.DataArray,
    da: xr.DataArray,
    rotation_dim: str = "position",
    inverse: bool | None = False,
) -> xr.DataArray:
    """
    Applies the rotation from the Rotation object to a DataArray of coordinates. Function can be used for
    rotation of polarization.

    Parameters
    ----------
    rotation : Rotation, xr.DataArray
        Rotation object to be applied
    da : xr.DataArray
        DataArray to apply the rotation to. Shall have dimension "position" with coordinate "x", "y", "z".
    rotation_dim: str="position"
        Rotation dim to apply to.
    inverse : bool
        Whether to apply the inverse of the rotation
    """
    # Get original dimensions
    olddims = deepcopy(list(da.dims))

    # Transpose so rotation dimension is last
    newdims = deepcopy(olddims)
    newdims.pop(newdims.index(rotation_dim))
    newdims = newdims + [rotation_dim]
    da = da.transpose(*newdims)

    # Grab Units
    units = None
    if isinstance(da.data, Quantity):
        units = da.data.units

    # Handle special case of non-dimnesional DataArray set rotation to be the item
    if isinstance(rotation, xr.DataArray):
        if rotation.dims == ():
            rotation = rotation.item()

    # Grab Shape and ravel
    shape = da.shape
    data = da.data.reshape(-1, 3)

    # Make sure data is type float
    if data.dtype == "object":
        data = data.astype(np.float64)

    # Rotate data
    newdata = rotation.apply(data, inverse=inverse)

    # Reshape and assign rotated data units if needed
    newdata = newdata.reshape(shape)

    # Add back in units
    if units is not None:
        newdata *= units

    # Assign back to da
    da.data = newdata

    # Make sure data is a float - sometimes it is an object array
    if isinstance(da.data, Quantity):
        if da.data.magnitude.dtype == "object":
            da.data = da.data.magnitude.astype(np.float64) * da.data.units
    elif da.data.dtype == "object":
        da.data = da.data.astype(np.float64)

    return da
