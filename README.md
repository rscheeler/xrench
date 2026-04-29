# 🔧 xrench

Utility functions for [pint](https://pint.readthedocs.io/) and [xarray](https://docs.xarray.dev/) to support RF and microwave design Python tools.

## Installation

```bash
uv add xrench
```

Or with pip:

```bash
pip install xrench
```

## Features

- **Unit-aware arrays** — Seamless integration between `pint` and `xarray` for physical quantities
- **Custom RF units** — Built-in support for dBK (decibel kelvin) and dBHz (decibel hertz)
- **Unit decorator** — `wraps_xr` decorator to strip and re-apply units around functions, preserving xarray coordinates and dimensions
- **Rotation utilities** — Apply `scipy` rotations to coordinate DataArrays, including polarization rotation
- **Logging control** — Mutable/unmutable logger via `XRENCHLogger` for clean library logging

## Usage

### Unit Registry

```python
from xrench.units import ureg

# Use built-in RF units
noise_temp = 290 * ureg.decibelkelvin
bandwidth = 1e6 * ureg.decibelhertz
```

### Convert kwargs to DataArrays

```python
from xrench.xrutils import kw2da

arrays = kw2da(frequency=freq_quantity, angle=angle_array)
```

### Unit-preserving function decorator

```python
from xrench.xrutils import wraps_xr

@wraps_xr(ret_units="meter", arg_units=["meter", "meter"])
def my_func(x, y):
    # work with unit-less DataArrays
    return x + y
```

### Logging

```python
from xrench.logcontrol import XRENCHLogger

XRENCHLogger.unmute()         # enable logs
XRENCHLogger.level = "DEBUG"  # change level
XRENCHLogger.mute()           # silence logs
```

## License

MIT

## Contact

Created by [Rob Scheeler](https://github.com/rscheeler)