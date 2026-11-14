"""Granule readers — SMAP HDF5 layouts and ECOSTRESS COG tiles → sample.

Synthetic files mirror the documented layouts: SPL3SMP_E keeps 2-D
`latitude`/`longitude` inside `Soil_Moisture_Retrieval_Data_AM`; SPL4SMGP
keeps `cell_lat`/`cell_lon` in the root and data in `Geophysical_Data`;
ECOSTRESS tiles are one UTM GeoTIFF per variable.
"""

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from nasa_farm.acquire.crawl import _open_h5
from nasa_farm.acquire.products import ECO_LST, SMAP_RZ, SMAP_SM
from nasa_farm.acquire.qc import CLOUD_QC, SMAP_QC
from nasa_farm.acquire.support import coverage, sample

LAT = np.linspace(43, 41, 30)[:, None] * np.ones((1, 40))
LON = np.ones((30, 1)) * np.linspace(-94.5, -92.5, 40)[None, :]


def test_smap_l4_root_coords(tmp_path):
    f = tmp_path / "SMAP_L4_SM_gph_20250701T133000_Vv8013_001.h5"
    with h5py.File(f, "w") as h:
        h["cell_lat"], h["cell_lon"] = LAT.astype("f4"), LON.astype("f4")
        h.create_group("Geophysical_Data")["sm_rootzone"] = \
            np.full((30, 40), 0.25, "f4")
    with _open_h5(str(f), SMAP_RZ) as ds:
        assert sample(ds, "sm_rootzone", 42.0, -93.5, 18.0) == \
            pytest.approx(0.25)
    with open(f, "rb") as fh, _open_h5(fh, SMAP_RZ) as ds:   # streamed
        assert SMAP_RZ.read(ds, 42.0, -93.5) == pytest.approx(0.25)


def test_smap_l3_latitude_longitude_and_qc(tmp_path):
    f = tmp_path / "SMAP_L3_SM_P_E_20250701_R19240_001.h5"
    q = np.zeros((30, 40), "u2")
    q[:, :20] = 7                                  # left half unusable
    with h5py.File(f, "w") as h:
        g = h.create_group("Soil_Moisture_Retrieval_Data_AM")
        g["latitude"], g["longitude"] = LAT.astype("f4"), LON.astype("f4")
        g["soil_moisture"] = np.full((30, 40), 0.25, "f4")
        g["retrieval_qual_flag"] = q
    with _open_h5(str(f), SMAP_SM) as ds:
        conf = SMAP_QC.mask(ds)
        assert sample(ds, "soil_moisture", 42.0, -93.5, 18.0, conf) == \
            pytest.approx(0.25)
        assert 0.0 < coverage(ds, "soil_moisture", 42.0, -93.5, 18.0,
                              conf) < 1.0


def test_ecostress_cog_tiles(tmp_path):
    rioxarray = pytest.importorskip("rioxarray")  # noqa: F841
    import xarray as xr
    from pyproj import Transformer

    from nasa_farm.acquire.crawl import _open_cog
    x = 445_000 + 70 * np.arange(200)
    y = 4_660_000 - 70 * np.arange(200)

    def tif(name, arr):
        da = xr.DataArray(arr[None], dims=("band", "y", "x"),
                          coords={"band": [1], "y": y, "x": x}
                          ).rio.write_crs(32615)
        p = tmp_path / f"ECOv002_L2T_LSTE_1_2_15TVG_20250701T180000_0713_01_{name}.tif"
        da.rio.to_raster(p)
        return str(p)

    cloud = np.zeros((200, 200), "u1")
    cloud[:, 100:] = 1
    ds = _open_cog([tif("LST", np.full((200, 200), 300.0, "f4")),
                    tif("cloud", cloud)])
    lon, lat = Transformer.from_crs(32615, 4326, always_xy=True).transform(
        445_000 + 70 * 50, 4_660_000 - 70 * 100)
    assert set(ds) >= {"LST", "cloud", "lat", "lon"}
    assert ECO_LST.read(ds, lat, lon) == pytest.approx(26.85)   # K → °C
    assert CLOUD_QC.mask(ds)[0, 150] == 0.0
