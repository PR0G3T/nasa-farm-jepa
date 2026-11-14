import json, os, time, requests
import numpy as np
import rasterio
from rasterio.windows import from_bounds, bounds as win_bounds
from rasterio.warp import transform_bounds
BASE='/home/veldrx/Documents/Hackathon/data-test/'
RAW=BASE+'raw/'
log=lambda *a:print('▶',*a,flush=True)
BBOX=(-99.13,40.62,-98.97,40.78)          # parcelle
STAC="https://planetarycomputer.microsoft.com/api/stac/v1"
SAS="https://planetarycomputer.microsoft.com/api/sas/v1/sign"
EXT={}

def sign(href):
    for i in range(3):
        try: return requests.get(SAS,params={"href":href},timeout=30).json()['href']
        except Exception: time.sleep(3)
    return href

def stac_search(coll,bbox=BBOX,dt="2025-07-01/2025-07-31",limit=5):
    r=requests.post(STAC+"/search",json={"collections":[coll],"bbox":list(bbox),
        "datetime":dt,"limit":limit},timeout=60)
    return r.json().get('features',[])

def raster_extent(key,coll,asset,dt,shape):
    """Recompute the window used for a saved array → exact bounds+res."""
    try:
        feats=stac_search(coll,dt=dt)
        if not feats: log(key,"no feats"); return
        # choisir le même type de scène que dans meta si possible
        f=feats[0]
        href=f['assets'][asset]['href']
        with rasterio.open(sign(href)) as ds:
            bb=transform_bounds("EPSG:4326",ds.crs,*BBOX)
            win=from_bounds(*bb,transform=ds.transform).round_offsets().round_shape()
            w,h=win.width,win.height
            if (h,w)!=shape: log(key,f"shape mismatch {shape} vs {(h,w)} — fenêtre recalculée quand même")
            bds=win_bounds(win,ds.transform)
            lo,bo,ri,to=transform_bounds(ds.crs,"EPSG:4326",*bds)
            EXT[key]={"bounds_ll":[round(lo,4),round(bo,4),round(ri,4),round(to,4)],
                      "crs":str(ds.crs)[:30],"res":f"{abs(ds.transform.a):.1f} m",
                      "scene":f['id'],"date":f['properties']['datetime'][:10]}
            log(key,"→",EXT[key]['bounds_ll'],EXT[key]['res'])
    except Exception as e: log(key,"FAIL",repr(e)[:150])

z=np.load(RAW+'s2_bands.npz'); raster_extent("s2","sentinel-2-l2a","B02","2025-07-15/2025-07-25",z['B02'].shape)
z=np.load(RAW+'landsat_bands.npz'); raster_extent("landsat","landsat-c2-l2","red","2025-07-01/2025-07-10",z['red'].shape)
z=np.load(RAW+'modis13.npz'); raster_extent("modis","modis-13Q1-061","250m_16_days_NDVI","2025-07-01/2025-07-31",z['ndvi'].shape)
z=np.load(RAW+'dem.npz'); raster_extent("dem","cop-dem-glo-30","data","2020-01-01/2025-12-31",z['dem'].shape)
naip=np.load(RAW+'naip.npy'); raster_extent("naip","naip","image","2022-01-01/2022-12-31",naip.shape[1:])

json.dump(EXT,open(BASE+'extents.json','w'),indent=1)
print("EXTENTS:",list(EXT.keys()))
