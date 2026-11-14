import json, os, time, requests, io
import numpy as np
import rasterio
from rasterio.windows import from_bounds, bounds as win_bounds
from rasterio.warp import transform_bounds
from PIL import Image
BASE='/home/veldrx/Documents/Hackathon/data-test/'
RAW=BASE+'raw/'
log=lambda *a:print('▶',*a,flush=True)
BBOX=(-99.13,40.62,-98.97,40.78)      # zone ~18 km
PARCEL=(-99.09,40.685,-99.01,40.735)  # zoom parcelle ~7 km
STAC="https://planetarycomputer.microsoft.com/api/stac/v1"
SAS="https://planetarycomputer.microsoft.com/api/sas/v1/sign"
EXT=json.load(open(BASE+'extents.json')) if os.path.exists(BASE+'extents.json') else {}
EXT['_meta']={"zone":list(BBOX),"parcel":list(PARCEL)}

def sign(href):
    for i in range(4):
        try: return requests.get(SAS,params={"href":href},timeout=30).json()['href']
        except Exception: time.sleep(3)
    return href
def stac(coll,dt,bbox=BBOX,limit=8):
    r=requests.post(STAC+"/search",json={"collections":[coll],"bbox":list(bbox),"datetime":dt,"limit":limit},timeout=60)
    return r.json().get('features',[])
def win_for(ds,bbox):
    bb=transform_bounds("EPSG:4326",ds.crs,*bbox)
    return from_bounds(*bb,transform=ds.transform).round_offsets().round_shape()
def record(key,ds,win,bbox):
    bds=win_bounds(win,ds.transform)
    lo,bo,ri,to=transform_bounds(ds.crs,"EPSG:4326",*bds)
    EXT[key]={"bounds_ll":[round(lo,4),round(bo,4),round(ri,4),round(to,4)],"res":f"{abs(ds.transform.a):.1f} m"}

# ── S2 grande zone + zoom parcelle ──
try:
    feats=sorted(stac("sentinel-2-l2a","2025-07-15/2025-07-25"),key=lambda f:f['properties'].get('eo:cloud_cover',99))
    f=feats[0]; log("S2 scene:",f['id'],f['properties'].get('eo:cloud_cover'))
    for tag,bb in [("s2",BBOX),("s2zoom",PARCEL)]:
        out={}
        for b in ["B02","B03","B04","B05","B08","B11","SCL"]:
            with rasterio.open(sign(f['assets'][b]['href'])) as ds:
                w=win_for(ds,bb)
                out[b]=ds.read(1,window=w)
                if b=="B02": record(tag,ds,w,bb)
        np.savez_compressed(RAW+f'{tag}_bands.npz',**out,meta=json.dumps({"id":f['id'],"date":f['properties']['datetime'],"cloud":f['properties'].get('eo:cloud_cover')}))
        log(tag,"ok",out['B02'].shape)
    EXT['s2']['scene']=f['id']; EXT['s2']['date']=f['properties']['datetime'][:10]
    EXT['s2zoom']['scene']=f['id']; EXT['s2zoom']['date']=f['properties']['datetime'][:10]
except Exception as e: log("S2 FAIL",repr(e)[:150])

# ── Landsat zone ──
try:
    feats=sorted(stac("landsat-c2-l2","2025-07-01/2025-07-15"),key=lambda f:f['properties'].get('eo:cloud_cover',99))
    f=feats[0]; out={}
    for b in ["red","nir08","swir16","lwir11"]:
        with rasterio.open(sign(f['assets'][b]['href'])) as ds:
            w=win_for(ds,BBOX); out[b]=ds.read(1,window=w)
            if b=="red": record("landsat",ds,w,BBOX)
    EXT['landsat']['scene']=f['id']; EXT['landsat']['date']=f['properties']['datetime'][:10]
    EXT['landsat']['scale']={'red':2.75e-5,'lwir11':0.00341802}; EXT['landsat']['offset']={'red':-0.2,'lwir11':149.0}
    np.savez_compressed(RAW+'landsat_bands.npz',**out,meta=json.dumps({"id":f['id'],"date":f['properties']['datetime']}))
    log("landsat ok",out['red'].shape)
except Exception as e: log("landsat FAIL",repr(e)[:150])

# ── MODIS NDVI région élargie (250m sur 18km = peu de pixels → 1°) ──
try:
    MB=(-100.0,40.0,-98.0,41.5)
    feats=stac("modis-13Q1-061","2025-07-01/2025-07-31",bbox=MB)
    f=feats[0]
    with rasterio.open(sign(f['assets']['250m_16_days_NDVI']['href'])) as ds:
        w=win_for(ds,MB); d=ds.read(1,window=w); record("modis",ds,w,MB)
    EXT['modis']['scene']=f['id']; EXT['modis']['date']="2025-07 (16j)"
    np.savez_compressed(RAW+'modis13.npz',ndvi=d,meta=json.dumps({"id":f['id']}))
    log("modis ok",d.shape)
except Exception as e: log("modis FAIL",repr(e)[:150])

# ── DEM zone ──
try:
    feats=stac("cop-dem-glo-30","2020-01-01/2025-12-31")
    f=feats[0]
    with rasterio.open(sign(f['assets']['data']['href'])) as ds:
        w=win_for(ds,BBOX); d=ds.read(1,window=w); record("dem",ds,w,BBOX)
    np.savez_compressed(RAW+'dem.npz',dem=d); log("dem ok",d.shape,"alt",float(np.nanmin(d)),"-",float(np.nanmax(d)))
except Exception as e: log("dem FAIL",repr(e)[:150])

# ── NAIP zoom parcelle (décimé ×2) ──
try:
    feats=stac("naip","2022-01-01/2022-12-31",bbox=PARCEL)
    f=sorted(feats,key=lambda f:f['properties']['datetime'])[-1]
    with rasterio.open(sign(f['assets']['image']['href'])) as ds:
        w=win_for(ds,PARCEL)
        nh,nw=int(w.height//2),int(w.width//2)
        d=ds.read(window=w,out_shape=(ds.count,nh,nw))
        record("naip",ds,w,PARCEL); EXT['naip']['res']=f"{abs(ds.transform.a)*2:.1f} m (décimé)"
        EXT['naip']['scene']=f['id']; EXT['naip']['date']=f['properties']['datetime'][:10]
    np.save(RAW+'naip.npy',d); log("naip ok",d.shape,EXT['naip']['date'])
except Exception as e: log("naip FAIL",repr(e)[:150])

# ── ECOSTRESS : emprises granules via CMR ──
try:
    eco={}
    for coll,short in [("C3550088218-ECOSTRESS","ECO_L3T_JET"),("C3550088222-ECOSTRESS","ECO_L4T_ESI"),("C3550088272-ECOSTRESS","ECO_L2T_LSTE")]:
        r=requests.get("https://cmr.earthdata.nasa.gov/search/granules.json",
            params={"collection_concept_id":coll,"bounding_box":"-99.2,40.6,-98.9,40.8",
                    "temporal":"2025-05-25T00:00:00Z,2025-06-30T00:00:00Z","page_size":5},timeout=60)
        items=r.json().get('feed',{}).get('entry',[])
        if items:
            g=items[0]
            bb=g.get('boxes',[None])[0]
            eco[short]={"id":g['id'][:60],"date":g.get('time_start','')[:10],"bbox":bb}
            log("ECO",short,g.get('time_start','')[:10],"bbox:",bb)
    EXT['ecostress']=eco
except Exception as e: log("ECO CMR FAIL",repr(e)[:150])

# ── ERA5 : ré-extract régional avec coords explicites ──
try:
    import xarray as xr
    ds=xr.open_dataset("gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
        engine="zarr",storage_options={"token":"anon"},chunks=None)
    t0=time.time()
    sub=ds['volumetric_soil_water_layer_1'].sel(time=slice("2025-07-14","2025-07-16"),
          latitude=slice(41.0,40.25),longitude=slice(260.0,261.5))
    v=sub.values  # (3h?, lats, lons)
    lats=sub.latitude.values; lons=sub.longitude.values
    np.savez_compressed(RAW+'era5_map.npz',sm=v,lats=lats,lons=lons)
    EXT['era5']={"bounds_ll":[round(float(lons.min()-360),3),round(float(lats.min()),3),
                 round(float(lons.max()-360),3),round(float(lats.max()),3)],
                 "res":"0.25° (~28 km)","date":"2025-07-15"}
    log("era5 map ok",v.shape,f"{time.time()-t0:.0f}s")
except Exception as e: log("ERA5 FAIL",repr(e)[:150])

json.dump(EXT,open(BASE+'extents.json','w'),indent=1)
print("DONE EXT:",list(EXT.keys()))
