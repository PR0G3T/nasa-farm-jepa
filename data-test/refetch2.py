import json, os, time, requests
import numpy as np
import rasterio
from rasterio.windows import from_bounds, bounds as win_bounds, Window
from rasterio.warp import transform_bounds
BASE='/home/veldrx/Documents/Hackathon/data-test/'
RAW=BASE+'raw/'
log=lambda *a:print('▶',*a,flush=True)
BBOX=(-99.13,40.62,-98.97,40.78); PARCEL=(-99.09,40.685,-99.01,40.735)
STAC="https://planetarycomputer.microsoft.com/api/stac/v1"
SAS="https://planetarycomputer.microsoft.com/api/sas/v1/sign"
EXT=json.load(open(BASE+'extents.json'))
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
    w=from_bounds(*bb,transform=ds.transform).round_offsets().round_shape()
    full=Window(0,0,ds.width,ds.height)
    return w.intersection(full)
def record(key,ds,w):
    bds=win_bounds(w,ds.transform)
    lo,bo,ri,to=transform_bounds(ds.crs,"EPSG:4326",*bds)
    EXT[key]={"bounds_ll":[round(lo,4),round(bo,4),round(ri,4),round(to,4)],"res":f"{abs(ds.transform.a):.1f} m"}
def open_retry(href,tries=6):
    for i in range(tries):
        try: return rasterio.open(sign(href))
        except rasterio.errors.RasterioIOError:
            time.sleep(4+i*3)
    raise RuntimeError("409 persistant")

# S2 zone + zoom (retry 409)
try:
    feats=sorted(stac("sentinel-2-l2a","2025-07-15/2025-07-25"),key=lambda f:f['properties'].get('eo:cloud_cover',99))
    f=feats[0]; log("S2:",f['id'],"cloud",f['properties'].get('eo:cloud_cover'))
    for tag,bb in [("s2",BBOX),("s2zoom",PARCEL)]:
        out={}
        for b in ["B02","B03","B04","B05","B08","B11","SCL"]:
            with open_retry(f['assets'][b]['href']) as ds:
                w=win_for(ds,bb); out[b]=ds.read(1,window=w)
                if b=="B02": record(tag,ds,w)
        np.savez_compressed(RAW+f'{tag}_bands.npz',**out,meta=json.dumps({"id":f['id'],"date":f['properties']['datetime'],"cloud":f['properties'].get('eo:cloud_cover')}))
        EXT[tag]['scene']=f['id']; EXT[tag]['date']=f['properties']['datetime'][:10]
        log(tag,"ok",out['B02'].shape)
except Exception as e: log("S2 FAIL",repr(e)[:150])

# MODIS borné à la tuile
try:
    MB=(-100.0,40.0,-98.0,41.5)
    feats=stac("modis-13Q1-061","2025-07-01/2025-07-31",bbox=MB)
    f=feats[0]
    with open_retry(f['assets']['250m_16_days_NDVI']['href']) as ds:
        w=win_for(ds,MB); d=ds.read(1,window=w); record("modis",ds,w)
    EXT['modis']['scene']=f['id']; EXT['modis']['date']="2025-07 (16j)"
    np.savez_compressed(RAW+'modis13.npz',ndvi=d,meta=json.dumps({"id":f['id']}))
    log("modis ok",d.shape,"valid",int((d>-3000).sum()))
except Exception as e: log("modis FAIL",repr(e)[:150])

# ECOSTRESS : trouver concept-id via collections short_name
try:
    eco={}
    for sn in ["ECO_L3T_JET","ECO_L4T_ESI","ECO_L2T_LSTE"]:
        r=requests.get("https://cmr.earthdata.nasa.gov/search/collections.json",
            params={"short_name":sn,"provider":"LPCLOUD"},timeout=60)
        cols=r.json().get('feed',{}).get('entry',[])
        if not cols: log(sn,"no collection"); continue
        cid=cols[0]['id']
        r=requests.get("https://cmr.earthdata.nasa.gov/search/granules.json",
            params={"collection_concept_id":cid,"bounding_box":"-99.5,40.4,-98.6,41.0",
                    "temporal":"2025-05-20T00:00:00Z,2025-07-31T00:00:00Z","page_size":10,
                    "sort_key":"-start_date"},timeout=60)
        items=r.json().get('feed',{}).get('entry',[])
        if items:
            # prendre le granule dont le browse existe
            g=items[0]; bb=[float(x) for x in g['boxes'][0].split()]
            south,west,north,east=bb  # CMR: S W N E
            eco[sn]={"id":g['id'][:70],"date":g.get('time_start','')[:10],
                     "bounds_ll":[west,south,east,north]}
            log("ECO",sn,g.get('time_start','')[:10],"bbox",[west,south,east,north])
        else: log(sn,"no granule")
    EXT['ecostress']=eco
except Exception as e: log("ECO FAIL",repr(e)[:150])

json.dump(EXT,open(BASE+'extents.json','w'),indent=1)
print("DONE:",{k:v.get('bounds_ll') for k,v in EXT.items() if isinstance(v,dict) and 'bounds_ll' in v})
