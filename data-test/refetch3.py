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
def stac(coll,dt,bbox=BBOX,limit=20):
    r=requests.post(STAC+"/search",json={"collections":[coll],"bbox":list(bbox),"datetime":dt,"limit":limit},timeout=60)
    return r.json().get('features',[])
def win_clip(ds,bbox):
    bb=transform_bounds("EPSG:4326",ds.crs,*bbox)
    w=from_bounds(*bb,transform=ds.transform).round_offsets().round_shape()
    return w.intersection(Window(0,0,ds.width,ds.height))
def open_retry(href,tries=6):
    for i in range(tries):
        try: return rasterio.open(sign(href))
        except rasterio.errors.RasterioIOError: time.sleep(5+i*3)
    raise RuntimeError("409")
def rec(key,ds,w,extra={}):
    lo,bo,ri,to=transform_bounds(ds.crs,"EPSG:4326",*win_bounds(w,ds.transform))
    EXT[key]={"bounds_ll":[round(lo,4),round(bo,4),round(ri,4),round(to,4)],"res":f"{abs(ds.transform.a):.1f} m",**extra}

# S2 : choisir la scène qui couvre le mieux le BBOX
try:
    feats=stac("sentinel-2-l2a","2025-06-25/2025-08-15")
    best=None
    for f in sorted(feats,key=lambda f:f['properties'].get('eo:cloud_cover',99))[:12]:
        try:
            with open_retry(f['assets']['B02']['href']) as ds:
                w=win_clip(ds,BBOX)
                cov=w.width*w.height
                log("candidat",f['properties']['datetime'][:10],f['id'].split('_')[2],"cloud",round(f['properties'].get('eo:cloud_cover',0),1),"→ fenêtre",w.width,"x",w.height)
                if best is None or cov>best[0]: best=(cov,f,ds.crs)
            if best and cov>0.95*BBOX[2] and False: pass
        except Exception as e: log("candidat err",repr(e)[:80])
        if best and best[0]>1_500_000: break
    cov,f,_=best; log("choisi:",f['id'],"couverture px:",cov)
    for tag,bb in [("s2",BBOX),("s2zoom",PARCEL)]:
        out={}
        for b in ["B02","B03","B04","B05","B08","B11","SCL"]:
            with open_retry(f['assets'][b]['href']) as ds:
                w=win_clip(ds,bb); out[b]=ds.read(1,window=w)
                if b=="B02": rec(tag,ds,w,{"scene":f['id'],"date":f['properties']['datetime'][:10]})
        np.savez_compressed(RAW+f'{tag}_bands.npz',**out,meta=json.dumps({"id":f['id'],"date":f['properties']['datetime'],"cloud":f['properties'].get('eo:cloud_cover')}))
        log(tag,"ok",out['B02'].shape)
except Exception as e: log("S2 FAIL",repr(e)[:200])

# MODIS : filtrer tuile h10v04
try:
    feats=stac("modis-13Q1-061","2025-07-01/2025-07-31",bbox=(-100,40,-98,41.5))
    for f in feats:
        if "h10v04" not in f['id']: continue
        with open_retry(f['assets']['250m_16_days_NDVI']['href']) as ds:
            w=win_clip(ds,(-100.0,40.0,-98.0,41.5))
            d=ds.read(1,window=w)
            if d.size and (d>-3000).any():
                rec("modis",ds,w,{"scene":f['id'],"date":"2025-07 (16j)"})
                np.savez_compressed(RAW+'modis13.npz',ndvi=d,meta=json.dumps({"id":f['id']}))
                log("modis ok",d.shape,"valid",int((d>-3000).sum())); break
    else: log("modis: aucun granule valide")
except Exception as e: log("modis FAIL",repr(e)[:150])

# ECOSTRESS : granule contenant le point
try:
    pt=(-99.05,40.70); eco={}
    for sn in ["ECO_L3T_JET","ECO_L4T_ESI","ECO_L2T_LSTE"]:
        cols=requests.get("https://cmr.earthdata.nasa.gov/search/collections.json",
            params={"short_name":sn,"provider":"LPCLOUD"},timeout=60).json().get('feed',{}).get('entry',[])
        if not cols: continue
        items=requests.get("https://cmr.earthdata.nasa.gov/search/granules.json",
            params={"collection_concept_id":cols[0]['id'],"bounding_box":"-99.5,40.4,-98.6,41.0",
                    "temporal":"2025-05-20T00:00:00Z,2025-07-31T00:00:00Z","page_size":30},timeout=60).json().get('feed',{}).get('entry',[])
        ok=[g for g in items if g.get('boxes') and (lambda b:float(b[1])<=pt[0]<=float(b[3]) and float(b[0])<=pt[1]<=float(b[2]))(g['boxes'][0].split())]
        g=(ok or items)[0]; b=[float(x) for x in g['boxes'][0].split()]
        eco[sn]={"id":g['id'][:70],"date":g.get('time_start','')[:10],"bounds_ll":[b[1],b[0],b[3],b[2]],
                 "contains_point":bool(ok)}
        log("ECO",sn,g.get('time_start','')[:10],"contient point:",bool(ok))
    EXT['ecostress']=eco
except Exception as e: log("ECO FAIL",repr(e)[:150])

json.dump(EXT,open(BASE+'extents.json','w'),indent=1)
print("DONE")
