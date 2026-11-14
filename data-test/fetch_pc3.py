import requests, json, time, re
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
from PIL import Image
RAW='/home/veldrx/Documents/Hackathon/data-test/raw/'
log=lambda *a:print('▶',*a,flush=True)
BBOX=(-99.13,40.62,-98.97,40.78)
STAC="https://planetarycomputer.microsoft.com/api/stac/v1"
SAS="https://planetarycomputer.microsoft.com/api/sas/v1/sign"
_tok={}
def sign(href):
    for i in range(3):
        try:
            r=requests.get(SAS,params={"href":href},timeout=30)
            h=r.json().get('href')
            if h: return h
        except Exception: pass
        time.sleep(2)
    return href

def read_win(href,bbox=BBOX):
    with rasterio.open(sign(href)) as ds:
        bb=transform_bounds("EPSG:4326",ds.crs,*bbox) if str(ds.crs).upper() not in ("EPSG:4326","OGC:CRS84") else bbox
        win=from_bounds(*bb,transform=ds.transform).round_offsets().round_shape()
        if win.width<1 or win.height<1: return None
        return ds.read(1,window=win)

def stac(coll,bbox,dt,q=None):
    body={"collections":[coll],"bbox":bbox,"datetime":dt,"limit":10}
    if q: body["query"]=q
    return requests.post(STAC+"/search",json=body,timeout=90).json()['features']

# ── Sentinel-2 ──
try:
    feats=stac("sentinel-2-l2a",BBOX,"2025-07-01/2025-07-31",{"eo:cloud_cover":{"lt":20}})
    feats.sort(key=lambda f:f['properties'].get('eo:cloud_cover',99))
    f=feats[0]; a=f['assets']
    bands={}
    for k in ['B02','B03','B04','B05','B08','B11','SCL']:
        bands[k]=read_win(a[k]['href']); time.sleep(1)
    bands={k:v for k,v in bands.items() if v is not None}
    np.savez_compressed(RAW+'s2_bands.npz',**bands,
        meta=json.dumps({"id":f['id'],"date":f['properties']['datetime'],"cloud":f['properties'].get('eo:cloud_cover')}))
    log("S2:",{k:v.shape for k,v in bands.items()},"|",f['properties']['datetime'][:10],"cloud",f['properties'].get('eo:cloud_cover'))
except Exception as e: log("S2 FAIL",repr(e)[:180])

# ── Landsat ──
try:
    feats=stac("landsat-c2-l2",BBOX,"2025-07-01/2025-08-10",{"eo:cloud_cover":{"lt":30}})
    feats.sort(key=lambda f:f['properties'].get('eo:cloud_cover',99))
    f=feats[0]; a=f['assets']
    bands={}
    for k in ['red','nir08','swir16','lwir11']:
        bands[k]=read_win(a[k]['href']); time.sleep(1)
    bands={k:v for k,v in bands.items() if v is not None}
    meta={"id":f['id'],"date":f['properties']['datetime'],"cloud":f['properties'].get('eo:cloud_cover'),
          "scale":{k:a[k].get('raster:bands',[{}])[0].get('scale') for k in ['red','lwir11']},
          "offset":{k:a[k].get('raster:bands',[{}])[0].get('offset') for k in ['red','lwir11']}}
    np.savez_compressed(RAW+'landsat_bands.npz',**bands,meta=json.dumps(meta))
    log("Landsat:",{k:v.shape for k,v in bands.items()},"|",f['properties']['datetime'][:10],meta['scale'])
except Exception as e: log("Landsat FAIL",repr(e)[:180])

# ── MODIS 13Q1 ──
try:
    feats=stac("modis-13Q1-061",BBOX,"2025-07-01/2025-07-31")
    f=feats[0]
    ndvi=read_win(f['assets']['250m_16_days_NDVI']['href']); time.sleep(1)
    evi=read_win(f['assets']['250m_16_days_EVI']['href'])
    np.savez_compressed(RAW+'modis13.npz',ndvi=ndvi,evi=evi,
        meta=json.dumps({"id":f['id'],"date":f['properties']['datetime']}))
    log("MODIS:",ndvi.shape if ndvi is not None else None,f['properties']['datetime'][:10])
except Exception as e: log("MODIS FAIL",repr(e)[:180])

# ── GRACE à 2022-07-01 (default) + 2021 ──
for t in ["2022-07-01T00:00:00Z","2021-07-01T00:00:00Z","2020-07-01T00:00:00Z"]:
    try:
        u=(f"https://wvs.earthdata.nasa.gov/api/v1/snapshot?REQUEST=GetSnapshot&TIME={t}"
           f"&BBOX=38.5,-101.5,42.5,-96.5&CRS=EPSG:4326&LAYERS=GRACE_Tellus_Liquid_Water_Equivalent_Thickness_Mascon_CRI,Coastlines_15m&FORMAT=image/png&WIDTH=1600&HEIGHT=1200")
        r=requests.get(u,timeout=120)
        if r.status_code==200 and 'png' in r.headers.get('Content-Type',''):
            im=np.array(Image.open(__import__('io').BytesIO(r.content)).convert('RGBA'))
            op=np.count_nonzero(im[...,3]>10)
            log(f"GRACE {t[:7]}: {len(r.content)}o opaque={op}")
            if op>1000:
                open(RAW+'gibs_grace.png','wb').write(r.content); log("  → sauvegardé"); break
    except Exception as e: log("GRACE FAIL",e)

# ── ERA5 via arco-era5 zarr (anonyme) ──
try:
    import zarr, gcsfs
    fs=gcsfs.GCSFileSystem(token='anon')
    log("gcsfs anon ok")
except Exception as e: log("ERA5 gcsfs FAIL",repr(e)[:120])
