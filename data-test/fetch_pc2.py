import requests, json, os, re
import numpy as np
from PIL import Image
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
RAW='/home/veldrx/Documents/Hackathon/data-test/raw/'
log=lambda *a:print('▶',*a,flush=True)
BBOX=(-99.13,40.62,-98.97,40.78)
STAC="https://planetarycomputer.microsoft.com/api/stac/v1"
SAS="https://planetarycomputer.microsoft.com/api/sas/v1/sign"

def sign(href):
    try:
        r=requests.get(SAS,params={"href":href},timeout=30)
        return r.json().get('href',href)
    except Exception: return href

def read_win(href,bbox=BBOX):
    with rasterio.open(sign(href)) as ds:
        if str(ds.crs).upper() not in ("EPSG:4326","OGC:CRS84"):
            bb=transform_bounds("EPSG:4326",ds.crs,*bbox)
        else: bb=bbox
        win=from_bounds(*bb,transform=ds.transform).round_offsets().round_shape()
        if win.width<1 or win.height<1: return None
        return ds.read(1,window=win)

def stac(coll,bbox,dt,q=None):
    body={"collections":[coll],"bbox":bbox,"datetime":dt,"limit":10}
    if q: body["query"]=q
    return requests.post(STAC+"/search",json=body,timeout=90).json()['features']

# S2 (reprojection fixée)
try:
    feats=stac("sentinel-2-l2a",BBOX,"2025-07-01/2025-07-31",{"eo:cloud_cover":{"lt":20}})
    feats.sort(key=lambda f:f['properties'].get('eo:cloud_cover',99))
    f=feats[0]; a=f['assets']
    bands={k:read_win(a[k]['href']) for k in ['B02','B03','B04','B05','B08','B11','SCL']}
    if bands['B04'] is not None and bands['B04'].size:
        np.savez_compressed(RAW+'s2_bands.npz',**{k:v for k,v in bands.items() if v is not None},
            meta=json.dumps({"id":f['id'],"date":f['properties']['datetime'],"cloud":f['properties'].get('eo:cloud_cover')}))
        log("S2:",{k:v.shape for k,v in bands.items() if v is not None},"| scene:",f['id'][:45],"cloud:",f['properties'].get('eo:cloud_cover'))
except Exception as e: log("S2 FAIL",repr(e)[:200])

# Landsat
try:
    feats=stac("landsat-c2-l2",BBOX,"2025-07-01/2025-08-10",{"eo:cloud_cover":{"lt":30}})
    feats.sort(key=lambda f:f['properties'].get('eo:cloud_cover',99))
    f=feats[0]; a=f['assets']
    bands={k:read_win(a[k]['href']) for k in ['red','nir08','swir16','lwir11','qa_pixel']}
    if bands['red'] is not None and bands['red'].size:
        meta={"id":f['id'],"date":f['properties']['datetime'],"cloud":f['properties'].get('eo:cloud_cover'),
              "scale":{k:a[k].get('raster:bands',[{}])[0].get('scale') for k in ['red','lwir11']},
              "offset":{k:a[k].get('raster:bands',[{}])[0].get('offset') for k in ['red','lwir11']}}
        np.savez_compressed(RAW+'landsat_bands.npz',**{k:v for k,v in bands.items() if v is not None},meta=json.dumps(meta))
        log("Landsat:",bands['red'].shape,f['id'][:45])
except Exception as e: log("Landsat FAIL",repr(e)[:200])

# NAIP
try:
    feats=stac("naip",BBOX,"2020-01-01/2025-12-31")
    feats.sort(key=lambda f:f['properties']['datetime'],reverse=True)
    f=feats[0]
    with rasterio.open(sign(f['assets']['image']['href'])) as ds:
        bb=transform_bounds("EPSG:4326",ds.crs,*BBOX)
        win=from_bounds(*bb,transform=ds.transform).round_offsets().round_shape()
        im=ds.read(window=win)
    np.save(RAW+'naip.npy',im)
    log("NAIP:",im.shape,f['properties']['datetime'][:10],f['id'][:45])
except Exception as e: log("NAIP FAIL",repr(e)[:200])

# MODIS 13Q1 NDVI+EVI
try:
    feats=stac("modis-13Q1-061",BBOX,"2025-07-01/2025-07-31")
    f=feats[0]
    ndvi=read_win(f['assets']['250m_16_days_NDVI']['href'])
    evi=read_win(f['assets']['250m_16_days_EVI']['href'])
    np.savez_compressed(RAW+'modis13.npz',ndvi=ndvi,evi=evi,
        meta=json.dumps({"id":f['id'],"date":f['properties']['datetime'],
        "scale":f['assets']['250m_16_days_NDVI'].get('raster:bands',[{}])[0].get('scale')}))
    log("MODIS13:",ndvi.shape if ndvi is not None else None,f['id'][:40])
except Exception as e: log("MODIS FAIL",repr(e)[:200])

# ECOSTRESS browse PNGs (publics!)
try:
    for sn in ["ECO_L3T_JET","ECO_L4T_ESI","ECO_L2T_LSTE"]:
        r=requests.get("https://cmr.earthdata.nasa.gov/search/granules.json",
            params={"short_name":sn,"bounding_box":"-99.5,40.3,-98.6,41.1",
                    "temporal":"2025-05-01T00:00:00Z,2025-08-31T00:00:00Z","page_size":"10"},timeout=60)
        e=r.json()['feed']['entry']
        log(sn,"granules:",len(e))
        got=0
        for g in e:
            links=[l['href'] for l in g.get('links',[]) if l['href'].endswith('.png')]
            if links and got<1:
                img=requests.get(links[0],timeout=120)
                if img.status_code==200 and len(img.content)>3000:
                    open(RAW+f'ecostress_{sn}.png','wb').write(img.content)
                    log("  →",g['title'][:70],len(img.content),"o"); got+=1
except Exception as e: log("ECOSTRESS FAIL",repr(e)[:200])

# LANDFIRE EVT GetMap
try:
    u=("https://edcintl.cr.usgs.gov/geoserver/landfire/conus_2024/ows?service=WMS&version=1.1.1&request=GetMap"
       "&layers=landfire:LF2024_EVT_CONUS&styles=&bbox=-11030800,4946000,-11012800,4974000&width=900&height=900"
       "&srs=EPSG:3857&format=image/png")
    r=requests.get(u,timeout=120)
    if r.status_code==200 and 'png' in r.headers.get('Content-Type',''):
        open(RAW+'landfire_evt.png','wb').write(r.content); log("LANDFIRE EVT:",len(r.content),"o")
    else: log("LANDFIRE:",r.status_code,r.text[:100])
except Exception as e: log("LANDFIRE FAIL",e)

# GRACE : dimension temporelle du layer
try:
    r=requests.get("https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/1.0.0/WMTSCapabilities.xml",timeout=120)
    m=re.search(r'<ows:Identifier>GRACE_Tellus[^<]*</ows:Identifier>.*?<Dimension>(.*?)</Dimension>',r.text,re.S)
    log("GRACE time dim:",m.group(1)[:200] if m else "n/a")
except Exception as e: log("dim FAIL",e)

# GAEZ : S3 public ?
try:
    for u in ["https://s3.eu-west-1.amazonaws.com/data.gaezdev.aws.fao.org/?list-type=2&max-keys=5",
              "https://gaez.fao.org/pages/data-and-model"]:
        r=requests.get(u,timeout=60); log("GAEZ probe:",r.status_code,u[:55],r.text[:80].replace('\n',' '))
except Exception as e: log("GAEZ FAIL",e)
print("DONE")
