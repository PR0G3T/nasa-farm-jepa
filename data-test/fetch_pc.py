import requests, json, os, re
import numpy as np
from PIL import Image
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
RAW='/home/veldrx/Documents/Hackathon/data-test/raw/'
MAP='/home/veldrx/Documents/Hackathon/data-test/maps/'
log=lambda *a:print('▶',*a,flush=True)
BBOX=(-99.13,40.62,-98.97,40.78)
STAC="https://planetarycomputer.microsoft.com/api/stac/v1"
SAS="https://planetarycomputer.microsoft.com/api/sas/v1/sign"

def sign(href):
    try:
        r=requests.get(SAS,params={"href":href},timeout=30)
        return r.json().get('href',href)
    except Exception:
        return href

def read_win(href,bbox=BBOX):
    with rasterio.open(sign(href)) as ds:
        win=from_bounds(*bbox,transform=ds.transform)
        data=ds.read(1,window=win)
        wtf=ds.window_transform(win)
        try: nodata=ds.nodata
        except: nodata=None
    return data,wtf,nodata

def stac_search(coll,bbox,dt,q=None):
    body={"collections":[coll],"bbox":bbox,"datetime":dt,"limit":10}
    if q: body["query"]=q
    r=requests.post(STAC+"/search",json=body,timeout=90); r.raise_for_status()
    return r.json()['features']

# ── Liste des collections dispo (vérif noms) ──
try:
    r=requests.get(STAC+"/collections",params={"limit":300},timeout=60)
    cols={c['id'] for c in r.json()['collections']}
    for want in ["sentinel-2-l2a","landsat-c2-l2","cop-dem-glo-30","naip","nass-cdl","modis-13Q1-061","modis-64A1-061","esa-worldcover","nlcd","hls"]:
        log("PC collection",want,":",want in cols or [c for c in cols if want.split('-')[0] in c][:5])
except Exception as e: log("PC list FAIL",e)

# ── Sentinel-2 L2A : meilleure scène juillet 2025 ──
try:
    feats=stac_search("sentinel-2-l2a",BBOX,"2025-07-01/2025-07-31",
                      {"eo:cloud_cover":{"lt":20}})
    feats.sort(key=lambda f:f['properties'].get('eo:cloud_cover',99))
    log("S2 scenes:",[(f['id'][:30],round(f['properties'].get('eo:cloud_cover',-1),1)) for f in feats[:3]])
    if feats:
        f=feats[0]; a=f['assets']
        b4,_,_=read_win(a['B04']['href']); b8,_,_=read_win(a['B08']['href'])
        b5,_,_=read_win(a['B05']['href']); b3,_,_=read_win(a['B03']['href']); b2,_,_=read_win(a['B02']['href'])
        np.savez_compressed(RAW+'s2_bands.npz',B02=b2,B03=b3,B04=b4,B05=b5,B08=b8,
                            meta=json.dumps({"id":f['id'],"date":f['properties']['datetime'],"cloud":f['properties'].get('eo:cloud_cover')}))
        log("S2 bandes:",b4.shape,"scene:",f['id'][:40])
except Exception as e: log("S2 FAIL",repr(e)[:200])

# ── Landsat 8/9 C2 L2 ──
try:
    feats=stac_search("landsat-c2-l2",BBOX,"2025-07-01/2025-08-10",
                      {"eo:cloud_cover":{"lt":30},"platform":{"in":["landsat-8","landsat-9"]}})
    feats.sort(key=lambda f:f['properties'].get('eo:cloud_cover',99))
    log("Landsat scenes:",[(f['id'][:35],round(f['properties'].get('eo:cloud_cover',-1),1)) for f in feats[:3]])
    if feats:
        f=feats[0]; a=f['assets']
        red,_,_=read_win(a['red']['href']); nir,_,_=read_win(a['nir08']['href'])
        st,_,nod=read_win(a['lwir11']['href'])
        np.savez_compressed(RAW+'landsat_bands.npz',red=red,nir=nir,st=st,
                            meta=json.dumps({"id":f['id'],"date":f['properties']['datetime'],"cloud":f['properties'].get('eo:cloud_cover'),
                                             "red_scale":a['red'].get('raster:bands',[{}])[0].get('scale'),"red_offset":a['red'].get('raster:bands',[{}])[0].get('offset')}))
        log("Landsat:",red.shape,f['id'][:45])
except Exception as e: log("Landsat FAIL",repr(e)[:200])

# ── Copernicus DEM 30m ──
try:
    feats=stac_search("cop-dem-glo-30",BBOX,"2021-01-01/2021-12-31")
    if feats:
        dem,tf,nd=read_win(feats[0]['assets']['data']['href'])
        np.savez_compressed(RAW+'dem.npz',dem=dem)
        log("DEM:",dem.shape,"min/max:",np.nanmin(dem),np.nanmax(dem),"m")
except Exception as e: log("DEM FAIL",repr(e)[:200])

# ── NAIP ──
try:
    feats=stac_search("naip",BBOX,"2020-01-01/2025-12-31")
    if feats:
        feats.sort(key=lambda f:f['properties']['datetime'],reverse=True)
        f=feats[0]
        im,_,_=read_win(f['assets']['image']['href'])
        np.save(RAW+'naip.npy',im)
        log("NAIP:",im.shape,f['properties']['datetime'][:10],f['id'][:40])
except Exception as e: log("NAIP FAIL",repr(e)[:200])

# ── MODIS 13Q1 NDVI ──
try:
    feats=stac_search("modis-13Q1-061",BBOX,"2025-07-01/2025-07-31")
    if feats:
        f=feats[0]
        log("MODIS assets:",[k for k in f['assets'].keys()][:10])
        nd,_,_=read_win(f['assets']['NDVI']['href'])
        np.save(RAW+'modis_ndvi.npy',nd)
        log("MODIS NDVI:",nd.shape,f['id'][:40])
except Exception as e: log("MODIS FAIL",repr(e)[:200])

# ── NASS CDL sur PC (rotation pixel-level!) ──
try:
    feats=stac_search("nass-cdl",BBOX,"2024-01-01/2024-12-31")
    log("nass-cdl feats:",len(feats))
    if feats:
        c,_,_=read_win(feats[0]['assets']['data']['href'])
        np.save(RAW+'cdl_2024_pc.npy',c)
        u,cnt=np.unique(c,return_counts=True)
        top=sorted(zip(cnt,u),reverse=True)[:8]
        log("CDL PC:",c.shape,"top classes:",[(int(uu),int(cc)) for cc,uu in top])
except Exception as e: log("CDL PC FAIL",repr(e)[:200])

# ── GIBS retry : SMAP L4 rootzone + surface + GPP + GRACE dates + freeze/thaw hiver ──
gibs={
 "smap_rz":("SMAP_L4_Analyzed_Root_Zone_Soil_Moisture","2025-07-15T00:00:00Z"),
 "smap_surf":("SMAP_L4_Analyzed_Surface_Soil_Moisture","2025-07-15T00:00:00Z"),
 "smap_gpp":("SMAP_L4_Mean_Gross_Primary_Productivity","2025-07-15T00:00:00Z"),
 "smap_l3":("SMAP_L3_Passive_Enhanced_Day_Soil_Moisture","2025-07-13T00:00:00Z"),
 "grace_may":("GRACE_Tellus_Liquid_Water_Equivalent_Thickness_Mascon_CRI","2025-05-01T00:00:00Z"),
 "ft_jan":("SMAP_L3_Passive_Enhanced_Day_Freeze_Thaw","2025-01-15T00:00:00Z"),
}
for k,(lay,t) in gibs.items():
    try:
        u=(f"https://wvs.earthdata.nasa.gov/api/v1/snapshot?REQUEST=GetSnapshot&TIME={t}"
           f"&BBOX=39.90,-100.20,41.50,-97.90&CRS=EPSG:4326&LAYERS={lay},Coastlines_15m&FORMAT=image/png&WIDTH=1600&HEIGHT=1100")
        r=requests.get(u,timeout=120)
        if r.status_code==200 and 'png' in r.headers.get('Content-Type',''):
            open(RAW+f'gibs_{k}.png','wb').write(r.content)
            im=np.array(Image.open(RAW+f'gibs_{k}.png').convert('RGBA'))
            log(f"GIBS {k}: {len(r.content)}o, opaque={np.count_nonzero(im[...,3]>10)}")
        else: log(f"GIBS {k}: HTTP {r.status_code}")
    except Exception as e: log(f"GIBS {k} FAIL",e)

# ── ECOSTRESS via CMR browse ? ──
try:
    r=requests.get("https://cmr.earthdata.nasa.gov/search/granules.json",
        params={"short_name":"ECO_L3T_JET","bounding_box":"-99.5,40.3,-98.6,41.1",
                "temporal":"2025-06-01T00:00:00Z,2025-08-31T00:00:00Z","page_size":"3"},timeout=60)
    e=r.json()['feed']['entry']
    log("ECOSTRESS granules:",len(e))
    for g in e[:2]:
        links=[l['href'] for l in g.get('links',[]) if l.get('type')=='image/jpeg' or 'browse' in l.get('rel','')]
        log("  ",g['title'][:60],"browse:",links[:2])
except Exception as ex: log("ECOSTRESS CMR FAIL",ex)

# ── LANDFIRE WMS conus_2024 ──
try:
    r=requests.get("https://edcintl.cr.usgs.gov/geoserver/landfire/conus_2024/ows",
                   params={"service":"WMS","request":"GetCapabilities"},timeout=60)
    names=set(re.findall(r'<Name>([^<]+)</Name>',r.text))
    log("LANDFIRE layers:",sorted(n for n in names if 'LF' in n or 'landfire' in n)[:15])
except Exception as e: log("LF capa FAIL",e)
print("DONE")
