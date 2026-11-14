"""Fetch #1 — APIs sans login. Parcelle Nebraska: bbox [-99.13,40.62,-98.97,40.78], pt -99.05,40.70"""
import requests, json, os, sys
R=lambda d,f: open(f'/home/veldrx/Documents/Hackathon/data-test/{d}/{f}','wb')
RAW='/home/veldrx/Documents/Hackathon/data-test/raw/'
os.makedirs(RAW,exist_ok=True)
BBOX="-99.13,40.62,-98.97,40.78"
log=lambda *a:print('▶',*a)

# ── 1. NASA POWER — daily AG params, été 2025 + monthly 10 ans ──────────────
try:
    p=dict(parameters="T2M,T2M_MAX,T2M_MIN,PRECTOTCORR,ALLSKY_SFC_SW_DWN,RH2M,WS2M,GWETROOT,GWETTOP,FROST_DAYS,EVPTRS,TSOIL1",
           community="AG",latitude=40.70,longitude=-99.05,start="20250601",end="20250831",format="JSON")
    r=requests.get("https://power.larc.nasa.gov/api/temporal/daily/point",params=p,timeout=120); r.raise_for_status()
    open(RAW+'power_daily_2025.json','w').write(r.text)
    d=r.json()['properties']['parameter']; log("POWER daily: OK,",len(d['T2M']),"jours, params:",list(d.keys()))
except Exception as e: log("POWER daily FAIL",e)
try:
    p=dict(parameters="T2M,PRECTOTCORR,ALLSKY_SFC_SW_DWN,GWETROOT",community="AG",
           latitude=40.70,longitude=-99.05,start="201501",end="202510",format="JSON")
    r=requests.get("https://power.larc.nasa.gov/api/temporal/monthly/point",params=p,timeout=120); r.raise_for_status()
    open(RAW+'power_monthly_10y.json','w').write(r.text)
    log("POWER monthly: OK,",len(r.json()['properties']['parameter']['T2M']),"mois")
except Exception as e: log("POWER monthly FAIL",e)

# ── 2. CropScape CDL — GeoTIFF 2024 + stats 2015-2024 ───────────────────────
try:
    r=requests.get("https://nassgeodata.gmu.edu/axis2/services/CDLService/GetCDLFile",
                   params={"year":"2024","bbox":BBOX},timeout=120)
    log("CDL GetCDLFile:",r.status_code,r.text[:120])
    if r.status_code==200 and 'return' in r.text:
        url=r.text.strip().split('<return>')[1].split('</return>')[0]
        t=requests.get(url,timeout=120); open(RAW+'cdl_2024.tif','wb').write(t.content)
        log("  GeoTIFF:",len(t.content),"octets")
except Exception as e: log("CDL file FAIL",e)
try:
    stats={}
    for y in range(2015,2025):
        r=requests.get("https://nassgeodata.gmu.edu/axis2/services/CDLService/GetCDLStat",
                       params={"year":str(y),"bbox":BBOX},timeout=60)
        stats[y]=r.text
    open(RAW+'cdl_stats_2015_2024.txt','w').write(json.dumps(stats))
    log("CDL stats 2015-2024: OK")
except Exception as e: log("CDL stats FAIL",e)

# ── 3. SDA / SSURGO — map units + propriétés sol ────────────────────────────
def sda(q):
    r=requests.post("https://sdmdataaccess.sc.egov.usda.gov/Tabular/post.rest",
                    data={"query":q,"format":"JSON"},timeout=120); r.raise_for_status(); return r.json()
try:
    q=f"""SELECT mukey, m.muacct FROM SDA_Get_Mukey_from_intersection_with_WktWgs84('POLYGON((-99.13 40.62,-98.97 40.62,-98.97 40.78,-99.13 40.78,-99.13 40.62))') g JOIN mapunit m ON m.mukey=g.mukey"""
    res=sda(q); open(RAW+'ssurgo_mukeys.json','w').write(json.dumps(res))
    keys=[row[0] for row in res['Table']]
    log("SDA mapunits:",len(keys),keys[:8])
except Exception as e: log("SDA mukey FAIL",e); keys=[]
if keys:
    try:
        kl=",".join(f"'{k}'" for k in keys[:60])
        q2=f"""SELECT TOP 200 m.mukey, m.musym, m.muname, c.compname, c.comppct_r, c.slope_r, c.taxorder,
               h.hzdept_r, h.hzdepb_r, h.om_r, h.dbthirdbar_r, h.awc_r, h.kwfact, h.drainagecl
               FROM mapunit m JOIN component c ON c.mukey=m.mukey LEFT JOIN chorizon h ON h.cokey=c.cokey
               WHERE m.mukey IN ({kl})"""
        res=sda(q2); open(RAW+'ssurgo_props.json','w').write(json.dumps(res))
        log("SDA props:",len(res.get('Table',[])),"lignes")
    except Exception as e: log("SDA props FAIL",e)
    try:  # rendements par culture (component crop yield)
        q3=f"""SELECT TOP 300 cc.mukey, c.compname, cc.cropname, cc.yldunits, cc.nonirryield_r, cc.irryield_r
               FROM cocropyld cc JOIN component c ON c.cokey=cc.cokey WHERE cc.mukey IN ({kl})"""
        res=sda(q3); open(RAW+'ssurgo_yields.json','w').write(json.dumps(res))
        log("SDA yields:",len(res.get('Table',[])),"lignes")
    except Exception as e: log("SDA yields FAIL",e)

# ── 4. USGS NWIS — stations + débits ────────────────────────────────────────
try:
    r=requests.get("https://waterservices.usgs.gov/nwis/iv/",params={"format":"json","bBox":BBOX,
        "parameterCd":"00060,00065,72019","siteStatus":"active"},timeout=60); r.raise_for_status()
    open(RAW+'nwis_sites.json','w').write(r.text)
    sites=[s['sourceInfo']['siteCode'][0]['value'] for s in r.json()['value']['timeSeries']]
    names={s['sourceInfo']['siteCode'][0]['value']:s['sourceInfo']['siteName'] for s in r.json()['value']['timeSeries']}
    log("NWIS sites:",sites[:6])
    if sites:
        r=requests.get("https://waterservices.usgs.gov/nwis/dv/",params={"format":"json","sites":",".join(sites[:5]),
            "startDT":"2025-06-01","endDT":"2025-08-31","parameterCd":"00060","siteStatus":"all"},timeout=60)
        open(RAW+'nwis_dv_2025.json','w').write(r.text); log("NWIS dv: OK")
        open(RAW+'nwis_names.json','w').write(json.dumps(names))
except Exception as e: log("NWIS FAIL",e)

# ── 5. GIBS snapshots — couches NASA prêtes en PNG ──────────────────────────
gibs_layers={
 "imerg":"IMERG_Precipitation_Rate",
 "smap":"SMAP_L3_Passive_Enhanced_Day_Soil_Moisture",
 "modis_ndvi":"MODIS_Terra_NDVI_8Day",
 "modis_lst":"MODIS_Aqua_Land_Surface_Temp_Day",
 "viirs_ndvi":"VIIRS_SNPP_NDVI_8Day",
 "grace":"GRACE_Tellus_Liquid_Water_Equivalent_Thickness_Mascon_CRI",
}
for k,lay in gibs_layers.items():
    try:
        u=(f"https://wvs.earthdata.nasa.gov/api/v1/snapshot?REQUEST=GetSnapshot&TIME=2025-07-15T00:00:00Z"
           f"&BBOX=40.62,-99.13,40.78,-98.97&CRS=EPSG:4326&LAYERS={lay},Coastlines_15m&FORMAT=image/png&WIDTH=1400&HEIGHT=1400")
        r=requests.get(u,timeout=120)
        if r.status_code==200 and 'png' in r.headers.get('Content-Type',''):
            open(RAW+f'gibs_{k}.png','wb').write(r.content); log(f"GIBS {k}: OK {len(r.content)}o")
        else: log(f"GIBS {k}: HTTP {r.status_code} {r.text[:80]}")
    except Exception as e: log(f"GIBS {k} FAIL",e)
print("DONE")
