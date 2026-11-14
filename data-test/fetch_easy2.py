import requests, json, os, math
RAW='/home/veldrx/Documents/Hackathon/data-test/raw/'
log=lambda *a:print('▶',*a,flush=True)

# ── POWER daily (params corrigés) + monthly ──
try:
    p=dict(parameters="T2M,T2M_MAX,T2M_MIN,PRECTOTCORR,ALLSKY_SFC_SW_DWN,RH2M,WS2M,T2MDEW,GWETROOT,GWETTOP,GWETPROF,FROST_DAYS,EVLAND,TSOIL1,CLRSKY_SFC_SW_DWN",
           community="AG",latitude=40.70,longitude=-99.05,start="20250601",end="20250831",format="JSON")
    r=requests.get("https://power.larc.nasa.gov/api/temporal/daily/point",params=p,timeout=180); r.raise_for_status()
    open(RAW+'power_daily_2025.json','w').write(r.text)
    d=r.json()['properties']['parameter']; log("POWER daily: OK,",len(d['T2M']),"jours ×",len(d),"params")
except Exception as e: log("POWER daily FAIL",e)
try:
    p=dict(parameters="T2M,T2M_MAX,T2M_MIN,PRECTOTCORR,ALLSKY_SFC_SW_DWN,GWETROOT,WS2M",
           community="AG",latitude=40.70,longitude=-99.05,start="201501",end="202510",format="JSON")
    r=requests.get("https://power.larc.nasa.gov/api/temporal/monthly/point",params=p,timeout=180); r.raise_for_status()
    open(RAW+'power_monthly_10y.json','w').write(r.text)
    log("POWER monthly: OK,",len(r.json()['properties']['parameter']['T2M']),"mois")
except Exception as e: log("POWER monthly FAIL",e)

# ── SDA : fonction d'intersection puis propriétés ──
def sda(q):
    r=requests.post("https://sdmdataaccess.sc.egov.usda.gov/Tabular/post.rest",
                    data={"query":q,"format":"JSON"},timeout=120); r.raise_for_status(); return r.json()
WKT="POLYGON((-99.13 40.62,-98.97 40.62,-98.97 40.78,-99.13 40.78,-99.13 40.62))"
try:
    res=sda(f"SELECT * FROM SDA_Get_Mukey_from_intersection_with_WktWgs84('{WKT}')")
    log("SDA intersect:",res.get('Table',[])[:5])
    keys=[str(row[0]) for row in res['Table']]
    log("  ",len(keys),"mukeys")
    open(RAW+'ssurgo_mukeys.json','w').write(json.dumps(res))
except Exception as e: log("SDA intersect FAIL",e); keys=[]
if keys:
    kl=",".join(f"'{k}'" for k in keys)
    for name,q in [
      ("props",f"""SELECT m.mukey,m.musym,m.muname,c.cokey,c.compname,c.comppct_r,c.slope_r,c.taxorder,
       h.hzdept_r,h.hzdepb_r,h.om_r,h.awc_r,h.kwfact,h.drainagecl
       FROM mapunit m JOIN component c ON c.mukey=m.mukey LEFT JOIN chorizon h ON h.cokey=c.cokey
       WHERE m.mukey IN ({kl})"""),
      ("yields",f"""SELECT cc.mukey,c.compname,cc.cropname,cc.yldunits,cc.nonirryield_r,cc.irryield_r
       FROM cocropyld cc JOIN component c ON c.cokey=cc.cokey WHERE cc.mukey IN ({kl})""")]:
        try:
            res=sda(q); open(RAW+f'ssurgo_{name}.json','w').write(json.dumps(res))
            log(f"SDA {name}:",len(res.get('Table',[])),"lignes")
        except Exception as e: log(f"SDA {name} FAIL",e)

# ── CDL : WMS maps 2020-2024 + GetCDLFile en mètres Mercator ──
for y in [2020,2021,2022,2023,2024]:
    try:
        u=(f"https://nassgeodata.gmu.edu/CropScapeService/wms_cdlall.cgi?SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap"
           f"&LAYERS=cdl_{y}&SRS=EPSG:4326&BBOX=-99.13,40.62,-98.97,40.78&WIDTH=1200&HEIGHT=1200&FORMAT=image/png&STYLES=")
        r=requests.get(u,timeout=120)
        ct=r.headers.get('Content-Type','')
        if r.status_code==200 and 'png' in ct:
            open(RAW+f'cdl_map_{y}.png','wb').write(r.content); log(f"CDL WMS {y}: OK {len(r.content)}o")
        else: log(f"CDL WMS {y}: {r.status_code} {r.text[:80]}")
    except Exception as e: log(f"CDL WMS {y} FAIL",e)
# bbox en Web Mercator
def merc(lon,lat):
    x=6378137*math.radians(lon); y=6378137*math.log(math.tan(math.pi/4+math.radians(lat)/2)); return x,y
x0,y0=merc(-99.13,40.62); x1,y1=merc(-98.97,40.78)
try:
    r=requests.get("https://nassgeodata.gmu.edu/axis2/services/CDLService/GetCDLFile",
                   params={"year":"2024","bbox":f"{x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f}"},timeout=240)
    log("CDL file (m):",r.status_code,r.text[:130])
    if 'http' in r.text:
        url=r.text[r.text.find('http'):r.text.find('</return>')]
        t=requests.get(url.strip(),timeout=240); open(RAW+'cdl_2024.tif','wb').write(t.content)
        log("  GeoTIFF:",len(t.content),"o")
except Exception as e: log("CDL file(m) FAIL",e)

# ── GIBS corrigé ──
gibs={
 "imerg":("IMERG_Precipitation_Rate","2025-07-15T12:00:00Z"),
 "smap":("SMAP_L3_Active_Passive_Soil_Moisture","2025-07-15T00:00:00Z"),
 "grace":("GRACE_Tellus_Liquid_Water_Equivalent_Thickness_Mascon_CRI","2025-06-01T00:00:00Z"),
 "modis_lst":("MODIS_Aqua_L3_Land_Surface_Temp_Daily_Day","2025-07-15T00:00:00Z"),
 "viirs_lst":("VIIRS_NOAA20_Land_Surface_Temp_Daily_Day","2025-07-15T00:00:00Z"),
}
for k,(lay,t) in gibs.items():
    try:
        u=(f"https://wvs.earthdata.nasa.gov/api/v1/snapshot?REQUEST=GetSnapshot&TIME={t}"
           f"&BBOX=40.30,-99.50,41.10,-98.60&CRS=EPSG:4326&LAYERS={lay},Coastlines_15m&FORMAT=image/png&WIDTH=1400&HEIGHT=1400")
        r=requests.get(u,timeout=120)
        if r.status_code==200 and 'png' in r.headers.get('Content-Type',''):
            open(RAW+f'gibs_{k}.png','wb').write(r.content); log(f"GIBS {k}: OK {len(r.content)}o")
        else: log(f"GIBS {k}: HTTP {r.status_code} {r.text[:80]}")
    except Exception as e: log(f"GIBS {k} FAIL",e)

# ── PRISM point API ──
try:
    for v in ['ppt','tmean']:
        u=f"https://services.nacse.org/prism/data/public/4km/{v}/20250715?lon=-99.05&lat=40.70"
        r=requests.get(u,timeout=60); log(f"PRISM {v}:",r.status_code,r.text[:120])
        if r.status_code==200: open(RAW+f'prism_{v}.json','w').write(r.text)
except Exception as e: log("PRISM FAIL",e)

# ── CoCoRaHS : stations NE Buffalo County ──
try:
    u="https://data.cocorahs.org/cocorahs/export/exportreports.aspx?ReportType=Daily&dtf=1&Format=CSV&State=NE&County=Buffalo&StartDate=07/01/2025&EndDate=07/31/2025"
    r=requests.get(u,timeout=120); log("CoCoRaHS:",r.status_code,r.text[:120].replace('\n',' | '))
    if r.status_code==200 and len(r.text)>100: open(RAW+'cocorahs_buffalo_0725.csv','w').write(r.text)
except Exception as e: log("CoCoRaHS FAIL",e)

# ── Ag Data Commons : API recherche ──
try:
    for u in ["https://agdatacommons.nal.usda.gov/api/datasets?search=crop%20rotation&limit=5",
              "https://agdatacommons.nal.usda.gov/api/search?query=crop%20rotation"]:
        r=requests.get(u,timeout=60); log("AgDataCommons:",r.status_code,u.split('/api/')[1][:30],r.text[:100])
except Exception as e: log("AgDC FAIL",e)

# ── OpenET / QuickStats / AMS sans clé (documenter) ──
for name,u in [("OpenET","https://openet-api.org/raster/timeseries/point?lon=-99.05&lat=40.70&start_date=2025-07-01&end_date=2025-07-31&model=ensemble&variable=ET&units=mm&interval=daily"),
               ("QuickStats","https://quickstats.nass.usda.gov/api/api_GET/?key=TEST&commodity_desc=CORN&year__GE=2020&state_alpha=NE&statisticcat_desc=YIELD&format=JSON"),
               ("AMS","https://marsapi.ams.usda.gov/services/v1.2/reports?q=report_title:corn")]:
    try:
        r=requests.get(u,timeout=60); log(name,":",r.status_code,r.text[:100])
    except Exception as e: log(name,"FAIL",e)

# ── LANDFIRE WMS capabilities ──
for u in ["https://www.landfire.gov/geoserver/wms?SERVICE=WMS&REQUEST=GetCapabilities",
          "https://lfps.usgs.gov/arcgis/services/LandfireProductService/GPServer?f=json"]:
    try:
        r=requests.get(u,timeout=60); log("LANDFIRE:",r.status_code,r.url[:60],r.text[:80])
    except Exception as e: log("LANDFIRE FAIL",u[:40],e)
print("DONE")
