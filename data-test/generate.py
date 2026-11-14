import json, os, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
BASE='/home/veldrx/Documents/Hackathon/data-test/'
RAW,MAP,CHT=BASE+'raw/',BASE+'maps/',BASE+'charts/'
os.makedirs(MAP,exist_ok=True); os.makedirs(CHT,exist_ok=True)
log=lambda *a:print('▶',*a,flush=True)
plt.rcParams.update({'figure.facecolor':'#161b22','axes.facecolor':'#161b22','text.color':'#e6edf3',
 'axes.edgecolor':'#3d4858','axes.labelcolor':'#c9d1d9','xtick.color':'#8b949e','ytick.color':'#8b949e',
 'font.size':9})

def savefig(fig,name):
    fig.savefig(CHT+name,dpi=140,bbox_inches='tight'); plt.close(fig); log("chart:",name)

def savemap(arr,name,cmap='viridis',vmin=None,vmax=None,title=''):
    fig,ax=plt.subplots(figsize=(6,5)); ax.axis('off')
    ax.imshow(arr,cmap=cmap,vmin=vmin,vmax=vmax)
    if title: ax.set_title(title,color='#e6edf3',fontsize=10,pad=8)
    fig.savefig(MAP+name,dpi=140,bbox_inches='tight',pad_inches=0.05); plt.close(fig); log("map:",name)

# ═══ 1. POWER : dashboard été 2025 ═══
try:
    d=json.load(open(RAW+'power_daily_2025.json'))['properties']['parameter']
    days=sorted(d['T2M'].keys()); x=range(len(days))
    fig,axs=plt.subplots(4,1,figsize=(10,11),sharex=True)
    axs[0].fill_between(x,[d['T2M_MIN'][k] for k in days],[d['T2M_MAX'][k] for k in days],alpha=.5,color='#f0a050')
    axs[0].plot(x,[d['T2M'][k] for k in days],color='#fff',lw=.8); axs[0].set_ylabel('T (°C)'); axs[0].set_title('POWER — daily agro-météo, parcelle Nebraska, juin-août 2025')
    axs[1].bar(x,[d['PRECTOTCORR'][k] for k in days],color='#4cc2ff',width=1); axs[1].set_ylabel('Pluie (mm/j)')
    axs[2].plot(x,[d['GWETROOT'][k] for k in days],color='#56d364',label='GWETROOT (racinaire)')
    axs[2].plot(x,[d['GWETTOP'][k] for k in days],color='#c8a165',label='GWETTOP (surface)')
    axs[2].plot(x,[d['GWETPROF'][k] for k in days],color='#8b949e',ls='--',label='GWETPROF (profil)')
    axs[2].set_ylabel('Humidité sol (fraction)'); axs[2].legend(loc='lower right',fontsize=8)
    axs[3].plot(x,[d['ALLSKY_SFC_SW_DWN'][k] for k in days],color='#e3c341')
    axs[3].set_ylabel('Rayonnement (MJ/m²/j)')
    axs[3].set_xticks(list(x)[::10]); axs[3].set_xticklabels([days[i] for i in list(x)[::10]],rotation=45)
    savefig(fig,'power_daily.png')
except Exception as e: log("POWER daily chart FAIL",e)

try:
    d=json.load(open(RAW+'power_monthly_10y.json'))['properties']['parameter']
    months=sorted(d['T2M'].keys()); x=np.arange(len(months))
    fig,axs=plt.subplots(3,1,figsize=(10,8),sharex=True)
    t2m=np.array([d['T2M'][k] for k in months]); t2m[t2m<-50]=np.nan
    axs[0].plot(x,t2m,color='#f0a050',lw=1); axs[0].set_ylabel('T2M °C'); axs[0].set_title('POWER — 10 ans de mensuel (2015→2025)')
    p=np.array([d['PRECTOTCORR'][k] for k in months]); p[p<-50]=np.nan
    axs[1].plot(x,p,color='#4cc2ff',lw=.8); axs[1].plot(x,np.convolve(np.nan_to_num(p),np.ones(12)/12,'same'),color='#fff',lw=1.5)
    axs[1].set_ylabel('Pluie mm/j (moy. mobile 12m)')
    gw=np.array([d['GWETROOT'][k] for k in months]); gw[gw<-50]=np.nan
    axs[2].plot(x,gw,color='#56d364',lw=1); axs[2].set_ylabel('GWETROOT')
    axs[2].set_xticks(x[::12]); axs[2].set_xticklabels(months[::12])
    savefig(fig,'power_10y.png')
except Exception as e: log("POWER 10y FAIL",e)

# ═══ 2. Sentinel-2 : vraie couleur + NDVI + NDRE + NDMI + sol nu ═══
try:
    z=np.load(RAW+'s2_bands.npz'); meta=json.loads(str(z['meta']))
    b2,b3,b4,b8=z['B02'].astype(float),z['B03'].astype(float),z['B04'].astype(float),z['B08'].astype(float)
    scl=z['SCL']
    rgb=np.clip(np.stack([b4,b3,b2],-1)/3000,0,1)**(1/1.8)
    savemap(rgb,'s2_rgb.png',title=f"Sentinel-2 L2A vraie couleur — {meta['date'][:10]}")
    ndvi=(b8-b4)/(b8+b4+1e-6)
    savemap(np.where(b8+b4>0,ndvi,np.nan),'s2_ndvi.png','RdYlGn',0,.9,f"NDVI 10 m — {meta['date'][:10]}")
    b5=z['B05'].astype(float); b8d=np.array(Image.fromarray(b8).resize((b5.shape[1],b5.shape[0]),Image.BILINEAR))
    ndre=(b8d-b5)/(b8d+b5+1e-6)
    savemap(np.where(b8d+b5>0,ndre,np.nan),'s2_ndre.png','RdYlGn',0,.8,"NDRE red-edge 20 m (stress précoce)")
    b11=z['B11'].astype(float)
    ndmi=(b8d-b11)/(b8d+b11+1e-6)
    savemap(np.where(b8d+b11>0,ndmi,np.nan),'s2_ndmi.png','BrBG',-.4,.6,"NDMI — contenu en eau du couvert")
    bare=(ndvi<0.25)&(b8+b4>0)
    overlay=rgb.copy(); overlay[bare]=[1,.2,.2]
    savemap(overlay,'s2_bare.png',title="Masque sol nu (rouge, NDVI<0.25)")
    sclcls=np.unique(scl,return_counts=True)
    log("S2 ok — SCL:",dict(zip(*[v.tolist() for v in sclcls])))
except Exception as e: log("S2 maps FAIL",e)

# ═══ 3. Landsat : NDVI + LST ═══
try:
    z=np.load(RAW+'landsat_bands.npz'); meta=json.loads(str(z['meta']))
    sc,off=meta['scale'],meta['offset']
    red=z['red'].astype(float)*sc['red']+off['red']; nir=z['nir08'].astype(float)*sc['red']+off['red']
    ndvi=(nir-red)/(nir+red+1e-6)
    savemap(np.where((nir+red)>0,ndvi,np.nan),'landsat_ndvi.png','RdYlGn',0,.9,f"Landsat 9 NDVI — {meta['date'][:10]}")
    st=z['lwir11'].astype(float)*sc['lwir11']+off['lwir11']-273.15
    savemap(st,'landsat_lst.png','inferno',20,45,f"Landsat 9 température de surface (°C)")
except Exception as e: log("Landsat FAIL",e)

# ═══ 4. MODIS NDVI ═══
try:
    z=np.load(RAW+'modis13.npz'); nd=z['ndvi'].astype(float)*0.0001
    savemap(np.where(nd>-0.3,nd,np.nan),'modis_ndvi.png','RdYlGn',0,.9,"MODIS 13Q1 NDVI 250 m — juillet 2025")
except Exception as e: log("MODIS FAIL",e)

# ═══ 5. DEM : hillshade + pente ═══
try:
    dem=np.load(RAW+'dem.npz')['dem']
    gy,gx=np.gradient(dem,30)
    slope=np.degrees(np.arctan(np.hypot(gx,gy)))
    hs=np.clip((np.sin(np.radians(45))*dem/np.nanmax(dem)+0.5),0,1)
    savemap(dem,'dem.png','terrain',title=f"Copernicus DEM 30 m — {np.nanmin(dem):.0f}–{np.nanmax(dem):.0f} m")
    savemap(slope,'slope.png','YlOrRd',0,10,"Pente (°) — proxy érosion/ruissellement")
except Exception as e: log("DEM FAIL",e)

# ═══ 6. NAIP ═══
try:
    naip=np.load(RAW+'naip.npy')
    rgb=np.clip(naip[:3].transpose(1,2,0)/255,0,1)
    # recadrer le centre
    h,w,_=rgb.shape; c=rgb[:,w//4:3*w//4]
    savemap(c,'naip.png',title="NAIP 1 m — photo aérienne 2022")
    cir=naip[3].astype(float); nred=naip[0].astype(float)
    nndvi=(cir-nred)/(cir+nred+1e-6)
    savemap(nndvi[:,w//4:3*w//4],'naip_ndvi.png','RdYlGn',0,.8,"NDVI 1 m depuis CIR NAIP")
except Exception as e: log("NAIP FAIL",e)

# ═══ 7. ERA5 : carte + profil ═══
try:
    z=np.load(RAW+'era5_soil.npz'); reg=z['regional']; ser=json.loads(str(z['series']))
    savemap(reg,'era5_sm_l1.png','YlGnBu',.15,.30,"ERA5 humidité sol 0-7 cm (m³/m³) — 15/07/2025")
    fig,ax=plt.subplots(figsize=(9,4.5))
    days=list(range(1,32))
    cols=['#4cc2ff','#56d364','#e3c341','#c8a165']
    for i,L in enumerate(['volumetric_soil_water_layer_1','volumetric_soil_water_layer_2','volumetric_soil_water_layer_3','volumetric_soil_water_layer_4']):
        ax.plot(days,ser[L],color=cols[i],label=f"L{i+1} (0-7/7-28/28-100/100-289 cm)"[i:],lw=1.4)
    ax.set_ylabel('m³/m³'); ax.set_xlabel('juillet 2025'); ax.legend(fontsize=8)
    ax.set_title('ERA5 — profil d\'humidité du sol, parcelle Nebraska')
    savefig(fig,'era5_profile.png')
except Exception as e: log("ERA5 FAIL",e)

# ═══ 8. NWIS hydrogramme ═══
try:
    d=json.load(open(RAW+'nwis_dv_2025.json'))
    for s in d['value']['timeSeries']:
        v=s['values'][0]['value']
        xs=[x['dateTime'][:10] for x in v]; ys=[float(x['value']) for x in v]
        fig,ax=plt.subplots(figsize=(9,3.5))
        ax.fill_between(range(len(xs)),ys,color='#4cc2ff',alpha=.6)
        ax.set_ylabel('Débit (cfs)'); ax.set_title(f"NWIS — {s['sourceInfo']['siteName']}")
        ax.set_xticks(range(0,len(xs),15)); ax.set_xticklabels([xs[i] for i in range(0,len(xs),15)],rotation=40)
        savefig(fig,'nwis_hydro.png')
except Exception as e: log("NWIS FAIL",e)

# ═══ 9. CDL stats → barres empilées rotation ═══
try:
    stats=json.loads(open(RAW+'cdl_stats_2015_2024.txt'))
    rows={}
    import xml.etree.ElementTree as ET
    for y,txt in stats.items():
        acres={}
        for m in re.finditer(r'<class name="([^"]+)" acres="([\d.]+)"',txt):
            acres[m.group(1)]=float(m.group(2))
        if not acres:
            for m in re.finditer(r'<([a-zA-Z_]+)>([\d.]+)</',txt): acres[m.group(1)]=float(m.group(2))
        rows[y]=acres
    log("CDL années parsées:",{y:len(v) for y,v in rows.items()})
    # top classes cumulées
    tot={}
    for y,a in rows.items():
        for k,v in a.items(): tot[k]=tot.get(k,0)+v
    top=[k for k,_ in sorted(tot.items(),key=lambda x:-x[1])[:8]]
    years=sorted(rows.keys())
    fig,ax=plt.subplots(figsize=(10,4.5))
    bottom=np.zeros(len(years))
    palette=['#ffd300','#267000','#a57000','#70a800','#69af3c','#a8de9c','#d1b391','#a5a5a5']
    for i,k in enumerate(top):
        vals=[rows[y].get(k,0) for y in years]
        ax.bar(range(len(years)),vals,bottom=bottom,label=k,color=palette[i%len(palette)])
        bottom+=np.array(vals)
    ax.set_xticks(range(len(years))); ax.set_xticklabels(years)
    ax.set_ylabel('Acres'); ax.legend(fontsize=7,ncol=4); ax.set_title('CDL — occupation des cultures 2015-2024 (parcelle + environs)')
    savefig(fig,'cdl_rotation.png')
except Exception as e: log("CDL stats FAIL",e)

# ═══ 10. SSURGO : tableau sols + yields ═══
try:
    props=json.load(open(RAW+'ssurgo_props.json'))['Table']
    comp={}
    for r in props:
        mukey,musym,muname,cokey,compname,pct,slope,tax,drain,hd,hb,om,awc,kw=r
        if cokey not in comp: comp[cokey]={'mu':musym,'name':compname,'pct':pct or 0,'slope':slope,'tax':tax,'drain':drain,'om':[],'awc':[]}
        comp[cokey]['pct']=max(comp[cokey]['pct'],pct or 0)
        if om is not None: comp[cokey]['om'].append(om)
        if awc is not None: comp[cokey]['awc'].append(awc)
    top=sorted(comp.values(),key=lambda c:-c['pct'])[:10]
    table=[[c['mu'],(c['name'] or '')[:38],round(c['pct'],1),c['slope'],(c['tax'] or '')[:14],
            (c['drain'] or '')[:16],round(np.mean(c['om']),2) if c['om'] else '—',
            round(np.mean(c['awc']),3) if c['awc'] else '—'] for c in top]
    open(MAP+'ssurgo_top.json','w').write(json.dumps(table))
    log("SSURGO top:",len(table),"sols ; AWC moy:",round(np.nanmean([np.mean(c['awc']) for c in top if c['awc']]),3))
    ylds=json.load(open(RAW+'ssurgo_yields.json'))['Table']
    ymap={}
    for mu,compname,crop,unit,nonirr,irr in ylds:
        ymap.setdefault(crop,[]).append((nonirr,irr))
    open(MAP+'ssurgo_yields.json','w').write(json.dumps(ymap))
    log("SSURGO yields cultures:",list(ymap.keys()))
except Exception as e: log("SSURGO FAIL",e)

# copier les images déjà prêtes dans maps/
import shutil
for f in ['cdl_map_2020.png','cdl_map_2021.png','cdl_map_2022.png','cdl_map_2023.png','cdl_map_2024.png',
          'gibs_modis_ndvi.png','gibs_modis_lst.png','gibs_smap_rz.png','gibs_smap_surf.png','gibs_smap_gpp.png',
          'gibs_ft_jan.png','gibs_grace.png','gibs_imerg.png',
          'ecostress_ECO_L3T_JET.png','ecostress_ECO_L4T_ESI.png','ecostress_ECO_L2T_LSTE.png','landfire_evt.png']:
    src=RAW+f
    if os.path.exists(src):
        shutil.copy(src,MAP+f); log("copié:",f)
print("GENERATION DONE")
