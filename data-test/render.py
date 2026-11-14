import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
BASE='/home/veldrx/Documents/Hackathon/data-test/'
RAW,MAP,CHT=BASE+'raw/',BASE+'maps/',BASE+'charts/'
EXT=json.load(open(BASE+'extents.json'))
os.makedirs(MAP,exist_ok=True); os.makedirs(CHT,exist_ok=True)
log=lambda *a:print('▶',*a,flush=True)
plt.rcParams.update({'figure.facecolor':'#fff','axes.facecolor':'#fff','font.size':8,
 'axes.edgecolor':'#333','axes.labelcolor':'#222','xtick.color':'#333','ytick.color':'#333'})

def geo_axes(ax,b,title):
    ax.set_xlim(b[0],b[2]); ax.set_ylim(b[1],b[3])
    ax.set_xlabel('Longitude (°E)'); ax.set_ylabel('Latitude (°N)')
    ax.set_title(title,fontsize=9.5,fontweight='bold',pad=6)
    ax.tick_params(labelsize=7)
    # échelle : segment en bas à gauche
    w=b[2]-b[0]
    km_per_deg=111.32*np.cos(np.radians((b[1]+b[3])/2))
    target=w*km_per_deg/4
    nice=min([x for x in [0.1,0.2,0.5,1,2,5,10,20,50,100,200,500] if x>0],key=lambda x:abs(x-target))
    seg=nice/km_per_deg
    x0=b[0]+w*0.04; y0=b[1]+(b[3]-b[1])*0.05
    ax.plot([x0,x0+seg],[y0,y0],color='k',lw=3,solid_capstyle='butt',zorder=10)
    ax.text(x0+seg/2,y0+(b[3]-b[1])*0.015,f"{nice:g} km",ha='center',fontsize=7,fontweight='bold',
            bbox=dict(fc='white',ec='none',alpha=.7,pad=1))
    # flèche nord
    ax.annotate('N',xy=(b[2]-w*0.04,b[3]-(b[3]-b[1])*0.10),xytext=(b[2]-w*0.04,b[3]-(b[3]-b[1])*0.20),
        arrowprops=dict(facecolor='k',width=2,headwidth=8),ha='center',fontsize=10,fontweight='bold')

def show_arr(arr,name,b,cmap=None,vmin=None,vmax=None,title='',rgb=False):
    fig,ax=plt.subplots(figsize=(7,5.5))
    if rgb: ax.imshow(np.clip(arr,0,1),extent=[b[0],b[2],b[1],b[3]],interpolation='bilinear')
    else:
        ax.imshow(arr,extent=[b[0],b[2],b[1],b[3]],cmap=cmap,vmin=vmin,vmax=vmax,interpolation='nearest')
        if not rgb:
            sm=ax.images[-1]
            fig.colorbar(sm,ax=ax,fraction=0.035,pad=0.02)
    geo_axes(ax,b,title)
    fig.savefig(MAP+name,dpi=130,bbox_inches='tight'); plt.close(fig); log(name)

def show_png(png,name,b,title):
    im=Image.open(png).convert('RGB')
    fig,ax=plt.subplots(figsize=(7,5.5))
    ax.imshow(im,extent=[b[0],b[2],b[1],b[3]],interpolation='bilinear')
    geo_axes(ax,b,title)
    fig.savefig(MAP+name,dpi=130,bbox_inches='tight'); plt.close(fig); log(name)

def deg2km(b):
    return round((b[2]-b[0])*111.32*np.cos(np.radians((b[1]+b[3])/2)),1),round((b[3]-b[1])*110.57,1)

# ═══ vue d'ensemble des emprises ═══
fig,axs=plt.subplots(1,2,figsize=(12,5.5),gridspec_kw={'width_ratios':[1.6,1]})
footprints=[("s2","S2 zone",'#2e8b57',1),("s2zoom","S2 parcelle",'#ff6347',1.6),
            ("landsat","Landsat",'#4682b4',1),("naip","NAIP",'#9370db',1.2),
            ("dem","DEM",'#a0522d',1),("modis","MODIS",'#daa520',1),
            ("era5","ERA5",'#20b2aa',1.2)]
for k,lab,c,lw in footprints:
    b=EXT.get(k,{}).get('bounds_ll')
    if not b: continue
    axs[0].add_patch(Rectangle((b[0],b[1]),b[2]-b[0],b[3]-b[1],fill=False,edgecolor=c,lw=lw))
    axs[0].text(b[0],b[3],lab,fontsize=7,color=c,va='bottom')
axs[0].plot(-99.05,40.70,'r*',ms=12); axs[0].text(-99.04,40.70,'parcelle',fontsize=8)
geo_axes(axs[0],[-100.2,40.0,-95.6,41.6],"Emprises des cartes — vue régionale")
wide=[("gibs_smap_rz","GIBS région (SMAP/MODIS/IMERG)",'#1e90ff'),
      ("cdl_2023","CDL",'#2e8b57'),("landfire","LANDFIRE",'#8b4513'),
      ("ecostress","ECOSTRESS (granule)",'#ff4500')]
for k,lab,c in wide:
    e=EXT.get(k) or EXT.get('ecostress',{}).get('ECO_L3T_JET') if k=='ecostress' else EXT.get(k)
    b=(e or {}).get('bounds_ll')
    if not b: continue
    axs[1].add_patch(Rectangle((b[0],b[1]),b[2]-b[0],b[3]-b[1],fill=False,edgecolor=c,lw=1.2))
    axs[1].text(b[0],b[3],lab,fontsize=6.5,color=c,va='bottom')
axs[1].plot(-99.05,40.70,'r*',ms=12)
geo_axes(axs[1],[-105,36.5,-93,45],"Vue contexte (régional)")
fig.savefig(CHT+'footprints.png',dpi=140,bbox_inches='tight'); plt.close(fig); log("footprints.png")

# ═══ rasters ═══
z=np.load(RAW+'s2_bands.npz'); b=EXT['s2']['bounds_ll']; meta=json.loads(str(z['meta']))
rgb=np.clip(np.stack([z['B04'],z['B03'],z['B02']],-1)/2800,0,1)**(1/1.7)
show_arr(rgb,'s2_rgb.png',b,rgb=True,title=f"Sentinel-2 vraie couleur — {meta['date'][:10]} · zone {deg2km(b)[0]}×{deg2km(b)[1]} km")
ndvi=(z['B08'].astype(float)-z['B04'])/(z['B08']+z['B04']+1e-6)
show_arr(np.where(z['B08']+z['B04']>0,ndvi,np.nan),'s2_ndvi.png',b,'RdYlGn',0,.9,"NDVI Sentinel-2 (10 m)")
b5=z['B05'].astype(float); b8d=np.array(Image.fromarray(z['B08'].astype(float)).resize((b5.shape[1],b5.shape[0]),Image.BILINEAR))
ndre=(b8d-b5)/(b8d+b5+1e-6)
show_arr(np.where(b8d+b5>0,ndre,np.nan),'s2_ndre.png',b,'RdYlGn',0,.8,"NDRE red-edge (20 m)")
b11=z['B11'].astype(float)
ndmi=(b8d-b11)/(b8d+b11+1e-6)
show_arr(np.where(b8d+b11>0,ndmi,np.nan),'s2_ndmi.png',b,'BrBG',-.4,.6,"NDMI — humidité du couvert (20 m)")
zz=np.load(RAW+'s2zoom_bands.npz'); bz=EXT['s2zoom']['bounds_ll']
rgbz=np.clip(np.stack([zz['B04'],zz['B03'],zz['B02']],-1)/2800,0,1)**(1/1.7)
show_arr(rgbz,'s2zoom_rgb.png',bz,rgb=True,title=f"Sentinel-2 zoom parcelle — {deg2km(bz)[0]}×{deg2km(bz)[1]} km")
ndviz=(zz['B08'].astype(float)-zz['B04'])/(zz['B08']+zz['B04']+1e-6)
show_arr(np.where(zz['B08']+zz['B04']>0,ndviz,np.nan),'s2zoom_ndvi.png',bz,'RdYlGn',0,.9,"NDVI zoom parcelle")

z=np.load(RAW+'landsat_bands.npz'); b=EXT['landsat']['bounds_ll']
sc={'red':2.75e-5,'lwir11':0.00341802}; off={'red':-0.2,'lwir11':149.0}
red=z['red'].astype(float)*sc['red']+off['red']; nir=z['nir08'].astype(float)*sc['red']+off['red']
ndvi=(nir-red)/(nir+red+1e-6)
show_arr(np.where(nir+red>0,ndvi,np.nan),'landsat_ndvi.png',b,'RdYlGn',0,.9,f"Landsat 9 NDVI (30 m) — {EXT['landsat']['date']}")
st=z['lwir11'].astype(float)*sc['lwir11']+off['lwir11']-273.15
show_arr(st,'landsat_lst.png',b,'inferno',15,45,"Landsat 9 temp. de surface °C")

z=np.load(RAW+'modis13.npz'); b=EXT['modis']['bounds_ll']
nd=z['ndvi'].astype(float)*0.0001
show_arr(np.where(nd>-0.3,nd,np.nan),'modis_ndvi.png',b,'RdYlGn',0,.9,f"MODIS 13Q1 NDVI (250 m) — zone {deg2km(b)[0]}×{deg2km(b)[1]} km")

dem=np.load(RAW+'dem.npz')['dem']; b=EXT['dem']['bounds_ll']
show_arr(dem,'dem.png',b,'terrain',640,705,"Copernicus DEM 30 m (m)")
gy,gx=np.gradient(dem.astype(float),30)
slope=np.degrees(np.arctan(np.hypot(gx,gy)))
show_arr(slope,'slope.png',b,'YlOrRd',0,8,"Pente (°) — proxy érosion")

naip=np.load(RAW+'naip.npy'); b=EXT['naip']['bounds_ll']
rgb=np.clip(naip[:3].transpose(1,2,0)/255,0,1)**(1/1.3)
show_arr(rgb,'naip.png',b,rgb=True,title=f"NAIP 1 m — {EXT['naip']['date']} · {deg2km(b)[0]}×{deg2km(b)[1]} km")
cir=naip[3].astype(float); nndvi=(cir-naip[0])/(cir+naip[0]+1e-6)
show_arr(nndvi,'naip_ndvi.png',b,'RdYlGn',0,.8,"NDVI 1 m depuis CIR NAIP")

z=np.load(RAW+'era5_map.npz'); b=EXT['era5']['bounds_ll']
sm=z['sm']; lats=z['lats']; lons=z['lons']-360
sm_day=sm[36] if sm.shape[0]>36 else sm[-1]
fig,ax=plt.subplots(figsize=(7,5.5))
ax.pcolormesh(lons,lats,sm_day,cmap='YlGnBu',vmin=.15,vmax=.35,shading='auto')
fig.colorbar(ax.collections[0],ax=ax,label='m³/m³',fraction=0.035,pad=0.02)
geo_axes(ax,b,"ERA5 humidité sol 0-7 cm — 15/07/2025 12Z · mailles 0.25°")
fig.savefig(MAP+'era5_sm_l1.png',dpi=130,bbox_inches='tight'); plt.close(fig); log("era5_sm_l1.png")

# ═══ PNG WMS ═══
for y in range(2020,2025):
    show_png(RAW+f'cdl_map_{y}.png',f'cdl_map_{y}.png',EXT[f'cdl_{y}']['bounds_ll'],f"CDL {y} — classification cultures (30 m)")
for fn,t in [("gibs_smap_rz","SMAP L4 humidité racinaire — 15/07/2025"),
             ("gibs_smap_surf","SMAP L4 humidité surface — 15/07/2025"),
             ("gibs_smap_gpp","SMAP L4 GPP — 15/07/2025"),
             ("gibs_grace","GRACE — eau totale (anomalie)"),
             ("gibs_imerg","IMERG taux de précipitation — 15/07/2025"),
             ("gibs_modis_ndvi","MODIS Terra NDVI 8j — 11/07/2025"),
             ("gibs_modis_lst","MODIS Terra LST jour 8j — 11/07/2025"),
             ("gibs_viirs_ndvi","VIIRS NDVI 8j — 11/07/2025"),
             ("landfire_evt","LANDFIRE EVT — végétation existante")]:
    src=RAW+('landfire_evt.png' if fn=='landfire_evt' else fn+'.png')
    key='landfire' if fn=='landfire_evt' else fn
    if os.path.exists(src) and key in EXT:
        show_png(src,fn+'.png',EXT[key]['bounds_ll'],t+" · région ~1000×800 km")
    else: log(fn,"SKIP (absent)")

# ECOSTRESS browse (emprise = granule enregistré)
for sn,title in [("ECO_L3T_JET","ECOSTRESS ET journalière (PT-JPL)"),
                 ("ECO_L4T_ESI","ECOSTRESS indice de stress évaporatif ESI"),
                 ("ECO_L2T_LSTE","ECOSTRESS température de surface")]:
    e=EXT.get('ecostress',{}).get(sn)
    src=RAW+f'ecostress_{sn}.png'
    if e and os.path.exists(src):
        show_png(src,f'ecostress_{sn}.png',e['bounds_ll'],f"{title} — granule {e['date']} (emprise orbite)")
    else: log(sn,"skip")
print("RENDER DONE")
