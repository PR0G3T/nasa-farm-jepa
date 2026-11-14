import json, os
B='/home/veldrx/Documents/Hackathon/data-test/'
EXT=json.load(open(B+'extents.json'))
CDL=json.load(open(B+'maps/cdl_rotation.json'))
try:
    SS=json.load(open(B+'maps/ssurgo_top.json')); YLD=json.load(open(B+'maps/ssurgo_yields.json'))
except: SS=[];YLD={}
def km(b):
    return f"{(b[2]-b[0])*111.32*__import__('math').cos(__import__('math').radians((b[1]+b[3])/2)):.0f} × {(b[3]-b[1])*110.57:.0f} km"
def ext(key,sub=None):
    e=(EXT.get(key) or {})
    if sub: e=(e.get(sub) or {})
    b=e.get('bounds_ll')
    if not b: return "emprise non documentée"
    return f"{b[0]:.3f}…{b[2]:.3f}°E · {b[1]:.3f}…{b[3]:.3f}°N · ≈{km(b)} · rés {e.get('res','?')}"
def fig(src,meta,what,derive,verdict,vc):
    return f'''<div class="fig">
      <img src="{src}" loading="lazy" onclick="window.open(this.src)">
      <table class="meta"><tr><th>Provenance</th><td>{meta}</td></tr>
      <tr><th>Emprise</th><td class="mono">{verdict[1] if isinstance(verdict,tuple) else ""}{ext_str if False else ""}</td></tr></table>
      <p class="what">🔎 <b>Ce que c'est :</b> {what}</p>
      <p class="drv">💡 <b>Ce que j'en tire :</b> {derive}</p>
      <span class="vd" style="background:{vc}">{verdict}</span></div>'''
# helpers d'emprise formatée
def E(key,sub=None): return ext(key,sub)
F=[]
F.append(('power','4. NASA POWER',[('charts/power_daily.png',"API POWER daily · point -99.05, 40.70 · juin-août 2025 · 15 paramètres",
  "Série temporelle météo complète d'un point : températures min/max/moy, pluie quotidienne, humidité du sol estimée (surface GWETTOP, racinaire GWETROOT, profil GWETPROF), rayonnement solaire.",
  "C'est LA source la plus simple du lot : un appel JSON, aucun login, couverture mondiale. On voit l'été 2025 : T max régulières >30°C, pluies épisodiques (~10-20 mm), sol racinaire souvent <50% de saturation en août — contexte idéal pour parler de stress hydrique et du choix de cultures tolérantes.","✅ INDISPENSABLE — base météo de l'app",'#238636'),
  ('charts/power_10y.png',"API POWER monthly · même point · 2015-2025",
  "Dix ans de mensuels : T2M, pluie, humidité racinaire. La ligne blanche est la moyenne mobile 12 mois.",
  "Permet de sortir du 'une seule année' : tendances long terme, hivers secs répétés, variabilité inter-annuelle forte des pluies d'été — argument clé du brief (s'adapter au changement climatique).","✅ UTILE — contexte décennal",'#238636')]))
F.append(('gpm','2. GPM / IMERG',[('maps/gibs_imerg.png',"GIBS WMS · couche IMERG_Precipitation_Rate · 15/07/2025 · "+E('gibs_imerg'),
  "Carte du taux de précipitation mesuré par la constellation GPM sur tout le Nebraska ce jour-là.",
  "Résolution ~10 km : trop grossier pour LA parcelle mais parfait pour situer un épisode pluvieux régional. En démo, c'est un fond de carte ; en prod, POWER (même grille) donne déjà la série chiffrée — IMERG sert surtout aux extrêmes (averse 30 min).","🟡 CONTEXTUEL — belle carte, données par ailleurs",'#9e6a03')]))
F.append(('landsat','3. Landsat / MODIS / VIIRS',[
  ('maps/s2_rgb.png',"Sentinel-2 L2A · scène S2B 2025-08-14 · nuages 0.2% · "+E('s2'),
  "Vraie couleur 10 m : ville de Kearney (gris), Platter River, parcelles agricoles, pivots d'irrigation.",
  "La scène du 14/08 couvre TOUTE la zone — après avoir découvert que d'autres granules du même tile étaient des datastrips partiels. Réalité du terrain : damier de ~160 ha, riverbottom irrigué.","✅ RÉFÉRENCE VISUELLE",'#238636'),
  ('maps/s2_ndvi.png',"même scène · NDVI=(B08-B04)/(B08+B04)",
  "NDVI 10 m : ville en rouge (pas de végétation), cultures vertes, sol nu/sénescent jaune-orange.",
  "Outil de diagnostic principal : hétérogénéité intra-parcellaire visible (pivots verts vs coins secs), riverbottom contrasté. C'est sur ça qu'on repère le stress végétal et qu'on valide les couverts.","✅ CŒUR DE L'APP",'#238636'),
  ('maps/s2_ndre.png',"même scène · NDRE red-edge 20 m",
  "Indice red-edge, plus sensible à la chlorophylle que le NDVI : détecte le stress AVANT qu'il soit visible en NDVI.",
  "Utile pour un signal d'alerte précoce (stress azote/eau). Bonus crédibilité scientifique — coûte rien, même scène.","✅ UTILE",'#238636'),
  ('maps/s2_ndmi.png',"même scène · NDMI=(B08-B11)/(B08+B11) 20 m",
  "Contenu en eau du couvert végétal (bande SWIR).",
  "Complément direct du stress hydrique : couplé à l'ET ECOSTRESS ça raconte qui transpire vraiment.","✅ UTILE",'#238636'),
  ('maps/s2zoom_rgb.png',"même scène · "+E('s2zoom'),
  "Zoom ~7 km sur la parcelle de démo : on distingue les limites de champs, chemins, pivots.",
  "Échelle de travail réelle du fermier — 10 m suffit pour suivre un champ.","✅ ÉCHELLE PARCELLE",'#238636'),
  ('maps/s2zoom_ndvi.png',"même scène · NDVI zoom parcelle",
  "NDVI sur la même emprise réduite : contraste entre parcelles irriguées rondes et coins anguleux secs.",
  "Montre ce que le scoring pourrait ingérer : NDVI moyen par champ, détection de zones faibles.","✅ ÉCHELLE PARCELLE",'#238636'),
  ('maps/landsat_ndvi.png',"Landsat 9 L2 · LC09 2025-07-05 · "+E('landsat'),
  "NDVI 30 m Landsat, même zone : signal identique à S2 mais plus grossier.",
  "Confirmé : mêmes patrons. Son vrai atout = l'archive 50 ans (rotations historiques avant CDL) + la thermique.","🟡 REDONDANT avec S2 ici — atout = archive+LST",'#9e6a03'),
  ('maps/landsat_lst.png',"même scène · bande thermique lwir11 · "+E('landsat'),
  "Température de surface le 5/07 : les zones sèches/nues chauffent (orange-blanc), l'irrigué et la rivière restent frais.",
  "Très parlant : carte de stress thermique des parcelles — directement lié au 'heat stress' du brief. Alternative si ECOSTRESS indisponible.","✅ UTILE",'#238636'),
  ('maps/modis_ndvi.png',"MODIS 13Q1 · tuile h10v04 · composite 16 j · "+E('modis'),
  "NDVI 250 m sur une région ~560×165 km : le gradient est-ouest des grandes plaines.",
  "À 250 m on perd le champ mais on gagne la SÉRIE : 8/16 jours, 25 ans d'archive → phénologie et comparaison d'années. Contexte régional, pas parcelle.","🟡 CONTEXTUEL (phénologie, historique)",'#9e6a03')]))
F.append(('smap','1. SMAP',[
  ('maps/gibs_smap_rz.png',"GIBS WMS · SMAP_L4_Analyzed_Root_Zone_Soil_Moisture · 15/07/2025 · "+E('gibs_smap_rz'),
  "Humidité du sol RACINAIRE (0-100 cm) assimilée, grille ~9 km : le Nebraska en bleu (humide) vs Kansas au sud plus sec.",
  "Le produit le plus pertinent du brief pour l'eau — racinaire = ce que boit la culture. MAIS 9 km = contexte régional, pas parcelle. En prod : valeurs via earthaccess+login ; ici les tuiles GIBS suffisent pour illustrer.","✅ UTILE en contexte / ⚠️ jamais champ-level",'#9e6a03'),
  ('maps/gibs_smap_surf.png',"GIBS WMS · SMAP_L4 surface · 15/07/2025 · "+E('gibs_smap_surf'),
  "Humidité des 5 cm de surface le même jour.",
  "La surface sèche plus vite que le racinaire — comparer les deux montre la structure verticale de la sécheresse. Même limite : 9 km.","🟡 CONTEXTUEL",'#9e6a03'),
  ('maps/gibs_smap_gpp.png',"GIBS WMS · SMAP_L4_Mean_GPP · 15/07/2025 · "+E('gibs_smap_gpp'),
  "Production primaire brute (carbone fixé) estimée par le modèle SMAP L4.",
  "Proxy productivité végétale régional ; la famille carbone L4 (GPP/respiration/NEE) est un vrai sous-dataset distinct comme demandé — utile en argument 'santé des sols/biomasse' mais pas prioritaire.","🟡 BONUS SCIENCE",'#9e6a03')]))
F.append(('era5','24. ERA5',[
  ('charts/era5_profile.png',"ERA5 Zarr cloud · point -99.05, 40.70 · juillet 2025 · couches sol 1-4",
  "Profil d'humidité du sol quotidien : 0-7 cm, 7-28, 28-100, 100-289 cm.",
  "Données BRUTES réelles (~0.20-0.24 m³/m³) : on voit la surface sécher après juillet pendant que les couches profondes restent stables — la réserve sur laquelle la culture compte. Accès libre, aucun login — meilleur substitut à NLDAS/GLDAS.","✅ TRÈS UTILE — profil sol",'#238636'),
  ('maps/era5_sm_l1.png',"ERA5 · maille régionale 0.25° · 15/07/2025 · "+E('era5'),
  "Carte de l'humidité 0-7 cm : mailles de ~28 km, le point est une maille.",
  "Illustre honnêtement la résolution : 1 maille ERA5 ≈ toute la zone de démo. À utiliser en série temporelle de contexte, jamais en carte de champ.","🟡 CONTEXTUEL — honnêteté de résolution",'#9e6a03')]))
F.append(('ecostress','6. ECOSTRESS',[
  ('maps/ecostress_ECO_L3T_JET.png',"Granule ECO_L3T_JET 2025-05-27 · image browse NASA · "+E('ecostress','ECO_L3T_JET'),
  "Évapotranspiration journalière (PT-JPL) d'une bande orbitale ISS — le granule contient notre point.",
  "Produit phare du brief (stress hydrique réel à ~70 m). MAIS : passage ISS irrégulier, nuages, login Earthdata pour les valeurs. Les images browse (publiques) suffisent pour une démo crédible.","✅ FORT EN DÉMO / ⚠️ dispo sporadique",'#9e6a03'),
  ('maps/ecostress_ECO_L4T_ESI.png',"Granule ECO_L4T_ESI · même date · "+E('ecostress','ECO_L4T_ESI'),
  "Indice de stress évaporatif : fraction d'ET réelle / ET potentielle — 1 = pas de stress.",
  "Encore plus direct que l'ET brute : un score de stress lisible par un agronome. Mêmes limites de dispo.","✅ FORT EN DÉMO",'#9e6a03'),
  ('maps/ecostress_ECO_L2T_LSTE.png',"Granule ECO_L2T_LSTE 2025-05-20 · "+E('ecostress','ECO_L2T_LSTE'),
  "Température de surface à ~70 m — encore plus fin que Landsat thermique.",
  "Confirmé dispo en browse ; en backup Landsat LST couvre le besoin de façon régulière.","🟡 REDONDANT avec Landsat LST",'#9e6a03')]))
F.append(('nwis','17. USGS NWIS',[('charts/nwis_hydro.png',"API NWIS iv · site 06770200 Platte River near Kearney · juin-août 2025",
  "Débit quotidien réel de la Platte : ~500-1500 cfs, moyenne 874 cfs (~25 m³/s).",
  "La Platte borde la zone au sud — c'est l'eau de surface qui alimente les canaux d'irrigation locaux. Lien direct eau↔irrigation du brief ; API REST gratuite sans clé.","✅ UTILE — volet eau de surface",'#238636')]))
F.append(('cdl','9. CDL / CropScape',[
  ('maps/cdl_map_2020.png',"WMS CropScape · cdl_2020 · "+E('cdl_2020'),"Carte nationale classée 30 m : maïs jaune, soja vert foncé, alfalfa rose, eau bleue, ville grise.","Chaque année reclassée → on reconstruit l'HISTORIQUE de rotation champ par champ (voir tableau). Jaune dominant = maïs ~45% de la zone.","✅ CŒUR ROTATION",'#238636'),
  ('maps/cdl_map_2021.png',"cdl_2021 · même emprise","Idem 2021.","Comparaison pixel-par-pixel : les champs alternent jaune↔vert — la rotation maïs/soja est lisible directement.","✅",'#238636'),
  ('maps/cdl_map_2022.png',"cdl_2022 · même emprise","Idem 2022.","Confirme la stabilité du système maïs-soja + prairies permanentes.","✅",'#238636'),
  ('maps/cdl_map_2023.png',"cdl_2023 · même emprise","Idem 2023.","La ville de Kearney et la Platte servent de repères visuels.","✅",'#238636'),
  ('maps/cdl_map_2024.png',"cdl_2024 · même emprise","Idem 2024 (plus récente dispo à l'écriture).","Extension à 2025 possible : la couche cdl_2025 existe déjà.","✅",'#238636')]))
F.append(('dem','5. SRTM / DEM',[
  ('maps/dem.png',"Copernicus DEM GLO-30 · "+E('dem'),
  "Altitude 640-702 m : la vallée de la Platte (bas, au sud) et les plateaux de loess.",
  "Plat mais pas plat : le riverbottom est en creux — utile pour l'irrigation par gravité et le risque de gel/nocturne.","✅ UTILE",'#238636'),
  ('maps/slope.png',"dérivé du même DEM",
  "Pente en degrés : la plupart des champs <2°, les berges de rivière et breaks plus raides.",
  "Proxy érosion/ruissellement immédiat pour le scoring 'soil health' — là où ça penche, couverture végétale obligatoire.","✅ UTILE — entrée du scoring",'#238636')]))
F.append(('naip','12. NAIP',[
  ('maps/naip.png',"NAIP Planetary Computer · tuile 2022-07-18 · "+E('naip'),
  "Photo aérienne 1 m : pivot d'irrigation, bâtiments de ferme, routes, texture des champs.",
  "Résolution maximale du lot : on voit les traces de machinisme, les coins non irrigués, l'état du couvert. Sert de 'photo de référence' pour l'agriculteur qui reconnaît SA parcelle.","✅ FORT impact utilisateur",'#238636'),
  ('maps/naip_ndvi.png',"même tuile · canal CIR",
  "NDVI calculé à 1 m depuis l'infra-rouge NAIP : chaque plante presque.",
  "Démo bluffante de précision ; limite : couverture ~1 an sur 3, pas de série temporelle dense.","🟡 IMPRESSIONNANT mais sporadique",'#9e6a03')]))
F.append(('grace','25. GRACE (bonus)',[
  ('maps/gibs_grace.png',"GIBS · GRACE mascon CRI · dernier mois dispo · "+E('gibs_grace'),
  "Anomalie d'eau totale (sol+nappes) sur mailles de ~300 km.",
  "L'histoire Ogallala : la nappe qui irrigue la région s'épuise depuis 20 ans — c'est LE narratif eau long-terme du brief. Totalement hors échelle parcelle : narratif/contexte uniquement.","🟡 NARRATIF RÉGIONAL",'#9e6a03')]))
F.append(('landfire','21. LANDFIRE',[
  ('maps/landfire_evt.png',"WMS GeoServer USGS · LF2024_EVT_CONUS · "+E('landfire'),
  "Types de végétation existante 30 m : cultures, prairies, riverbottom boisé.",
  "Chevauche CDL pour l'agri mais apporte le non-cultivé (zones naturelles, ripisylves). Priorité basse pour la rotation — gardez-le pour les habitats/bords de champ.","🟡 SECONDAIRE",'#9e6a03')]))
F.append(('modisg','(divers) MODIS/VIIRS via GIBS',[
  ('maps/gibs_modis_ndvi.png',"GIBS · MODIS_Terra_NDVI_8Day · 11/07/2025 · "+E('gibs_modis_ndvi'),"NDVI régional 8 jours.","Fond de carte instantané sans traitement — pratique, remplacé par MODIS natif pour l'analyse.","🟡 VUE RAPIDE",'#9e6a03'),
  ('maps/gibs_modis_lst.png',"GIBS · MODIS LST 8j jour · 11/07/2025 · "+E('gibs_modis_lst'),"Température de surface régionale 8 jours.","Montre l'anomalie de chaleur régionale en un clin d'œil.","🟡 VUE RAPIDE",'#9e6a03'),
  ('maps/gibs_viirs_ndvi.png',"GIBS · VIIRS_SNPP_NDVI_8Day · 11/07/2025 · "+E('gibs_viirs_ndvi'),"NDVI VIIRS — le successeur de MODIS.","Même rôle que MODIS, prolonge la série au-delà de la fin de vie des satellites Terra/Aqua.","🟡 VUE RAPIDE",'#9e6a03')]))

# tables directes
ext_rows="".join(f"<tr><td>{k}</td><td class='mono'>{e.get('bounds_ll')}</td><td>{e.get('res','')}</td><td>{e.get('date',e.get('scene',''))}</td></tr>"
  for k,e in EXT.items() if isinstance(e,dict) and 'bounds_ll' in e)
cdl_rows="".join(f"<tr><td>{y}</td>"+"".join(f"<td>{v[0]} <span class='pct'>({v[1]}%)</span></td>" for v in CDL['stats'][y][:5])+"</tr>" for y in ['2020','2021','2022','2023','2024'])
rot=" → ".join(f"<b>{y}</b>&nbsp;{CDL['center_pixel'][y]}" for y in ['2020','2021','2022','2023','2024'])
ss_rows="".join(f"<tr><td>{r[0]}</td><td>{r[1]}</td><td>{r[2]}%</td><td>{r[3]}</td><td>{r[4]}</td><td>{r[5]}</td><td>{r[6]}</td><td>{r[7]}</td></tr>" for r in SS)
y_rows="".join(f"<tr><td>{k}</td><td>{v[0] or '—'}</td><td>{v[1] or '—'}</td></tr>" for k,v in YLD.items())

sections=""
for anchor,titre,figs in F:
    body="".join(fig(src,meta,w,d,v,c) for src,meta,w,d,v,c in figs)
    sections+=f'<h2 id="{anchor}">{titre}</h2><div class="grid">{body}</div>'

html=f'''<!DOCTYPE html><html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Field Shift — Données réelles · Nebraska 2025</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:system-ui,-apple-system,sans-serif;background:#0d1117;color:#e6edf3;padding:22px;max-width:1500px;margin:auto}}
h1{{font-size:1.45rem}} .sub{{color:#8b949e;font-size:.9rem;margin:4px 0 16px}}
h2{{font-size:1.12rem;margin:30px 0 10px;color:#58a6ff;border-bottom:1px solid #21262d;padding-bottom:6px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:14px}}
.fig{{background:#161b22;border:1px solid #21262d;border-radius:10px;padding:10px}}
.fig img{{width:100%;border-radius:6px;cursor:zoom-in;background:#fff}}
.meta{{width:100%;font-size:.72rem;border-collapse:collapse;margin:6px 0}}
.meta th{{background:#21262d;color:#8b949e;text-align:left;padding:4px 7px;width:76px;font-weight:600;border:1px solid #30363d}}
.meta td{{padding:4px 7px;border:1px solid #30363d;color:#c9d1d9}}
.mono{{font-family:ui-monospace,monospace;font-size:.68rem}}
.what,.drv{{font-size:.8rem;color:#9da7b3;line-height:1.5;margin:6px 0}}
.drv{{color:#c9d1d9}}
.vd{{display:inline-block;font-size:.66rem;font-weight:700;color:#fff;border-radius:4px;padding:3px 9px;margin-top:4px}}
table.big{{border-collapse:collapse;font-size:.78rem;width:100%;margin:10px 0}}
table.big th,table.big td{{border:1px solid #21262d;padding:5px 9px;text-align:left}}
table.big th{{background:#161b22;color:#58a6ff}} table.big tr:nth-child(even){{background:#11161d}}
.pct{{color:#8b949e;font-size:.7rem}}
.rot{{background:#161b22;border:1px solid #2ea04355;border-left:4px solid #2ea043;border-radius:8px;padding:12px;margin:10px 0;font-size:.9rem}}
a{{color:#58a6ff}}
.hi{{background:#161b22;border-radius:10px;padding:14px;border:1px solid #21262d}}
.hi img{{width:100%;border-radius:8px;cursor:zoom-in}}
</style></head><body>
<h1>🛰️ Field Shift — Données réelles croisées sur une même parcelle</h1>
<p class="sub">Point : <b>-99.05°E, 40.70°N</b> (Kearney, Nebraska · Ogallala) · Saison été 2025 · 100% données réelles récupérées le jour même · chaque carte porte ses axes lon/lat, son échelle et sa flèche nord.</p>
<h2>🗺️ Toutes les emprises sur un même repère</h2>
<div class="hi"><img src="charts/footprints.png" onclick="window.open(this.src)">
<p class="drv" style="margin-top:8px">Réponse à "sont-ils tous de la même zone ?" : <b>non</b>. Les rasters Sentinel-2/Landsat/DEM sont calés sur la zone ~14×18 km, NAIP et S2-zoom sur la parcelle ~6×5 km, MODIS/ERA5 couvrent la région (~560 km et ~165 km), les tuiles GIBS tout le Nebraska (~1000 km), et les granules ECOSTRESS des bandes orbitales ~1.3°. La colonne « Emprise » de chaque fiche donne le détail exact.</p></div>
<h2>📐 Tableau des emprises (insertion directe)</h2>
<table class="big"><tr><th>Jeu de données</th><th>Emprise [Ouest, Sud, Est, Nord]</th><th>Résolution</th><th>Date / scène</th></tr>{ext_rows}</table>
{sections}
<h2>🌽 Rotation réelle décodée (CDL × palette officielle)</h2>
<div class="rot">Pixel central (-99.05, 40.675) : {rot} — <b>la rotation maïs↔soja classique du Nebraska, puis mise en prairie</b>. Le décodage corrige un bug uint8 qui faisait passer le soja pour du millet.</div>
<table class="big"><tr><th>Année</th><th>1ʳᵉ classe</th><th>2ᵉ</th><th>3ᵉ</th><th>4ᵉ</th><th>5ᵉ</th></tr>{cdl_rows}</table>
<h2>🪨 Sols SSURGO (SDA — même base que Web Soil Survey)</h2>
<table class="big"><tr><th>Symbole</th><th>Série de sol</th><th>Part</th><th>Pente %</th><th>Taxonomie</th><th>Drainage</th><th>MO %</th><th>AWC</th></tr>{ss_rows}</table>
<h3 style="color:#58a6ff">Rendements potentiels NRCS (bu/ac — *tonnes)</h3>
<table class="big"><tr><th>Culture</th><th>Non-irrigué</th><th>Irrigué</th></tr>{y_rows}</table>
<p class="drv">Le contraste non-irrigué/irrigué (maïs 51.5 → 123.6 bu/ac) quantifie exactement ce que vaut l'eau ici — argument central pour les rotations économes en eau.</p>
<p class="sub" style="margin-top:24px">Tout est local : <code>data-test/</code> (raw = données brutes, maps = cartes géoréférencées, charts = graphiques). <a href="../ressources-field-shift.html">→ Analyse des 24 ressources</a></p>
</body></html>'''
open(B+'index.html','w').write(html)
print("index.html v2:",len(html),"octets")
