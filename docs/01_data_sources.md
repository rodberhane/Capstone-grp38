# Data sources: what to download and where to put it

Every dataset the pipeline uses, with a direct source link and the licence. Put the files under `./data/` inside the repo, keeping the exact subfolder layout below (that is where `config.DATA_DIR` points; `data/` is gitignored so it is not on GitHub). Files marked "registration" need a free account; everything else is a direct or open download.

## Folder layout under `data/`

```
data/
  irish_buildings/d05_seai_ber/building_energy_ratings.csv
  irish_buildings/d06_cso_census/saps_2022/SAPS_2022_Small_Area_UR_171024.csv
  irish_buildings/d06_cso_census/saps_2022_boundaries/Small_Area_..._2022_Ungeneralised_...gpkg
  europe/curves/d19_jrc_huizinga_curves/copy_of_global_flood_depth-damage_functions__30102017.xlsx
  europe/curves/d20_middlesex_irish_curves/middlesex_irish_residential.csv
  irish_hazard/d24_copernicus_dem/dem_dublin_itm.tif
  irish_hazard/d03_coastal_hazard/ncfhm_itm_dep_c_c_0100_f_00.tif
  irish_hazard/d02_nifm_fluvial/ing_07_dep_f_c_d_0100_f_01.tif  (and ing_08, ing_09)
  irish_context/d08_met_eireann/IE_RR_8110_V2.txt
```


| Key | Dataset | What it gives | Direct source | Licence |
|---|---|---|---|---|
| BER | SEAI BER (dwelling energy ratings) | dwelling type, floor area, storeys, year, wall type for ~1.4M Irish dwellings (exposure features) | SEAI BER Research Tool: https://ndber.seai.ie/BERResearchTool/Register/Register.aspx (registration) | SEAI research terms |
| SAPS | CSO Census 2022 Small Area Population Statistics | dwelling counts by coarse type per small area | https://www.cso.ie/en/census/census2022/census2022smallareapopulationstatistics/ | CC-BY 4.0 |
| SAPS boundaries | Small Area boundaries 2022 (GPKG) | small-area geometry for the deployment map | Tailte Éireann / data.gov.ie: https://data.gov.ie/dataset/small-areas-ungeneralised-osi-national-statistical-boundaries-2022 | CC-BY 4.0 |
| Fluvial | OPW NIFM river flood depth (D02) | fluvial flood depth labels | OPW open portal: https://www.floodinfo.ie/open-spatial-data-portal/ | CC-BY-NC-ND 4.0 |
| Coastal | OPW national coastal flood depth (D03) | coastal flood depth labels | OPW open portal: https://www.floodinfo.ie/open-spatial-data-portal/ | CC-BY-NC-ND 4.0 |
| DEM | Copernicus GLO-30 DEM (D24) | elevation and slope (hazard features) | AWS open data: https://copernicus-dem-30m.s3.amazonaws.com/ (index at https://registry.opendata.aws/copernicus-dem/) | free/open |
| Rainfall | Met Éireann 1981-2010 gridded annual rainfall (D08) | national rainfall feature | https://www.met.ie/cms/assets/uploads/2024/07/IE_RR_8110_V2.zip | CC-BY 4.0 |
| JRC curves | JRC Huizinga (2017) global depth-damage functions | reference curves (Europe, N.America, Oceania) for synthetic labels | https://publications.jrc.ec.europa.eu/repository/handle/JRC105688 | EC reuse |
| Middlesex | Middlesex FHRC / Multi-Coloured Manual residential curves | Irish-anchor per-type reference curve | FHRC Multi-Coloured Manual (not free online; shared via sponsor) | proprietary |
| Watercourses | OpenStreetMap waterways | distance-to-river feature (fluvial hazard) | pulled live in code via the Overpass API (no download) | ODbL |

## Notes for the tester

- **Minimum to run the vulnerability model:** BER, JRC curves, and Middlesex (these feed harmonisation and the V-model). The hazard and deployment scripts additionally need the OPW rasters, DEM, and SAPS.
- **CFRAM licence (CC-BY-NC-ND):** fine for academic testing with attribution; do not redistribute modified copies.
- **Reprojection:** the DEM is used reprojected to Irish Transverse Mercator (EPSG:2157); the OPW fluvial rasters are TM75 Irish Grid (EPSG:29903). The code handles the reprojection at sampling time.
- **Two datasets we deliberately do NOT use** (documented in `docs/03_methodology.md`): pluvial flood extents (not in the open portal) and SCSI rebuild costs (not openly machine-readable). Euro is reported as a band instead.
