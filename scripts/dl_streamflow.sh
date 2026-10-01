#!/bin/bash
# Caravan v1.6 core + GRDC-Caravan extension, 48 parallel range streams each (Zenodo is per-connection throttled)
cd $KNOU_FLOOD_ROOT/data/raw
python3 ../../scripts/pdl.py "https://zenodo.org/api/records/15349031/files/GRDC_Caravan_extension_nc.zip/content" GRDC_Caravan_extension_nc.zip 7575685313 32 32 > ../../logs/dl_grdc.log 2>&1 &
python3 ../../scripts/pdl.py "https://zenodo.org/api/records/15529786/files/Caravan-nc.tar.gz/content" Caravan-v1.6-nc.tar.gz 24775734275 48 32 > ../../logs/dl_caravan16.log 2>&1 &
wait
