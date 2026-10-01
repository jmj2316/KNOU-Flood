#!/bin/bash
# Caravan v1.4 netCDF (streamflow + attributes). Resumable.
cd $KNOU_FLOOD_ROOT/data/raw
curl -L -C - --retry 5 --retry-delay 10 -o Caravan-nc.tar.xz "https://zenodo.org/api/records/10968468/files/Caravan-nc.tar.xz/content"
echo DONE_CARAVAN_NC $(date)
