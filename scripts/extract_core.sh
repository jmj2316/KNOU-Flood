#!/bin/bash
cd $KNOU_FLOOD_ROOT/data/caravan/core
if which pigz >/dev/null 2>&1; then pigz -dc ../../raw/Caravan-v1.6-nc.tar.gz | tar x --wildcards 'Caravan-nc/timeseries/*' 'Caravan-nc/attributes/*' 'Caravan-nc/licenses/*'
else tar xzf ../../raw/Caravan-v1.6-nc.tar.gz --wildcards 'Caravan-nc/timeseries/*' 'Caravan-nc/attributes/*' 'Caravan-nc/licenses/*'; fi
echo DONE_EXTRACT $(date)
