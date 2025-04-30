#!/usr/bin/env python3

import json
from sys import argv
import csv
from functools import reduce

with open(argv[1], 'r') as f:
    dat = json.load(f)

trace_evs = dat.get('traceEvents')
assert trace_evs is not None


events = []

for ev in trace_evs:
    cat = ev.get('cat')
    if not cat:
        continue
    if cat not in {'kernel', 'gpu_memset', 'gpu_memcpy'}:
        continue
    name = ev['name']
    args = ev.get('args')
    if not args:
        args = {}
    bdim = args.get('block')
    gdim = args.get('grid')
    ts = ev.get("ts")
    k = (name, gdim, bdim, ts)
    v = (ev['dur'], args)
    events.append((k, v))

with open('out.csv', 'w') as f:
    writer = csv.writer(f)
    fields = ['kernel' , 'gdim', 'bdim', 'tot_threads', 'dur', 'blocks_per_sm', 'warps_per_sm', 'registers_per_thread', 'occupancy', 'time_stamp']
    writer.writerow(fields)
    for k,v in events:
        kernel = k[0]
        gdim = k[1]
        bdim = k[2]
        ts = k[3]
        dur = v[0]
        args = v[1]
        nthreads = None
        if bdim and gdim:
            f = lambda x,y:x*y
            nthreads = reduce(f, bdim) * reduce(f, gdim)
        writer.writerow([kernel, gdim, bdim, nthreads, dur, args.get('blocks per SM'), args.get('warps per SM'), args.get('registers per thread'), args.get('est. achieved occupancy %'), ts])
