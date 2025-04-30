#!/usr/bin/env python3

import json
from sys import argv
import csv
from functools import reduce

with open(argv[1], 'r') as f:
    dat = json.load(f)

trace_evs = dat.get('traceEvents')
assert trace_evs is not None


events = {}

for ev in trace_evs:
    cat = ev.get('cat')
    if not cat:
        continue
    if cat not in {"kernel", "gpu_memset", "gpu_memcpy"}:
        continue
    name = ev['name']
    args = ev.get('args')
    if not args:
        args = {}
    bdim = args.get('block')
    if not bdim:
        bdim = []
    gdim = args.get('grid')
    if not gdim:
        gdim = []
    k = (name, tuple(gdim), tuple(bdim))
    if k in events.keys():
        events[k][0] += ev['dur']
        events[k][1] += 1
    else:
        events[k] = [ev['dur'], 1, args]

with open('out.csv', 'w') as f:
    writer = csv.writer(f)
    fields = ['kernel' , 'gdim', 'bdim', 'tot_threads' ,'dur', 'blocks_per_sm', 'warps_per_sm', 'registers_per_thread', 'occupancy', 'num_calls']
    writer.writerow(fields)
    for k,v in events.items():
        kernel = k[0]
        gdim = k[1]
        bdim = k[2]
        ncalls = v[1]
        dur = v[0] / ncalls
        args = v[2]
        if bdim and gdim:
            f = lambda x,y: x*y
            nthreads = reduce(f, bdim) * reduce(f, gdim)
        else:
            nthreads = None
        writer.writerow([kernel, gdim, bdim, nthreads, dur, args.get('blocks per SM'), args.get('warps per SM'), args.get('registers per thread'), args.get('est. achieved occupancy %'), ncalls])
