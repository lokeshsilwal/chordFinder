import json, sys, time
from evaluate import run
variants = json.loads(open(sys.argv[1]).read())
pens = json.loads(sys.argv[2]) if len(sys.argv)>2 else [0.26,0.46,0.85]
for name, cfg in variants.items():
    best=None; t0=time.time()
    for p in pens:
        c=dict(cfg); c['penalty']=p
        agg,_=run(c)
        if best is None or agg['majmin']>best[1]['majmin']: best=(p,agg)
    p,a=best
    print(f"{name:34s} pen={p:<5} majmin={a['majmin']:.3f} root={a['root']:.3f} seg={a['seg']:.3f} chg={a['changes_ratio']:.2f}  ({time.time()-t0:.0f}s)", flush=True)
