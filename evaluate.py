import sys, os, glob, json, numpy as np, mir_eval, warnings
from multiprocessing import Pool
import chords
warnings.filterwarnings('ignore')
SONGS = sorted(glob.glob(os.path.join(os.path.dirname(__file__), 'songs', '*.wav')))
_cache = {}

def score_song(args):
    path, cfg = args
    x = chords.load(path)
    segs, key, tuning = chords.analyze(x, cfg)
    ref_i, ref_l = mir_eval.io.load_labeled_intervals(path[:-4]+'.lab')
    est_i = np.array([[s[0], s[1]] for s in segs]); est_l = [s[2] for s in segs]
    est_i, est_l = mir_eval.util.adjust_intervals(est_i, est_l, ref_i.min(), ref_i.max(), mir_eval.chord.NO_CHORD, mir_eval.chord.NO_CHORD)
    m = mir_eval.chord.evaluate(ref_i, ref_l, est_i, est_l)
    dur = ref_i.max()-ref_i.min()
    return dict(majmin=m['majmin'], sevenths=m['sevenths'], mirex=m['mirex'], root=m['root'],
                seg=m['seg'], n_est=len(segs), n_ref=len(ref_l), dur=dur, name=os.path.basename(path))

def run(cfg, songs=SONGS, procs=2):
    with Pool(procs) as p:
        res = p.map(score_song, [(s, cfg) for s in songs])
    W = np.array([r['dur'] for r in res])
    agg = {k: float(np.sum(W*np.array([r[k] for r in res]))/W.sum()) for k in ('majmin','sevenths','mirex','root','seg')}
    agg['changes_ratio'] = float(np.sum([r['n_est'] for r in res])/np.sum([r['n_ref'] for r in res]))
    return agg, res

if __name__ == '__main__':
    cfg = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    agg, res = run(cfg)
    print(json.dumps({k: round(v,4) for k,v in agg.items()}))
