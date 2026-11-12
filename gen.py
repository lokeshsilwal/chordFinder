"""Synthetic song generator with ground-truth chord labels.

Each song: random key (major/minor), tempo, an 8-bar progression played twice,
with a chord instrument (piano / strummed guitar / pad / organ), a bass line
(sometimes inversions and walking notes), a loud melody that uses non-chord
tones, drums, reverb, global detuning, noise, and sometimes a "phone mic"
band-limited degradation.  Labels are written in mir_eval .lab format.
"""
import numpy as np, wave, os, sys, json

SR = 22050
rng = None
NOTE = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
QIV = {'maj':[0,4,7],'min':[0,3,7],'7':[0,4,7,10],'maj7':[0,4,7,11],'min7':[0,3,7,10],
       'sus4':[0,5,7],'dim':[0,3,6]}

def hz(m, detune): return 440*2**((m-69+detune)/12)

def env_adsr(n, a, d, s, r, sr=SR):
    t = np.arange(n)/sr
    e = np.ones(n)*s
    ai = min(n, int(a*sr)); e[:ai] = np.linspace(0,1,ai,endpoint=False) if ai else e[:ai]
    di = min(n-ai, int(d*sr))
    if di>0: e[ai:ai+di] = 1-(1-s)*np.linspace(0,1,di)
    ri = min(n, int(r*sr))
    if ri>0: e[-ri:] *= np.linspace(1,0,ri)
    return e

def additive(f, n, amps, inharm=0.0, decay_k=None, vib=0.0, phase_rand=True):
    t = np.arange(n)/SR
    out = np.zeros(n)
    vibr = 1 + vib*np.sin(2*np.pi*5.2*t + rng.uniform(0,6.28)) if vib else 1
    for k, a in enumerate(amps, start=1):
        fk = f*k*np.sqrt(1+inharm*k*k)
        if fk > SR/2-500: break
        ph = rng.uniform(0,2*np.pi) if phase_rand else 0
        if np.isscalar(vibr):
            s = np.sin(2*np.pi*fk*t + ph)
        else:
            s = np.sin(2*np.pi*np.cumsum(fk*vibr)/SR + ph)
        if decay_k is not None: s *= np.exp(-t*decay_k*k)
        out += a*s
    return out

def piano(f, n):
    amps = [1/(k**1.1) for k in range(1,14)]
    x = additive(f, n, amps, inharm=0.0004, decay_k=0.6)
    return x*env_adsr(n, .005, .3, .5, .08)*np.exp(-np.arange(n)/SR*1.0)

def guitar(f, n):
    amps = [ (1/k)*(0.8+0.4*np.sin(k*1.7)) for k in range(1,16)]
    x = additive(f, n, amps, inharm=0.0001, decay_k=1.1)
    return x*env_adsr(n, .002, .05, .9, .05)*np.exp(-np.arange(n)/SR*1.5)

def pad(f, n):
    amps = [ (1/k)**1.6 for k in range(1,12)]       # filtered saw
    return additive(f, n, amps, vib=0.003)*env_adsr(n, .25, .2, .9, .3)

def organ(f, n):
    amps = [1, .8, .6, .5, .2, .3, 0, .25]           # drawbar-ish
    return additive(f, n, amps)*env_adsr(n, .01, .05, .95, .05)

def bass_tone(f, n):
    amps = [1, .6, .35, .25, .15, .1, .06]
    return additive(f, n, amps, decay_k=0.8)*env_adsr(n, .005, .15, .7, .05)

def lead(f, n, kind):
    if kind == 'vox':   # vowel-like: formant-shaped harmonics, vibrato
        amps = [1, .9, 1.0, .6, .5, .7, .4, .25, .2, .15, .1]
        return additive(f, n, amps, vib=0.006)*env_adsr(n, .04, .1, .85, .06)
    if kind == 'saw':
        return additive(f, n, [1/k for k in range(1,18)], vib=0.004)*env_adsr(n,.01,.1,.8,.05)
    return additive(f, n, [1, 0, 1/3, 0, 1/5, 0, 1/7, 0, 1/9], vib=0.003)*env_adsr(n,.01,.1,.8,.05)  # square-ish flute

def kick(n):
    t = np.arange(n)/SR
    f = 50 + 90*np.exp(-t*30)
    return np.sin(2*np.pi*np.cumsum(f)/SR)*np.exp(-t*9)

def snare(n):
    t = np.arange(n)/SR
    noise = rng.standard_normal(n)
    return (0.8*noise*np.exp(-t*18) + 0.5*np.sin(2*np.pi*185*t)*np.exp(-t*25))

def hat(n):
    t = np.arange(n)/SR
    x = rng.standard_normal(n); x = np.diff(np.concatenate([[0], x]))  # crude highpass
    return x*np.exp(-t*60)*0.5

def add(buf, x, start):
    s = int(start*SR)
    if s >= len(buf): return
    e = min(len(buf), s+len(x)); buf[s:e] += x[:e-s]

def make_progression(key, minor):
    scale = [0,2,3,5,7,8,10] if minor else [0,2,4,5,7,9,11]
    if minor:
        diat = {1:'min',2:'dim',3:'maj',4:'min',5:'min',6:'maj',7:'maj'}
        common = [[1,6,3,7],[1,4,5,1],[1,7,6,7],[1,4,7,3],[6,7,1,1],[1,6,4,5]]
    else:
        diat = {1:'maj',2:'min',3:'min',4:'maj',5:'maj',6:'min',7:'dim'}
        common = [[1,5,6,4],[1,4,5,4],[6,4,1,5],[1,6,4,5],[2,5,1,1],[1,3,4,5],[4,1,5,6]]
    bars = []
    for _ in range(2):
        bars += list(common[rng.integers(len(common))])
    chords = []
    for deg in bars:
        root = (key + scale[deg-1]) % 12
        q = diat[deg]
        if q == 'dim' and rng.random() < .7:      # avoid too many dims
            deg = 5; root = (key+scale[4]) % 12; q = diat[5]
        r = rng.random()
        if minor and deg == 5 and r < .5: q = '7' if r < .3 else 'maj'   # harmonic-minor V
        elif deg == 5 and q == 'maj' and r < .3: q = '7'
        elif q == 'maj' and deg in (1,4) and r < .12: q = 'maj7'
        elif q == 'min' and r < .15: q = 'min7'
        elif q == 'maj' and r < .05: q = 'sus4'
        # sometimes split the bar into two chords
        if rng.random() < .2:
            d2 = rng.integers(1,8); r2 = (key+scale[d2-1]) % 12; q2 = diat[d2]
            if q2 == 'dim': q2 = 'min' if not minor else 'maj';
            chords.append([(root,q,.5),(r2,q2,.5)])
        else:
            chords.append([(root,q,1.0)])
    return chords, scale

def label(root, q):
    return f"{NOTE[root]}:{q}"

def render(seed, outdir):
    global rng
    rng = np.random.default_rng(seed)
    key = int(rng.integers(12)); minor = rng.random() < .35
    bpm = float(rng.uniform(72, 140)); beat = 60/bpm; bar = 4*beat
    detune = float(rng.uniform(-0.4, 0.4))   # semitones
    prog, scale = make_progression(key, minor)
    reps = 2
    sections = prog*reps
    lead_in = float(rng.uniform(0.5, 2.0))
    total = lead_in + len(sections)*bar + 2.0
    n = int(total*SR)
    harm = np.zeros(n); bassb = np.zeros(n); mel = np.zeros(n); drums = np.zeros(n)

    inst_name = rng.choice(['piano','guitar','pad','organ'])
    inst = {'piano':piano,'guitar':guitar,'pad':pad,'organ':organ}[inst_name]
    lead_kind = rng.choice(['vox','saw','flute'])
    pattern = rng.choice(['block','strum8','sustain','offbeat'])
    walk = rng.random() < .4
    inversions = rng.random() < .5
    has_drums = rng.random() < .85

    labels = []
    t = lead_in
    labels.append((0.0, lead_in, 'N'))
    for barchords in sections:
        for (root, q, frac) in barchords:
            dur = bar*frac
            labels.append((t, t+dur, label(root, q)))
            iv = QIV[q]
            # voicing
            base = 48 + root
            if base > 57: base -= 12
            notes = [base+i for i in iv]
            if inversions and rng.random() < .5:
                k = int(rng.integers(1, len(notes)))
                notes = notes[k:] + [x+12 for x in notes[:k]]
            if rng.random() < .5: notes.append(notes[0]+12)
            # rhythm
            if pattern == 'sustain':
                hits = [(0, dur)]
            elif pattern == 'block':
                hits = [(b*beat, beat*0.95) for b in range(int(round(dur/beat)))]
            elif pattern == 'offbeat':
                hits = [(b*beat+beat/2, beat*0.45) for b in range(int(round(dur/beat)))]
            else:
                hits = [(e*beat/2, beat/2*0.98) for e in range(int(round(dur/beat*2)))]
            for (o, d) in hits:
                for j, m in enumerate(notes):
                    st = t + o + (j*0.012 if pattern=='strum8' else 0)
                    nn = int((d+0.25)*SR)
                    vel = rng.uniform(.6,1.0)
                    harm[int(st*SR):int(st*SR)+nn] += 0  # placeholder for clarity
                    add(harm, vel*inst(hz(m,detune), nn)/len(notes)**0.5, st)
            # bass
            broot = 36 + root
            if broot > 45: broot -= 12
            bnotes = []
            nbeats = int(round(dur/beat))
            for b in range(nbeats):
                m = broot
                if inversions and b == 0 and rng.random() < .15: m = broot + iv[1]   # slash chord
                elif walk and b == nbeats-1 and rng.random() < .6: m = broot + rng.choice([2, -1, 5, -2])
                elif b == 2 and rng.random() < .4: m = broot + 7
                bnotes.append((b*beat, m))
            for (o, m) in bnotes:
                add(bassb, bass_tone(hz(m,detune), int(beat*0.95*SR)), t+o)
            t += dur

    # melody: phrase over each bar, eighths/quarters, chord + scale tones
    t = lead_in
    ml = []
    for (s0, s1, lab) in labels[1:]:
        root = NOTE.index(lab.split(':')[0]); q = lab.split(':')[1]
        ctones = [(root+i)%12 for i in QIV[q]]
        tt = s0
        while tt < s1 - 1e-6:
            d = beat*rng.choice([0.5, 1, 1, 1.5, 2])
            d = min(d, s1-tt)
            if rng.random() < .15: tt += d; continue   # rest
            if rng.random() < .55: pc = int(rng.choice(ctones))
            else: pc = (key + int(rng.choice(scale))) % 12
            m = 60 + pc + (12 if rng.random() < .5 else 0)
            if m > 81: m -= 12
            add(mel, lead(hz(m,detune), int(d*0.97*SR), lead_kind), tt)
            tt += d
    # drums
    if has_drums:
        tt = lead_in
        while tt < lead_in + len(sections)*bar - 1e-6:
            for b in range(4):
                bt = tt + b*beat
                if b in (0, 2) or rng.random() < .15: add(drums, kick(int(.4*SR)), bt)
                if b in (1, 3): add(drums, snare(int(.25*SR)), bt)
                add(drums, hat(int(.08*SR)), bt); add(drums, hat(int(.08*SR))*.6, bt+beat/2)
            tt += bar

    norm = lambda x: x/(np.sqrt(np.mean(x**2))+1e-9)
    g_h = rng.uniform(.6, 1.0); g_b = rng.uniform(.4, .9); g_m = rng.uniform(.6, 1.4); g_d = rng.uniform(.4, 1.2) if has_drums else 0
    mix = g_h*norm(harm) + g_b*norm(bassb) + g_m*norm(mel) + (g_d*norm(drums) if has_drums else 0)
    # reverb
    rt = rng.uniform(.2, 1.2)
    ir_n = int(rt*SR); ir = rng.standard_normal(ir_n)*np.exp(-np.arange(ir_n)/SR*6.9/rt)
    ir[0] = 1; ir /= np.sqrt(np.sum(ir**2))
    from scipy.signal import fftconvolve, butter, sosfilt
    wet = fftconvolve(mix, ir)[:n]
    mix = mix + rng.uniform(.2,.6)*wet
    phone = rng.random() < .35
    if phone:   # phone-mic recording of a speaker in a room
        sos = butter(2, [rng.uniform(150,350), rng.uniform(4000,7000)], btype='band', fs=SR, output='sos')
        mix = sosfilt(sos, mix)
        mix = np.tanh(mix/np.max(np.abs(mix))*1.5)
        mix += rng.uniform(.01,.05)*rng.standard_normal(n)*np.std(mix)*3
    mix /= np.max(np.abs(mix))*1.05
    name = f"song{seed:03d}"
    w = wave.open(os.path.join(outdir, name+'.wav'), 'wb'); w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((mix*32767).astype(np.int16).tobytes()); w.close()
    labels.append((labels[-1][1], total, "N"))
    with open(os.path.join(outdir, name+'.lab'), 'w') as f:
        for s, e, l in labels: f.write(f"{s:.4f}\t{e:.4f}\t{l}\n")
    meta = dict(key=NOTE[key]+(' minor' if minor else ' major'), bpm=round(bpm,1), inst=str(inst_name), lead=str(lead_kind),
                pattern=str(pattern), drums=bool(has_drums), phone=bool(phone), detune=round(detune,2), walk=bool(walk), inv=bool(inversions))
    return name, meta

if __name__ == '__main__':
    out = sys.argv[1]; count = int(sys.argv[2]); os.makedirs(out, exist_ok=True)
    metas = {}
    for s in range(count):
        name, meta = render(s, out); metas[name] = meta
    json.dump(metas, open(os.path.join(out,'meta.json'),'w'), indent=1)
    print('wrote', count)
