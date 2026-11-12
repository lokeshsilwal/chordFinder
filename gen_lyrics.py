"""Synthetic songs with sung lyric lines and ground-truth line timings.

Structure: intro (instrumental) -> verse lines -> optional instrumental break
(sometimes with an instrument solo) -> chorus lines -> outro.
Each lyric line is a vocal phrase: one note per syllable (some syllables held
over two notes), consonant noise at syllable onsets, pitch glides, vibrato.
Lines are usually followed by a breath, but ~20% run straight into the next line.
"""
import numpy as np, os, sys, json, wave
import gen
from gen import SR, QIV, NOTE, hz, add, additive, env_adsr

SYL = ['la','na','ti','mi','ro','sa','ka','re','jo','ma','ni','da','pa','ho','ye','bi','tu','ra','go','me']

def voice(f0_track, n):
    """Vowel-like voice following an f0 track (per-sample array)."""
    t = np.arange(n)/SR
    vib = 1 + 0.008*np.sin(2*np.pi*5.5*t + gen.rng.uniform(0,6.28)) * np.clip(t/0.25, 0, 1)
    f = f0_track*vib
    ph = 2*np.pi*np.cumsum(f)/SR
    formants = gen.rng.choice([[700,1200],[400,2000],[500,900],[300,2300],[600,1700]])
    out = np.zeros(n)
    for k in range(1, 25):
        fk = f*k
        if np.mean(fk) > 5000: break
        a = sum(np.exp(-((fk-F)/150)**2) for F in formants) + 0.15/k   # formant shaping
        out += a*np.sin(k*ph)
    return out

def render(seed, outdir):
    gen.rng = np.random.default_rng(seed); rng = gen.rng
    key = int(rng.integers(12)); minor = rng.random() < .35
    bpm = float(rng.uniform(70, 130)); beat = 60/bpm; bar = 4*beat
    detune = float(rng.uniform(-0.3, 0.3))
    prog, scale = gen.make_progression(key, minor)
    verse = prog[:4]*2; chorus = prog[4:]*2
    intro_bars = int(rng.integers(2, 5))
    has_break = rng.random() < .6
    solo = has_break and rng.random() < .6
    plan = [('intro', prog[:intro_bars])] + [('verse', verse)]
    if has_break: plan.append(('break', prog[:4]))
    plan += [('chorus', chorus), ('outro', prog[:2])]
    lead_in = float(rng.uniform(0.3, 1.0))
    total_bars = sum(len(b) for _, b in plan)
    total = lead_in + total_bars*bar + 2.0
    n = int(total*SR)
    harm = np.zeros(n); bassb = np.zeros(n); vox = np.zeros(n); inst = np.zeros(n); drums = np.zeros(n)
    inst_name = rng.choice(['piano','guitar','pad','organ']); instf = {'piano':gen.piano,'guitar':gen.guitar,'pad':gen.pad,'organ':gen.organ}[inst_name]
    pattern = rng.choice(['block','strum8','sustain','offbeat']); has_drums = rng.random() < .85

    # --- accompaniment (same style as gen.py) ---
    t = lead_in; sections = []
    for name, bars in plan:
        s0 = t
        for barchords in bars:
            for (root, q, frac) in barchords:
                dur = bar*frac; iv = QIV[q]
                base = 48+root; base -= 12 if base > 57 else 0
                notes = [base+i for i in iv]
                hits = {'sustain':[(0,dur)], 'block':[(b*beat,beat*.95) for b in range(int(round(dur/beat)))],
                        'offbeat':[(b*beat+beat/2,beat*.45) for b in range(int(round(dur/beat)))],
                        'strum8':[(e*beat/2,beat/2*.98) for e in range(int(round(dur/beat*2)))]}[pattern]
                for (o, d) in hits:
                    for j, m in enumerate(notes):
                        add(harm, rng.uniform(.6,1)*instf(hz(m,detune), int((d+.25)*SR))/len(notes)**.5, t+o+(j*.012 if pattern=='strum8' else 0))
                broot = 36+root; broot -= 12 if broot > 45 else 0
                for b in range(int(round(dur/beat))):
                    add(bassb, gen.bass_tone(hz(broot + (7 if b==2 and rng.random()<.4 else 0), detune), int(beat*.95*SR)), t+b*beat)
                t += dur
        sections.append((name, s0, t))
    if has_drums:
        tt = lead_in
        while tt < t - 1e-6:
            for b in range(4):
                bt = tt+b*beat
                if b in (0,2): add(drums, gen.kick(int(.4*SR)), bt)
                if b in (1,3): add(drums, gen.snare(int(.25*SR)), bt)
                add(drums, gen.hat(int(.08*SR)), bt); add(drums, gen.hat(int(.08*SR))*.6, bt+beat/2)
            tt += bar

    # --- sung lines ---
    lines = []
    for name, s0, s1 in sections:
        if name not in ('verse', 'chorus'): continue
        nlines = 4 if (s1-s0) > 5*bar else 2
        span = (s1-s0)/nlines                       # each line gets 2 bars (or so)
        for li in range(nlines):
            ls = s0 + li*span + rng.uniform(0, beat)   # start somewhere in first beat
            run_on = rng.random() < .2 and li < nlines-1
            le_max = s0 + (li+1)*span - (0.05 if run_on else rng.uniform(0.3, 1.2))
            nsyl = int(rng.integers(5, 13))
            # syllable durations from rhythm values, scaled to fill [ls, le_max] roughly
            durs = rng.choice([0.5, 0.5, 1, 1, 1.5, 2], size=nsyl)*beat
            scale_f = (le_max-ls)/durs.sum()
            durs *= min(scale_f, 1.3) if scale_f > 0.6 else scale_f
            words, w = [], []
            for k in range(nsyl):
                w.append(rng.choice(SYL))
                if rng.random() < .45 or k == nsyl-1: words.append(''.join(w)); w = []
            tt = ls; prev_m = None
            pcs = [(key+s) % 12 for s in scale]
            for k in range(nsyl):
                d = durs[k]
                pc = int(rng.choice(pcs)); m = 60+pc; m += -12 if m > 74 else 0
                # f0 track with glide from previous note
                ns = int(d*SR*.97); f_to = hz(m, detune)
                f_from = hz(prev_m, detune) if prev_m is not None else f_to
                g = min(ns, int(.06*SR))
                track = np.full(ns, f_to); track[:g] = np.linspace(f_from, f_to, g)
                v = voice(track, ns)*env_adsr(ns, .03, .05, .9, .05)
                cons = rng.standard_normal(int(.03*SR)); cons = np.diff(np.concatenate([[0],cons]))*np.linspace(1,0,len(cons))*0.6
                add(vox, v, tt); add(vox, cons, tt)
                prev_m = m; tt += d
            lines.append(dict(text=' '.join(words), syl=nsyl, start=round(ls,3), end=round(tt,3), section=name))
    # instrument solo during the break (should NOT be matched to lyrics)
    if solo:
        name, s0, s1 = [s for s in sections if s[0]=='break'][0]
        tt = s0 + beat
        while tt < s1 - beat:
            d = beat*rng.choice([.5, 1]); m = 72 + int(rng.choice(scale))
            add(inst, gen.lead(hz(m, detune), int(d*.95*SR), rng.choice(['saw','flute'])), tt); tt += d

    norm = lambda x: x/(np.sqrt(np.mean(x**2))+1e-9) if np.any(x) else x
    mix = (rng.uniform(.6,1)*norm(harm) + rng.uniform(.4,.9)*norm(bassb) + rng.uniform(.8,1.5)*norm(vox)
           + (rng.uniform(.6,1.2)*norm(inst) if solo else 0) + (rng.uniform(.4,1.1)*norm(drums) if has_drums else 0))
    from scipy.signal import fftconvolve, butter, sosfilt
    rt = rng.uniform(.2, 1.0); irn = int(rt*SR); ir = rng.standard_normal(irn)*np.exp(-np.arange(irn)/SR*6.9/rt); ir[0]=1; ir/=np.sqrt(np.sum(ir**2))
    mix = mix + rng.uniform(.2,.5)*fftconvolve(mix, ir)[:n]
    phone = rng.random() < .35
    if phone:
        mix = sosfilt(butter(2,[rng.uniform(150,350), rng.uniform(4000,7000)], btype='band', fs=SR, output='sos'), mix)
        mix = np.tanh(mix/np.max(np.abs(mix))*1.5); mix += rng.uniform(.01,.05)*rng.standard_normal(n)*np.std(mix)*3
    mix /= np.max(np.abs(mix))*1.05
    name = f"lyr{seed:03d}"
    w = wave.open(os.path.join(outdir, name+'.wav'),'wb'); w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((mix*32767).astype(np.int16).tobytes()); w.close()
    meta = dict(bpm=round(bpm,1), inst=str(inst_name), drums=bool(has_drums), phone=bool(phone), solo=bool(solo), brk=bool(has_break))
    json.dump(dict(lines=lines, meta=meta), open(os.path.join(outdir, name+'.json'),'w'), indent=1)
    return name

if __name__ == '__main__':
    out = sys.argv[1]; a, b = int(sys.argv[2]), int(sys.argv[3]); os.makedirs(out, exist_ok=True)
    for s in range(a, b): render(s, out)
    print('ok')
