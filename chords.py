"""Python port of the Chord Finder pipeline, with switchable improvements.

cfg keys (all default to the app's current behaviour):
  hpss        : remove percussive energy before peak picking (median filtering)
  whiten      : subtract smoothed log-spectrum (spectral whitening) before peaks
  templates   : 'binary' | 'harmonic'
  nnls        : approximate-transcription chroma (remove overtones with NNLS)
  cost        : 'cosine' | 'kl' | 'gauss'
  beat        : beat-synchronous decoding
  key_bias    : bonus for chords diatonic to the detected key
  smooth      : 'mean' | 'median' ; smooth_r frames radius
  bass_w      : weight of bass-root bonus
  penalty     : switch penalty (Viterbi)
"""
import numpy as np, wave
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.optimize import nnls

SR, N, HOP = 22050, 16384, 4096
QUALS = {'maj':([0,4,7],[1,1,1],0),'min':([0,3,7],[1,1,1],0),'7':([0,4,7,10],[1,1,1,.8],.035),
         'maj7':([0,4,7,11],[1,1,1,.8],.045),'min7':([0,3,7,10],[1,1,1,.8],.035),
         'sus4':([0,5,7],[1,1,1],.06),'sus2':([0,2,7],[1,1,1],.07),'dim':([0,3,6],[1,1,1],.07)}
MODES = {'simple':['maj','min'], 'detailed':['maj','min','7','maj7','min7','sus4','sus2','dim']}
NOTE = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
DEFAULT = dict(hpss=False, whiten=False, templates='binary', nnls=False, cost='cosine', beat=False,
               key_bias=0.0, smooth='mean', smooth_r=2, bass_w=0.12, penalty=0.46, mode='simple', harm_s=0.6)

def load(path):
    w = wave.open(path); x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float64)/32768
    assert w.getframerate() == SR
    return x

def spectra(x, cfg):
    if len(x) < N: x = np.pad(x, (0, N-len(x)))
    T = (len(x)-N)//HOP + 1
    win = 0.5-0.5*np.cos(2*np.pi*np.arange(N)/(N-1))
    idx = np.arange(N)[None,:] + HOP*np.arange(T)[:,None]
    fr = x[idx]
    rms = np.sqrt(np.mean(fr**2, axis=1))
    kmax = int(np.ceil(2200*N/SR))+2
    mag = np.abs(np.fft.rfft(fr*win, axis=1))[:, :kmax+1]
    return mag, rms, T

def hpss_mask(x, kmax):
    """Harmonic/percussive separation on a short STFT; returns a per-bin, per-long-frame harmonic ratio."""
    n, h = 2048, 512
    if len(x) < n: x = np.pad(x, (0, n-len(x)))
    T = (len(x)-n)//h + 1
    win = np.hanning(n)
    idx = np.arange(n)[None,:] + h*np.arange(T)[:,None]
    S = np.abs(np.fft.rfft(x[idx]*win, axis=1))
    H = median_filter(S, size=(17,1)); P = median_filter(S, size=(1,17))
    return S, H, P, h, n

def peaks_from_mag(mag, cfg):
    T = mag.shape[0]
    kmin = int(np.floor(38*N/SR)); kmax = int(np.ceil(2200*N/SR))
    out = []
    for t in range(T):
        m = mag[t]
        if cfg['whiten']:
            lm = np.log(m+1e-9)
            # moving average over ~ 1/3 octave equivalent: use variable window approximated by bins
            bg = uniform_filter1d(lm, size=41)
            m = np.exp(np.maximum(lm-bg, 0))-1
        mx = m[kmin-1:kmax+2].max()
        thr = mx*0.012
        k = np.arange(kmin, kmax+1)
        b = m[k]; a = m[k-1]; c = m[k+1]
        sel = (b > thr) & (b >= a) & (b > c)
        k = k[sel]; a = np.log(m[k-1]+1e-12); bb = np.log(m[k]+1e-12); cc = np.log(m[k+1]+1e-12)
        den = a-2*bb+cc; p = np.where(den != 0, 0.5*(a-cc)/np.where(den==0,1,den), 0)
        f = (k+p)*SR/N
        out.append((69+12*np.log2(f/440), m[k]))
    return out

def chroma_from_peaks(peaks, T):
    sc = ss = 0
    for midi, mg in peaks:
        s = midi >= 50; w = np.sqrt(mg[s]); ang = 2*np.pi*midi[s]
        sc += np.sum(w*np.cos(ang)); ss += np.sum(w*np.sin(ang))
    tuning = np.arctan2(ss, sc)/(2*np.pi)
    ch = np.zeros((T,12)); bs = np.zeros((T,12))
    for t, (midi, mg) in enumerate(peaks):
        m = midi - tuning; r = np.round(m); d = np.abs(m-r)
        w = np.cos(d*np.pi)**2*np.sqrt(mg)
        pc = (r.astype(int) % 12); f = 440*2**((m-69)/12)
        s1 = (f >= 80) & (f <= 2100); s2 = (f >= 38) & (f <= 260)
        np.add.at(ch[t], pc[s1], w[s1]); np.add.at(bs[t], pc[s2], w[s2])
    return ch, bs, tuning

def nnls_chroma(peaks, T, tuning, s=0.6):
    """Approximate transcription: semitone spectrum -> note activations via NNLS -> chroma."""
    lo, hi = 28, 98   # midi range of the semitone spectrum
    nb = hi-lo
    notes = np.arange(28, 88)
    D = np.zeros((nb, len(notes)))
    for j, n0 in enumerate(notes):
        for k in range(1, 9):
            mk = n0 + 12*np.log2(k)
            r = int(round(mk)) - lo
            if 0 <= r < nb: D[r, j] += s**(k-1)
    ch = np.zeros((T,12)); bs = np.zeros((T,12))
    for t, (midi, mg) in enumerate(peaks):
        m = midi - tuning; r = np.round(m).astype(int); d = np.abs(m-r)
        w = np.cos(d*np.pi)**2*np.sqrt(mg)
        v = np.zeros(nb); ok = (r >= lo) & (r < hi)
        np.add.at(v, r[ok]-lo, w[ok])
        if v.sum() == 0: continue
        a, _ = nnls(D, v, maxiter=200)
        for j, n0 in enumerate(notes):
            if a[j] <= 0: continue
            if n0 >= 43: ch[t, n0 % 12] += a[j]
            if n0 <= 60: bs[t, n0 % 12] += a[j]*(1.0 if n0 <= 52 else 0.5)
    return ch, bs

def beats(x):
    """Spectral-flux onset envelope + autocorrelation tempo + DP beat tracker (Ellis 2007)."""
    n, h = 2048, 512
    if len(x) < n: x = np.pad(x, (0, n-len(x)))
    T = (len(x)-n)//h + 1
    idx = np.arange(n)[None,:] + h*np.arange(T)[:,None]
    S = np.log1p(100*np.abs(np.fft.rfft(x[idx]*np.hanning(n), axis=1)))
    flux = np.maximum(np.diff(S, axis=0), 0).sum(axis=1)
    flux = np.concatenate([[0], flux])
    flux = flux - uniform_filter1d(flux, 16); flux = np.maximum(flux, 0)
    flux /= (flux.std()+1e-9)
    fps = SR/h
    # tempo via autocorrelation weighted around 120 bpm
    ac = np.correlate(flux, flux, 'full')[len(flux)-1:]
    lags = np.arange(len(ac)); bpm = 60*fps/np.maximum(lags, 1)
    wgt = np.exp(-0.5*(np.log2(bpm/120)/0.9)**2)
    rng_ = (bpm >= 60) & (bpm <= 180)
    lag = lags[rng_][np.argmax((ac*wgt)[rng_])]
    period = lag
    # DP
    score = flux.copy(); back = -np.ones(len(flux), int)
    tight = 100
    for i in range(len(flux)):
        lo, hi = i-int(2*period), i-int(period/2)
        if hi <= 0: continue
        lo = max(lo, 0); js = np.arange(lo, hi)
        pen = -tight*(np.log((i-js)/period))**2
        c = score[js] + pen; k = np.argmax(c)
        score[i] = flux[i] + c[k]; back[i] = js[k]
    i = int(np.argmax(score[-int(period*2):]) + len(score)-int(period*2))
    bt = []
    while i >= 0: bt.append(i); i = back[i]
    bt = np.array(bt[::-1])
    return (bt*h + n/2)/SR, 60*fps/period

KMAJ=np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
KMIN=np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
def detect_key(total):
    best=(-2,0,False)
    for r in range(12):
        x = np.roll(total, -r)
        for prof, mn in ((KMAJ,False),(KMIN,True)):
            c = np.corrcoef(x, prof)[0,1]
            if c > best[0]: best=(c,r,mn)
    return best[1], best[2]

def build_chords(mode, templates, s):
    out = []
    for q in MODES[mode]:
        iv, w, pen = QUALS[q]
        for r in range(12):
            v = np.zeros(12)
            for i, ww in zip(iv, w):
                if templates == 'harmonic':
                    for k in range(1, 7):
                        v[(r+i+int(round(12*np.log2(k)))) % 12] += ww*s**(k-1)
                else:
                    v[(r+i) % 12] = ww
            out.append((r, q, v/np.linalg.norm(v), pen))
    return out

def analyze(x, cfg_in=None):
    cfg = dict(DEFAULT); cfg.update(cfg_in or {})
    dur = len(x)/SR
    mag, rms, T = spectra(x, cfg)
    if cfg['hpss']:
        S, H, P, h, n = hpss_mask(x, mag.shape[1])
        # ratio per short-STFT bin, map to long frames/bins by averaging
        ratio = (H**2/(H**2+P**2+1e-12))
        fS = np.arange(S.shape[1])*SR/n
        for t in range(T):
            c = (t*HOP + N/2 - n/2)/h
            lo = int(max(0, c - (N/h)/2)); hi = int(min(S.shape[0], c + (N/h)/2 + 1))
            rr = ratio[lo:hi].mean(axis=0)
            fL = np.arange(mag.shape[1])*SR/N
            mag[t] *= np.interp(fL, fS, rr)
    peaks = peaks_from_mag(mag, cfg)
    ch, bs, tuning = chroma_from_peaks(peaks, T)
    if cfg['nnls']:
        ch, bs = nnls_chroma(peaks, T, tuning, cfg['harm_s'])
    # time axis of frames
    tc = (np.arange(T)*HOP + N/2)/SR
    if cfg['beat']:
        bt, tempo = beats(x)
        # use beat boundaries; pool frames between boundaries
        edges = np.concatenate([[0], (bt[:-1]+bt[1:])/2 if len(bt)>1 else [], [dur]])
        edges = np.concatenate([[0], bt, [dur]]); edges = np.unique(np.clip(edges, 0, dur))
        segs_ix = np.searchsorted(edges, tc, side='right')-1
        B = len(edges)-1
        def pool(a):
            o = np.zeros((B, a.shape[1]))
            for b in range(B):
                sel = segs_ix == b
                if sel.any(): o[b] = np.median(a[sel], axis=0) if cfg['smooth']=='median' else a[sel].mean(axis=0)
                else:  # interpolate nearest frame
                    o[b] = a[np.argmin(np.abs(tc-(edges[b]+edges[b+1])/2))]
            return o
        chS, bsS = pool(ch), pool(bs)
        rmsB = pool(rms[:,None])[:,0]
        starts = edges[:-1]; ends = edges[1:]
    else:
        R = cfg['smooth_r']
        if cfg['smooth'] == 'median': chS = median_filter(ch, size=(2*R+1,1), mode='nearest'); bsS = median_filter(bs, size=(2*R+1,1), mode='nearest')
        else: chS = uniform_filter1d(ch, 2*R+1, axis=0, mode='nearest'); bsS = uniform_filter1d(bs, 2*R+1, axis=0, mode='nearest')
        rmsB = rms
        starts = np.maximum(0, tc - HOP/SR/2*0); starts = np.concatenate([[0], tc[1:]])  # JS: segment starts at frame centre
        ends = np.concatenate([tc[1:], [dur]])
    U = chS.shape[0]
    total = (chS/np.maximum(np.linalg.norm(chS,axis=1,keepdims=True),1e-9)).sum(axis=0)
    keyRoot, keyMinor = detect_key(total)
    ref = np.sort(rmsB)[int(len(rmsB)*0.75)] if len(rmsB) else 0
    low = (rmsB < ref*0.08) | (rmsB < 2e-4)
    chords = build_chords(cfg['mode'], cfg['templates'], cfg['harm_s'])
    K = len(chords)+1
    E = np.zeros((U, K))
    cn = chS/np.maximum(np.linalg.norm(chS,axis=1,keepdims=True),1e-9)
    bn = bsS/np.maximum(bsS.max(axis=1,keepdims=True),1e-9)
    scale = np.array([0,2,3,5,7,8,10] if keyMinor else [0,2,4,5,7,9,11])
    inkey = set(((keyRoot+scale) % 12).tolist())
    for j, (r, q, tmpl, pen) in enumerate(chords):
        if cfg['cost'] == 'cosine':
            s = cn @ tmpl
        elif cfg['cost'] == 'kl':
            p = chS/np.maximum(chS.sum(axis=1,keepdims=True),1e-9)
            qv = tmpl/tmpl.sum(); qv = 0.85*qv + 0.15/12
            s = -np.sum(p*np.log((p+1e-9)/qv), axis=1)   # -KL(p||q)
            s = s*cfg.get('kl_scale', 1.0)
        elif cfg['cost'] == 'gauss':
            s = -np.sum((cn - tmpl)**2, axis=1)
        s = s + cfg['bass_w']*bn[:, r] - pen
        if cfg['key_bias']:
            tones = set(((r+np.array(QUALS[q][0])) % 12).tolist())
            if tones <= inkey: s = s + cfg['key_bias']
        if cfg.get('key_prior'):
            # graded soft prior: diatonic > one outside note (V in minor, secondary dominants, borrowed chords) > rest
            tones = set(((r+np.array(QUALS[q][0])) % 12).tolist())
            out = len(tones - inkey)
            s = s - cfg['key_prior']*min(out, 2)/2
        E[:, j] = np.where(low, s-2, s)
    nc_base = cfg.get('nc_score', 0.3)
    E[:, -1] = np.where(low, 3, nc_base if cfg['cost']=='cosine' else E[:, :-1].max(axis=1)-1.0)
    # Viterbi with constant switch penalty
    P = cfg['penalty']
    if cfg['beat']: P = cfg.get('beat_penalty', P*1.5)
    # per-unit weight: beats are longer than frames, scale emissions by duration in frames for comparability
    if cfg['beat']:
        wdur = np.maximum((ends-starts)/(HOP/SR), 0.5)
        E = E*wdur[:,None]
    prev = E[0].copy(); back = np.zeros((U, K), int)
    for t in range(1, U):
        bi = int(np.argmax(prev)); sw = prev[bi]-P
        stay = prev >= sw
        back[t] = np.where(stay, np.arange(K), bi)
        prev = np.where(stay, prev, sw) + E[t]
    path = np.zeros(U, int); path[-1] = int(np.argmax(prev))
    for t in range(U-1, 0, -1): path[t-1] = back[t, path[t]]
    segs = []
    for t in range(U):
        j = path[t]; lab = 'N' if j == K-1 else f"{NOTE[chords[j][0]]}:{chords[j][1]}"
        if segs and segs[-1][2] == lab: segs[-1][1] = ends[t]; continue
        if segs: segs[-1][1] = starts[t]
        segs.append([0.0 if not segs else starts[t], ends[t], lab])
    segs[-1][1] = dur
    # remove blips < 0.4 s
    changed = True
    while changed and len(segs) > 1:
        changed = False
        for i, s in enumerate(segs):
            if s[1]-s[0] < 0.4:
                if i > 0: segs[i-1][1] = s[1]
                else: segs[1][0] = s[0]
                segs.pop(i); changed = True; break
        i = 1
        while i < len(segs):
            if segs[i][2] == segs[i-1][2]: segs[i-1][1] = segs[i][1]; segs.pop(i); changed = True
            else: i += 1
    return segs, (keyRoot, keyMinor), tuning
