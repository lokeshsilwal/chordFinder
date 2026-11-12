// ===== Lyrics alignment module (no dependencies; runs in browser and Node) =====
const Lyr = (() => {

  // ---------- Syllable counting ----------
  const EN_WORDS = new Set('the you and i me my to a in it is of on your love all we be for so that this with but no not when what can just like oh baby know now go dont im youre its was are do will one up down out'.split(' '));
  function sylDevanagari(w){
    // count aksharas: independent vowels + consonants not followed by virama (halant)
    let n = 0;
    for (let i=0;i<w.length;i++){
      const c = w.charCodeAt(i);
      if (c >= 0x0905 && c <= 0x0914) n++;                       // independent vowels
      else if ((c >= 0x0915 && c <= 0x0939) || (c >= 0x0958 && c <= 0x095F)){
        if (w.charCodeAt(i+1) !== 0x094D && !(w.charCodeAt(i+1) === 0x093C && w.charCodeAt(i+2) === 0x094D)) n++;
      }
    }
    return Math.max(1, n);
  }
  function sylLatin(w, english){
    w = w.toLowerCase().replace(/[^a-z]/g, '');
    if (!w) return 0;
    const groups = w.match(/[aeiouy]+/g);
    let n = groups ? groups.length : 1;
    if (english && n > 1){
      if (/[^aeiouy]e$/.test(w) && !/[^aeiouy]le$/.test(w)) n--;      // silent e (but "-le" is a syllable)
      if (/[^aeiouy]es$|[^aeiouy]ed$/.test(w) && !/[td]ed$|[sxz]es$|[cs]hes$/.test(w)) n--;
    }
    return Math.max(1, n);
  }
  function lineSyllables(text){
    const words = text.split(/\s+/).filter(Boolean);
    const english = words.filter(w => EN_WORDS.has(w.toLowerCase().replace(/[^a-z]/g,''))).length >= Math.max(1, words.length*0.2);
    return words.map(w => /[ऀ-ॿ]/.test(w) ? sylDevanagari(w) : sylLatin(w, english));
  }

  // ---------- Parse pasted lyrics ----------
  // Returns items: {type:'line', text, syl:[per word], words:[...]} | {type:'header', text} | {type:'blank'}
  function parse(text){
    const out = [];
    for (let raw of text.replace(/\r/g,'').split('\n')){
      const t = raw.trim();
      if (!t){ if (out.length && out[out.length-1].type !== 'blank') out.push({type:'blank'}); continue; }
      if (/^\[.*\]$/.test(t) || /^(verse|chorus|bridge|intro|outro|pre-chorus|hook|refrain|antara|asthayi|sthayi)\b.*:?$/i.test(t) && t.split(/\s+/).length <= 3){
        out.push({type:'header', text:t.replace(/^\[|\]$/g,'').replace(/:$/,'')}); continue;
      }
      // ignore chord-only lines someone pasted along with the lyrics
      if (/^([A-G][#b]?(m|maj|min|dim|aug|sus|add)?\d*(\/[A-G][#b]?)?\s*)+$/.test(t)) continue;
      const words = t.split(/\s+/);
      const syl = lineSyllables(t);
      out.push({type:'line', text:t, words, syl, nsyl: syl.reduce((a,b)=>a+b,0)});
    }
    while (out.length && out[out.length-1].type === 'blank') out.pop();
    return out;
  }

  // ---------- Vocal phrase detection ----------
  // Singing is usually the most prominent sound in the 300–3500 Hz band. We track that band's
  // loudness and compare it with a local threshold (between the quiet and loud levels of the
  // surrounding ±12 s), so it adapts to each part of the song. Then we clean up into phrases.
  function detectPhrases(x, SR, makeFFT, opts={}){
    const n = 2048, h = 256, fps = SR/h;
    const T = Math.max(1, Math.floor((x.length - n)/h) + 1);
    const fft = makeFFT(n), re = new Float64Array(n), im = new Float64Array(n);
    const win = new Float64Array(n); for (let i=0;i<n;i++) win[i] = 0.5-0.5*Math.cos(2*Math.PI*i/(n-1));
    const k0 = Math.ceil(300*n/SR), k1 = Math.floor(3500*n/SR);
    const band = new Float64Array(T);
    for (let t=0;t<T;t++){
      const o = t*h;
      for (let i=0;i<n;i++){ re[i] = (x[o+i]||0)*win[i]; im[i] = 0; }
      fft(re, im);
      let e = 0; for (let k=k0;k<=k1;k++) e += re[k]*re[k] + im[k]*im[k];
      band[t] = Math.log(e + 1e-9);
    }
    // smooth ~0.12 s
    const R = Math.max(1, Math.round(0.06*fps));
    const fs = new Float64Array(T);
    { let acc = 0, c = 0; const pre = new Float64Array(T+1); for (let t=0;t<T;t++) pre[t+1] = pre[t]+band[t];
      for (let t=0;t<T;t++){ const a = Math.max(0,t-R), b = Math.min(T,t+R+1); fs[t] = (pre[b]-pre[a])/(b-a); } }
    const pct = (arr, q) => { const s = Array.from(arr).sort((a,b)=>a-b); return s[Math.min(s.length-1, Math.floor(q*(s.length-1)))] ?? 0; };
    const W = Math.round((opts.win ?? 12)*fps), step = Math.max(1, Math.round(fps/2));
    const qLo = opts.qLo ?? 0.2, qHi = opts.qHi ?? 0.9, mid = opts.mid ?? 0.5;
    const gmax = pct(fs, 0.95);
    // song-wide floor, so loud instrumental passages (intros, solos) aren't taken for singing
    const gLo = pct(fs, qLo), gHi = pct(fs, qHi), gFloor = gLo + (opts.gmid ?? 0.3)*(gHi-gLo);
    const voiced = new Uint8Array(T), thrA = new Float64Array(T);
    for (let c=0;c<T;c+=step){
      const seg = fs.subarray(Math.max(0,c-W), Math.min(T,c+W));
      const lo = pct(seg, qLo), hi = pct(seg, qHi);
      const thr = Math.max(lo + mid*(hi-lo), gmax - 6, gFloor);       // never call near-silence "singing"
      for (let t=c;t<Math.min(T,c+step);t++){ thrA[t] = thr; voiced[t] = fs[t] > thr ? 1 : 0; }
    }
    const minGap = Math.round((opts.minGap ?? 0.4)*fps), minRun = Math.round((opts.minRun ?? 0.3)*fps);
    const runs = arr => { const r=[]; let s=-1; for (let t=0;t<=arr.length;t++){ const v = t<arr.length ? arr[t] : 0; if (v && s<0) s=t; if (!v && s>=0){ r.push([s,t]); s=-1; } } return r; };
    let rr = runs(voiced);
    for (let i=1;i<rr.length;i++) if (rr[i][0]-rr[i-1][1] < minGap){ for (let t=rr[i-1][1]; t<rr[i][0]; t++) voiced[t]=1; }
    rr = runs(voiced).filter(([a,b]) => b-a >= minRun);
    const offs = (n/2)/SR;
    const phrases = rr.map(([a,b]) => ({ s: a/fps + offs, e: b/fps + offs }));
    return { phrases, fps, voiced };
  }

  // ---------- Alignment of lines to phrases (dynamic programming) ----------
  // lines: [{nsyl}], phrases: [{s,e}], returns per line {start,end}
  // opts.anchorFirst: the first line must start at the first phrase (used between pinned lines)
  function alignSegment(lines, phrases, P, anchorFirst){
    const nL = lines.length, nP = phrases.length;
    if (!nL) return [];
    if (!nP) return null;
    const totalSyl = lines.reduce((a,l)=>a+l.nsyl, 0);
    let rate = P.rate || (totalSyl / Math.max(1e-3, phrases.reduce((a,p)=>a+(p.e-p.s),0)));
    let result = null;
    for (let pass=0; pass<2; pass++){
      const INF = 1e18;
      const D = Array.from({length:nL+1}, () => new Float64Array(nP+1).fill(INF));
      const B = Array.from({length:nL+1}, () => new Array(nP+1).fill(null));
      D[0][0] = 0;
      const dcost = (dur, syl) => { const l = Math.log(Math.max(dur,0.05)/(syl/rate)); return P.w*l*l; };
      for (let i=0;i<=nL;i++) for (let j=0;j<=nP;j++){
        const cur = D[i][j]; if (cur >= INF) continue;
        // skip phrase j (instrumental lead, ad-lib, misdetection)
        if (j < nP && !(anchorFirst && i === 0)){
          const c = cur + P.skip + P.skipDur*(phrases[j].e-phrases[j].s);
          if (c < D[i][j+1]){ D[i][j+1] = c; B[i][j+1] = ['skip']; }
        }
        if (i >= nL || j >= nP) continue;
        // one line over 1..3 consecutive phrases
        for (let k=1;k<=(P.maxSplit||3) && j+k<=nP;k++){
          const dur = phrases[j+k-1].e - phrases[j].s;
          let gap = 0; for (let q=j+1;q<j+k;q++) gap += phrases[q].s - phrases[q-1].e;
          const c = cur + dcost(dur - P.gapShrink*gap, lines[i].nsyl) + P.split*(k-1) + P.gap*gap;
          if (c < D[i+1][j+k]){ D[i+1][j+k] = c; B[i+1][j+k] = ['line', k]; }
        }
        // 2..3 lines sharing one phrase (no breath between them)
        for (let m=2;m<=3 && i+m<=nL;m++){
          let syl = 0; for (let q=i;q<i+m;q++) syl += lines[q].nsyl;
          const c = cur + dcost(phrases[j].e-phrases[j].s, syl) + P.merge*(m-1);
          if (c < D[i+m][j+1]){ D[i+m][j+1] = c; B[i+m][j+1] = ['merge', m]; }
        }
      }
      if (D[nL][nP] >= INF) return null;
      // backtrack
      const res = new Array(nL); let i = nL, j = nP; let used = 0, usedSyl = 0;
      while (i > 0 || j > 0){
        const b = B[i][j];
        if (!b) break;
        if (b[0] === 'skip'){ j--; continue; }
        if (b[0] === 'line'){
          const k = b[1], s = phrases[j-k].s, e = phrases[j-1].e;
          res[i-1] = {start:s, end:e}; used += e-s; usedSyl += lines[i-1].nsyl; i--; j -= k; continue;
        }
        const m = b[1], ph = phrases[j-1];
        let syl = 0; for (let q=i-m;q<i;q++) syl += lines[q].nsyl;
        let acc = 0;
        for (let q=i-m;q<i;q++){
          const s = ph.s + (ph.e-ph.s)*acc/syl; acc += lines[q].nsyl;
          res[q] = {start:s, end: ph.s + (ph.e-ph.s)*acc/syl};
        }
        used += ph.e-ph.s; usedSyl += syl; i -= m; j--;
      }
      result = res;
      if (used > 0) rate = usedSyl/used;   // refine singing rate from the matched phrases, then redo
    }
    return result;
  }

  const DEFAULTS = { w:1.0, skip:0.8, skipDur:0.04, split:0.35, merge:1.0, maxSplit:6, gap:0.8, gapShrink:0.5 };

  // Align all lines; pins: {lineIndex: time} (hard anchors from taps or word recognition)
  function align(lines, phrases, pins={}, params={}){
    const P = Object.assign({}, DEFAULTS, params);
    const nL = lines.length;
    const keys = Object.keys(pins).map(Number).filter(k => k>=0 && k<nL).sort((a,b)=>a-b);
    // keep anchors monotonic in time
    const anchors = []; for (const k of keys){ if (!anchors.length || pins[k] > anchors[anchors.length-1][1]) anchors.push([k, pins[k]]); }
    const bounds = [[0, -Infinity, false]].concat(anchors.map(([k,t]) => [k, t, true]));
    const out = new Array(nL);
    for (let b=0;b<bounds.length;b++){
      const [i0, t0, pinned] = bounds[b];
      const i1 = b+1 < bounds.length ? bounds[b+1][0] : nL;
      const t1 = b+1 < bounds.length ? bounds[b+1][1] : Infinity;
      if (i1 <= i0) continue;
      let ph = phrases.filter(p => p.e > t0 + 0.05 && p.s < t1 - 0.05).map(p => ({s: Math.max(p.s, t0), e: Math.min(p.e, t1)}));
      if (pinned){
        // the pinned line starts exactly at its pin
        if (!ph.length || ph[0].s > t0 + 0.6) ph.unshift({s: t0, e: Math.min(t1, t0 + Math.max(1, lines[i0].nsyl/3))});
        else ph[0].s = t0;
      }
      const seg = lines.slice(i0, i1);
      let r = alignSegment(seg, ph, P, pinned);
      if (!r){   // not enough phrases: spread lines evenly over the available time
        const a = pinned ? t0 : (phrases[0]?.s ?? 0), z = isFinite(t1) ? t1 : (phrases.length ? phrases[phrases.length-1].e : a + seg.length*3);
        const tot = seg.reduce((s,l)=>s+l.nsyl,0); let acc = 0;
        r = seg.map(l => { const s = a + (z-a)*acc/tot; acc += l.nsyl; return {start:s, end:a + (z-a)*acc/tot}; });
      }
      for (let q=0;q<seg.length;q++) out[i0+q] = Object.assign(r[q], {pinned: pinned && q===0});
    }
    return out;
  }

  // ---------- Word timings inside a line (syllable-proportional) ----------
  function wordTimes(line, start, end){
    const tot = line.nsyl || 1; let acc = 0;
    return line.syl.map(s => { const t = start + (end-start)*acc/tot; acc += s; return t; });
  }

  // ---------- Match recognized words (speech-to-text) to the pasted lyrics ----------
  const normW = w => w.toLowerCase().normalize('NFKD').replace(/[^\p{L}\p{N}']/gu,'').replace(/'/g,'');
  function editDist(a, b){
    const m=a.length, n=b.length; if (!m) return n; if (!n) return m;
    let prev = Array.from({length:n+1},(_,i)=>i);
    for (let i=1;i<=m;i++){ const cur=[i]; for (let j=1;j<=n;j++) cur[j]=Math.min(prev[j]+1, cur[j-1]+1, prev[j-1]+(a[i-1]===b[j-1]?0:1)); prev=cur; }
    return prev[n];
  }
  // lyricWords: [{w, line}], rec: [{w, t}] -> pins {line: time}
  function pinsFromRecognition(lines, rec){
    const lw = []; lines.forEach((l,li) => l.words.forEach((w,wi) => { const n = normW(w); if (n) lw.push({w:n, line:li, wi}); }));
    const rw = rec.map(r => ({w: normW(r.w), t: r.t})).filter(r => r.w);
    const m = lw.length, n = rw.length; if (!m || !n) return {};
    const sim = (a,b) => a===b ? 2 : (a.length>=4 && b.length>=4 && editDist(a,b) <= 1 ? 1.2 : -1);
    // semi-global alignment: unmatched recognized words are free (repeats, ad-libs); skipping lyric words costs a little
    const GL = -0.3, GR = -0.05;
    const H = Array.from({length:m+1}, () => new Float64Array(n+1));
    const Bk = Array.from({length:m+1}, () => new Int8Array(n+1));
    for (let i=1;i<=m;i++){ H[i][0] = H[i-1][0] + GL; Bk[i][0] = 1; }
    for (let j=1;j<=n;j++){ H[0][j] = 0; Bk[0][j] = 2; }
    for (let i=1;i<=m;i++) for (let j=1;j<=n;j++){
      const d = H[i-1][j-1] + sim(lw[i-1].w, rw[j-1].w), u = H[i-1][j] + GL, l = H[i][j-1] + GR;
      if (d >= u && d >= l){ H[i][j] = d; Bk[i][j] = 0; } else if (u >= l){ H[i][j] = u; Bk[i][j] = 1; } else { H[i][j] = l; Bk[i][j] = 2; }
    }
    const match = new Array(m).fill(null);
    let i = m, j = n;
    while (i > 0 && j > 0){
      const b = Bk[i][j];
      if (b === 0){ if (sim(lw[i-1].w, rw[j-1].w) > 0) match[i-1] = rw[j-1].t; i--; j--; }
      else if (b === 1) i--; else j--;
    }
    // a line gets a pin when enough of its words were heard
    const pins = {};
    lines.forEach((l, li) => {
      const idx = lw.map((x,k)=>[x,k]).filter(([x]) => x.line === li);
      const hits = idx.filter(([,k]) => match[k] !== null);
      if (hits.length >= 2 && hits.length >= idx.length*0.4){
        // syllable position of each heard word, then fit start time with the singing rate within this line
        const sylPos = []; let acc = 0; l.syl.forEach(s => { sylPos.push(acc); acc += s; });
        const pts = hits.map(([x,k]) => [sylPos[x.wi], match[k]]);
        const [p0, pN] = [pts[0], pts[pts.length-1]];
        let rate = (pN[0] > p0[0] && pN[1] > p0[1]) ? (pN[1]-p0[1])/(pN[0]-p0[0]) : 0.35;
        rate = Math.min(Math.max(rate, 0.15), 1.0);           // seconds per syllable, sane range
        pins[li] = Math.max(0, p0[1] - p0[0]*rate);
      }
    });
    // drop pins that go backwards in time
    let last = -Infinity; for (const k of Object.keys(pins).map(Number).sort((a,b)=>a-b)){ if (pins[k] <= last + 0.3) delete pins[k]; else last = pins[k]; }
    return pins;
  }

  return { parse, lineSyllables, detectPhrases, align, alignSegment, wordTimes, pinsFromRecognition, DEFAULTS };
})();
if (typeof module !== 'undefined') module.exports = Lyr;
