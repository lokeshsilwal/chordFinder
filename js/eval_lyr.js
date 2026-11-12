const fs = require('fs'), path = require('path');
const makeFFT = require('./fft.js'), Lyr = require('./lyrics.js');
function readWav(p){ const b = fs.readFileSync(p); let o = 12; let data, sr;
  while (o < b.length){ const id = b.toString('ascii',o,o+4), sz = b.readUInt32LE(o+4);
    if (id==='fmt ') sr = b.readUInt32LE(o+12); if (id==='data'){ data = b.subarray(o+8, o+8+sz); break; } o += 8+sz; }
  const x = new Float32Array(data.length/2); for (let i=0;i<x.length;i++) x[i] = data.readInt16LE(i*2)/32768; return {x, sr}; }
const dir = process.argv[2], params = JSON.parse(process.argv[3]||'{}'), dopts = JSON.parse(process.argv[4]||'{}');
const mode = process.argv[5] || 'auto';
const files = fs.readdirSync(dir).filter(f => f.endsWith('.json')).sort();
let errs = [], vr = [0,0], fa = [0,0], per = [];
for (const f of files){
  const g = JSON.parse(fs.readFileSync(path.join(dir,f)));
  const {x, sr} = readWav(path.join(dir, f.replace('.json','.wav')));
  const det = Lyr.detectPhrases(x, sr, makeFFT, dopts);
  // voicing accuracy
  const fps = det.fps, offs = 1024/sr;
  det.voiced.forEach((v,t) => { const tt = t/fps+offs; const gt = g.lines.some(l => tt>=l.start && tt<l.end);
    if (gt){ vr[1]++; if (v) vr[0]++; } else { fa[1]++; if (v) fa[0]++; } });
  const items = Lyr.parse(g.lines.map(l=>l.text).join('\n')).filter(i=>i.type==='line');
  let pins = {};
  if (mode === 'rec'){   // simulated speech-to-text: 50% of words heard, some misheard, timing jitter
    let seed = 1; const rnd = () => (seed = (seed*16807) % 2147483647)/2147483647;
    const rec = [];
    g.lines.forEach((l) => { const ws = l.text.split(' '); const tw = Lyr.wordTimes(items[0] && {syl: Lyr.lineSyllables(l.text), nsyl: l.syl}, l.start, l.end);
      ws.forEach((w,k) => { const r = rnd(); if (r < 0.5) rec.push({w: r < 0.1 ? w+'x' : w, t: tw[k] + (rnd()-0.5)*0.3}); }); });
    pins = Lyr.pinsFromRecognition(items, rec);
  }
  let res = Lyr.align(items, det.phrases, pins, params);
  const taps = +(process.env.TAPS||0);
  for (let k=0;k<taps;k++){   // user taps the first wrong line at the right moment (with ~0.15 s reaction jitter)
    const bad = res.findIndex((r,i) => Math.abs(r.start - g.lines[i].start) > 1.0 && !(i in pins));
    if (bad < 0) break;
    pins[bad] = g.lines[bad].start + 0.1*Math.sin(bad*7);
    res = Lyr.align(items, det.phrases, pins, params);
  }
  const e = res.map((r,i) => Math.abs(r.start - g.lines[i].start));
  errs.push(...e); per.push([f, e.filter(v=>v<0.75).length/e.length, det.phrases.length, g.lines.length, g.meta]);
}
const within = t => (100*errs.filter(v=>v<t).length/errs.length).toFixed(0)+'%';
console.log(JSON.stringify({within_0_5s: within(0.5), within_1s: within(1.0), median_err: errs.sort((a,b)=>a-b)[Math.floor(errs.length/2)].toFixed(2),
  voicing_recall: (vr[0]/vr[1]).toFixed(2), false_alarm: (fa[0]/fa[1]).toFixed(2)}));
if (process.env.PER) per.sort((a,b)=>a[1]-b[1]).slice(0,8).forEach(p => console.log(' ', p[0], (p[1]*100).toFixed(0)+'%', 'phrases', p[2], 'lines', p[3], JSON.stringify(p[4])));
