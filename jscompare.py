import sys, chords, json
from playwright.sync_api import sync_playwright
PCS={'C':0,'C#':1,'Db':1,'D':2,'D#':3,'Eb':3,'E':4,'F':5,'F#':6,'Gb':6,'G':7,'G#':8,'Ab':8,'A':9,'A#':10,'Bb':10,'B':11}
def norm_js(n):
    if n=='N.C.': return 'N'
    r = n[:2] if len(n)>1 and n[1] in '#b' else n[:1]; q=n[len(r):]
    return f"{PCS[r]}:{ {'':'maj','m':'min'}.get(q,q)}"
def norm_py(l):
    if l=='N': return 'N'
    r,q=l.split(':'); return f"{PCS[r]}:{q}"
page_path = sys.argv[1]; songs = sys.argv[2:]
with sync_playwright() as p:
    b=p.chromium.launch(executable_path='/usr/bin/google-chrome'); pg=b.new_page()
    for s in songs:
        pg.goto('file://'+page_path)
        pg.set_input_files('#file', f'songs/{s}.wav'); pg.wait_for_selector('#results:not([hidden])', timeout=60000)
        js=[norm_js(c.query_selector('.n').inner_text()) for c in pg.query_selector_all('.cell')]
        py=[norm_py(x[2]) for x in chords.analyze(chords.load(f"songs/{s}.wav"), {"beat":True,"whiten":True,"templates":"harmonic","bass_w":0.25,"penalty":0.26})[0]]
        print(s, 'match' if js==py else 'DIFF', len(js), len(py))
        if js!=py: print(' js', js[:12]); print(' py', py[:12])
    b.close()
