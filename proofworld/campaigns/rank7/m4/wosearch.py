import subprocess, json, random, os
random.seed(1)
best=None
for it in range(40):
    WO={o:random.randint(1,20-o) for o in (1,2,4,5,7,8,10)}
    W11=random.randint(1,9)
    env=dict(os.environ,WO=json.dumps(WO),W11=str(W11),OUT=f'wo_{it}.pkl')
    out=subprocess.run(['nice','python3','coords2.py'],env=env,capture_output=True,text=True).stdout
    sc=[l for l in out.splitlines() if l.startswith('SCORE') or l.startswith('R check')]
    print(it,WO,W11,sc,flush=True)
