import urllib.parse
import os, json, urllib.request, re, sys

BASE = "https://huggingface.co"
ROOT = r"C:\Users\electrolux\Desktop\codehub\github\VoiceChanger"
OUT = os.path.join(ROOT, "rvc_models")
os.makedirs(OUT, exist_ok=True)
UA = {"User-Agent": "vc-downloader/1.0"}
WANT = 4
MINP, MAXP = 4_000_000, 300_000_000
BAD = ("hubert","rmvpe","pretrain","content","base","index.pt","crepe")
QUERIES = ["male rvc v2","rvc male voice","deep male rvc","obama rvc","trump rvc","drake rvc","goku rvc","spongebob rvc","morgan freeman rvc","eminem rvc","snoop rvc","male singer rvc"]

def gj(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=40) as r:
        return json.load(r)
def tree(mid):
    try: return gj(f"{BASE}/api/models/{mid}/tree/main?recursive=true")
    except Exception: return []
def dl(url, dest):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r, open(dest,"wb") as f:
        tot=0
        while True:
            b=r.read(1<<20)
            if not b: break
            f.write(b); tot+=len(b)
    return tot
def safe(s): return re.sub(r"[^A-Za-z0-9._-]","_",s)[:40]

existing = set(os.listdir(OUT)) if os.path.isdir(OUT) else set()
got=[]; seen=set()
for q in QUERIES:
    if len(got)>=WANT: break
    try: models=gj(f"{BASE}/api/models?search={urllib.parse.quote(q)}&limit=25&sort=downloads&direction=-1")
    except Exception as e: print("qfail",q,e); continue
    for m in models:
        if len(got)>=WANT: break
        mid=m.get("id",""); low=mid.lower()
        if any(b in low for b in BAD): continue
        name=safe(mid.split("/")[-1])
        if name in existing or name in seen: continue
        files=tree(mid)
        pths=[f for f in files if f.get("path","").lower().endswith(".pth") and MINP<=f.get("size",0)<=MAXP and not os.path.basename(f["path"]).lower().startswith(("g_","d_"))]
        idxs=[f for f in files if f.get("path","").lower().endswith(".index")]
        if not pths or not idxs: continue
        seen.add(name)
        pth=sorted(pths,key=lambda f:f.get("size",0))[0]; idx=sorted(idxs,key=lambda f:f.get("size",0))[-1]
        d=os.path.join(OUT,name); os.makedirs(d,exist_ok=True)
        try:
            pb=dl(f"{BASE}/{mid}/resolve/main/{pth['path']}",os.path.join(d,"model.pth"))
            ib=dl(f"{BASE}/{mid}/resolve/main/{idx['path']}",os.path.join(d,"model.index"))
            got.append((name,mid,q)); print(f"OK  {name:28s} pth={pb/1e6:.1f}MB idx={ib/1e6:.1f}MB  <- {mid}  [{q}]")
        except Exception as e: print("ERR",name,e)

print("\n=== MALE DOWNLOADED", len(got), "===")
for n,mid,q in got: print(f" - {n}  (repo {mid})")
