import os, json, urllib.request, re, sys

BASE = "https://huggingface.co"
ROOT = r"C:\Users\electrolux\Desktop\codehub\github\VoiceChanger"
OUT = os.path.join(ROOT, "rvc_models")
os.makedirs(OUT, exist_ok=True)
UA = {"User-Agent": "vc-downloader/1.0"}
WANT = 5
MINP, MAXP = 4_000_000, 300_000_000   # typical RVC v2 .pth size range
BAD = ("hubert", "rmvpe", "pretrain", "content", "base", "index.pt", "crepe")

def gj(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=40) as r:
        return json.load(r)

def tree(mid):
    try:
        return gj(f"{BASE}/api/models/{mid}/tree/main?recursive=true")
    except Exception:
        return []

def dl(url, dest):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as f:
        tot = 0
        while True:
            b = r.read(1 << 20)
            if not b: break
            f.write(b); tot += len(b)
    return tot

def safe(s):
    return re.sub(r"[^A-Za-z0-9._-]", "_", s)[:40]

got = []
seen = set()
try:
    models = gj(f"{BASE}/api/models?search=rvc&limit=120&sort=downloads&direction=-1")
except Exception as e:
    print("SEARCH_FAIL", e); sys.exit(1)

for m in models:
    if len(got) >= WANT: break
    mid = m.get("id", "")
    low = mid.lower()
    if any(b in low for b in BAD): continue
    files = tree(mid)
    pths = [f for f in files if f.get("path","").lower().endswith(".pth")
            and MINP <= f.get("size",0) <= MAXP
            and not os.path.basename(f["path"]).lower().startswith(("g_","d_"))]
    idxs = [f for f in files if f.get("path","").lower().endswith(".index")]
    if not pths or not idxs: continue
    pth = sorted(pths, key=lambda f: f.get("size",0))[0]
    idx = sorted(idxs, key=lambda f: f.get("size",0))[-1]
    name = safe(mid.split("/")[-1])
    if name in seen: continue
    seen.add(name)
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    try:
        pb = dl(f"{BASE}/{mid}/resolve/main/{pth['path']}", os.path.join(d, "model.pth"))
        ib = dl(f"{BASE}/{mid}/resolve/main/{idx['path']}", os.path.join(d, "model.index"))
        got.append((name, mid, round(pb/1e6,1), round(ib/1e6,1)))
        print(f"OK  {name:30s} pth={pb/1e6:.1f}MB index={ib/1e6:.1f}MB  <- {mid}")
    except Exception as e:
        print(f"ERR {name}: {e}")

print("\n=== DOWNLOADED", len(got), "VOICES ===")
for n, mid, pb, ib in got:
    print(f" - {n}  (repo {mid})")
