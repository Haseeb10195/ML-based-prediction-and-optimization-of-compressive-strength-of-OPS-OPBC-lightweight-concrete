"""
OPS/OPBC Concrete Mix Design Optimiser  — v3
XGBoost surrogate + GA / MFO / GA+MFO Hybrid

HOW TO RUN:
  1. Put this file AND Input_output_Revised.csv in the SAME folder
  2. pip install flask xgboost scikit-learn deap numpy pandas
  3. python mix_design_gui_v2.py
  4. Open http://127.0.0.1:5000
"""

from flask import Flask, render_template_string, request, jsonify
import pandas as pd, numpy as np, random, os, warnings
warnings.filterwarnings('ignore')
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from deap import base, creator, tools, algorithms

app = Flask(__name__)

# ── Find CSV ──────────────────────────────────────────────────────────────────
def find_csv():
    names = ['Input_output_Revised.csv',
             'Input_output_Claude.csv',
             'Input_output_dataset.csv']
    dirs  = [os.path.dirname(os.path.abspath(__file__)), os.getcwd(),
             '/mnt/user-data/uploads']
    for d in dirs:
        for n in names:
            p = os.path.join(d, n)
            if os.path.exists(p):
                print(f"  CSV: {p}"); return p
    # last-resort: search uploads for any matching file
    up = '/mnt/user-data/uploads'
    if os.path.isdir(up):
        for f in os.listdir(up):
            if 'Revised' in f or 'Input_output' in f:
                p = os.path.join(up, f)
                print(f"  CSV (auto): {p}"); return p
    raise FileNotFoundError("Cannot find CSV. Place Input_output_Revised.csv "
                            "in the same folder as this script.")

# ── Load data ─────────────────────────────────────────────────────────────────
df_raw = pd.read_csv(find_csv())
df_raw['ID'] = df_raw['ID'].ffill()          # forward-fill mix IDs

# Feature columns (skip ID + Cement + Water + Sand → start at col 4)
feature_cols = df_raw.columns[4:16].tolist() # 12 inputs
target_col   = df_raw.columns[16]

X = df_raw[feature_cols].values
y = df_raw[target_col].values

# Column indices within the 12-feature vector
IDX_SUPER = 0   # Superplasticizer
IDX_OPS   = 1   # OPS (kg/m3)
IDX_OPBC  = 2   # OPBC (kg/m3)
IDX_CURE  = list(range(3, 11))   # 8 curing flags
IDX_AGE   = 11  # Age (Days)

# ── Discrete OPS/OPBC mix combinations (locked ratios) ───────────────────────
# Each mix ID has a fixed (OPS, OPBC) pair — optimiser selects mix_id ∈ {0..5}
MIX_COMBOS = {}
for mid in df_raw['ID'].unique():
    row = df_raw[df_raw['ID'] == mid].iloc[0]
    MIX_COMBOS[mid] = (float(row['OPS (kg/m3)']), float(row['OPBC (kg/m3)']))

MIX_IDS    = sorted(MIX_COMBOS.keys())          # ['C0','C10','C20','C30','C40','C50']
MIX_LIST   = [(mid, *MIX_COMBOS[mid]) for mid in MIX_IDS]
# MIX_LIST[i] = (mix_id_str, OPS, OPBC)  for i in 0..5

print("  Mix combinations locked:")
for i,(mid,ops,opbc) in enumerate(MIX_LIST):
    print(f"    [{i}] {mid}: OPS={ops:.0f}  OPBC={opbc:.0f}  "
          f"ratio={opbc/ops:.4f}" if ops > 0 else f"    [{i}] {mid}: OPS={ops:.0f}  OPBC={opbc:.0f}")

# Curing names (updated for new CSV: Curing_3W)
CURE_NAMES = [
    'Full Water Curing (FW)',
    'Air Curing (AC)',
    '3-Day Water Curing (3W)',
    '5-Day Water Curing (5W)',
    '7-Day Water Curing (7W)',
    'Watered Twice/Day for 2 Days (2T2D)',
    'Watered Twice/Day for 6 Days (2T6D)',
    'Plastic Sheet Wrapped (PS)',
]
CURE_SHORT = ['FW','AC','3W','5W','7W','2T2D','2T6D','PS']

# Bounds
SUPER_MIN = float(df_raw[feature_cols[IDX_SUPER]].min())
SUPER_MAX = float(df_raw[feature_cols[IDX_SUPER]].max())
AGE_MIN   = max(1.0, float(df_raw[feature_cols[IDX_AGE]].min()))  # physical floor: age can't be < 1 day
AGE_MAX   = float(df_raw[feature_cols[IDX_AGE]].max())

# Constant materials
CEMENT = int(df_raw['Cement (kg/m3)'].iloc[0])
WATER  = int(df_raw['Water (kg/m3)'].iloc[0])
SAND   = int(df_raw['Sand (kg/m3)'].iloc[0])

# Optimisation search space: [super, mix_id(0-5), cure_id(0-7), age]
# 4 variables instead of 5 (OPS+OPBC replaced by single discrete mix_id)
LB4 = np.array([SUPER_MIN, 0.0, 0.0, AGE_MIN])
UB4 = np.array([SUPER_MAX, 5.0, 7.0, AGE_MAX])

# ── Train XGBoost ─────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_sc   = scaler.fit_transform(X)
xgb    = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5,
                       subsample=0.8, colsample_bytree=0.8,
                       random_state=42, verbosity=0)
xgb.fit(X_sc, y)
print(f"  XGBoost R² = {xgb.score(X_sc, y):.4f}")
print(f"  CS range   = {y.min():.2f} – {y.max():.2f} MPa")

# ── Surrogate helpers ─────────────────────────────────────────────────────────
def decode_mix(mix_idx):
    """Return (mix_id_str, OPS, OPBC) for integer mix_idx 0-5."""
    idx = int(round(np.clip(mix_idx, 0, len(MIX_LIST)-1)))
    return MIX_LIST[idx]

def predict_cs(sv, mix_idx, cure_idx, age):
    mid_str, ops, opbc = decode_mix(mix_idx)
    ci  = int(round(np.clip(cure_idx, 0, 7)))
    age = max(AGE_MIN, min(AGE_MAX, float(age)))  # defensive: never let age go out of bounds
    # Age constraint:
    #   Curing_FW (ci=0) → age ∈ [1, 56] days (as tested)
    #   All other curing → age fixed at 28 days
    if ci != 0:
        age = 28.0
    vec = np.zeros(12)
    vec[IDX_SUPER] = sv
    vec[IDX_OPS]   = ops
    vec[IDX_OPBC]  = opbc
    for i, c in enumerate(IDX_CURE):
        vec[c] = 1.0 if i == ci else 0.0
    vec[IDX_AGE] = age
    return float(xgb.predict(scaler.transform(vec.reshape(1,-1)))[0])

def obj4(sv, mix_idx, cure_idx, age, target):
    return abs(predict_cs(sv, mix_idx, cure_idx, age) - target)

def best_mix_scan(sv, cure_idx, age, target):
    """
    After the algorithm converges, scan all 6 mix IDs at the best
    [super, cure, age] and return the position with the lowest
    |predicted CS - target CS|.

    This guarantees the reported mix ID is always the most accurate one,
    not an arbitrary result of random initialisation.
    """
    ci = int(round(np.clip(cure_idx, 0, 7)))
    # Apply age constraint for display in log
    eff_age = float(age) if ci == 0 else 28.0

    print(f"\n  ── Mix ID scan  (target={target} MPa | "
          f"super={sv:.3f} | cure={CURE_SHORT[ci]} | age={int(round(eff_age))}d) ──")

    best_pos = None
    best_err = float('inf')

    for mix_i in range(len(MIX_LIST)):
        mid_str, ops, opbc = MIX_LIST[mix_i]
        pred = predict_cs(sv, mix_i, cure_idx, age)
        err  = abs(pred - target)
        marker = ''
        if err < best_err:
            best_err = err
            best_pos = [sv, float(mix_i), float(cure_idx), age]
            marker = '  ← best so far'
        print(f"    {mid_str} (OPS={ops:.0f}, OPBC={opbc:.0f}):  "
              f"pred={pred:.3f} MPa  |error|={err:.4f} MPa{marker}")

    winning = MIX_LIST[int(best_pos[1])]
    print(f"  → Selected: {winning[0]}  "
          f"(OPS={winning[1]:.0f}, OPBC={winning[2]:.0f})  "
          f"error={best_err:.4f} MPa\n")

    return best_pos

def make_result(pos4, target, curve, **extra):
    sv       = float(pos4[0])
    mix_idx  = pos4[1]
    cure_idx = pos4[2]
    age      = pos4[3]

    mid_str, ops, opbc = decode_mix(mix_idx)
    ci   = int(round(np.clip(cure_idx, 0, 7)))
    # Enforce age constraint before final display
    constrained_age = float(age) if ci == 0 else 28.0
    constrained_age = max(AGE_MIN, min(AGE_MAX, constrained_age))  # hard re-clip (defensive)
    pred = predict_cs(sv, mix_idx, ci, constrained_age)

    final_age = int(round(constrained_age))
    final_age = max(1, final_age)  # absolute floor: age can never be 0 or negative

    return dict(
        superplasticiser = round(sv, 2),
        mix_id           = mid_str,
        ops              = ops,
        opbc             = opbc,
        age              = final_age,   # integer, constraint + floor applied
        age_note         = '1 – 56 days (FW curing)' if ci == 0 else 'Fixed at 28 days (non-FW)',
        cure_idx         = ci,
        cure_name        = CURE_NAMES[ci],
        cure_short       = CURE_SHORT[ci],
        predicted_cs     = round(pred, 3),
        pred_error       = round(abs(pred - target), 4),
        target           = target,
        curve            = [round(v, 4) for v in curve],
        cement           = CEMENT,
        water            = WATER,
        sand             = SAND,
        **extra
    )

# ── GA ────────────────────────────────────────────────────────────────────────
def run_ga(target, n_gen=150, pop_size=200, seed=42):
    random.seed(seed); np.random.seed(seed)
    for a in ['_FGAV2','_IGAV2']:
        if hasattr(creator, a): delattr(creator, a)
    creator.create('_FGAV2', base.Fitness, weights=(-1.0,))
    creator.create('_IGAV2', list, fitness=creator._FGAV2)

    tb = base.Toolbox()
    tb.register('ind', lambda: creator._IGAV2([
        random.uniform(SUPER_MIN, SUPER_MAX),
        float(random.randint(0, 5)),          # mix_id  (discrete 0-5)
        float(random.randint(0, 7)),          # cure_id (discrete 0-7)
        random.uniform(AGE_MIN, AGE_MAX),
    ]))
    tb.register('pop', tools.initRepeat, list, tb.ind)
    tb.register('evaluate', lambda ind: (obj4(ind[0],ind[1],ind[2],ind[3],target),))

    def cx(a, b, alpha=0.5):
        # continuous: super, age — blend then CLIP to valid bounds
        for i, (lo_b, hi_b) in [(0, (SUPER_MIN, SUPER_MAX)), (3, (AGE_MIN, AGE_MAX))]:
            lo = min(a[i],b[i]) - alpha*abs(a[i]-b[i])
            hi = max(a[i],b[i]) + alpha*abs(a[i]-b[i])
            a[i] = max(lo_b, min(hi_b, random.uniform(lo, hi)))
            b[i] = max(lo_b, min(hi_b, random.uniform(lo, hi)))
        # discrete: swap mix_id and cure_id
        a[1], b[1] = b[1], a[1]
        a[2], b[2] = b[2], a[2]
        return a, b

    def mut(ind, sig=0.15, p=0.3):
        # super
        if random.random() < p:
            sp = SUPER_MAX - SUPER_MIN
            ind[0] += random.gauss(0, sig*sp)
            ind[0] = max(SUPER_MIN, min(SUPER_MAX, ind[0]))
        # mix_id (discrete)
        if random.random() < p:
            ind[1] = float(random.randint(0, 5))
        # cure_id (discrete)
        if random.random() < p:
            ind[2] = float(random.randint(0, 7))
        # age
        if random.random() < p:
            sp = AGE_MAX - AGE_MIN
            ind[3] += random.gauss(0, sig*sp)
            ind[3] = max(AGE_MIN, min(AGE_MAX, ind[3]))
        return (ind,)

    tb.register('mate',   cx)
    tb.register('mutate', mut)
    tb.register('select', tools.selTournament, tournsize=3)

    pop = tb.pop(n=pop_size); hof = tools.HallOfFame(1)
    for ind in pop: ind.fitness.values = tb.evaluate(ind)
    hof.update(pop)
    curve = []
    for _ in range(n_gen):
        offs = algorithms.varAnd(pop, tb, cxpb=0.7, mutpb=0.3)
        for ind in offs: ind.fitness.values = tb.evaluate(ind)
        pop = tb.select(offs+pop, k=pop_size); hof.update(pop)
        curve.append(min(ind.fitness.values[0] for ind in pop))

    # Scan all mix IDs at best solution to pick the most accurate one
    best = list(hof[0])
    best = best_mix_scan(best[0], best[2], best[3], target)
    return make_result(best, target, curve)

# ── MFO ───────────────────────────────────────────────────────────────────────
def run_mfo(target, n=200, iters=150, b=1.0, seed=42):
    np.random.seed(seed)
    moths   = np.random.uniform(LB4, UB4, (n, 4))
    # snap discrete dims to integers on init
    moths[:, 1] = np.round(moths[:, 1]).clip(0, 5)
    moths[:, 2] = np.round(moths[:, 2]).clip(0, 7)

    fitness = np.array([obj4(m[0],m[1],m[2],m[3],target) for m in moths])
    si      = np.argsort(fitness)
    flames  = moths[si].copy(); ff = fitness[si].copy()
    curve   = []

    for it in range(iters):
        nf = max(1, round(n - it*(n-1)/iters))
        for i in range(n):
            fp = flames[min(i, nf-1)]
            t  = np.random.uniform(-1, 1, 4)
            D  = np.abs(fp - moths[i])
            moths[i] = np.clip(D*np.exp(b*t)*np.cos(2*np.pi*t)+fp, LB4, UB4)
            # snap discrete dims
            moths[i,1] = round(moths[i,1])
            moths[i,2] = round(moths[i,2])

        fitness = np.array([obj4(m[0],m[1],m[2],m[3],target) for m in moths])
        cp = np.vstack([flames, moths]); cf = np.concatenate([ff, fitness])
        si2 = np.argsort(cf)
        flames = cp[si2[:n]].copy(); ff = cf[si2[:n]].copy()
        curve.append(float(ff[0]))

    # Scan all mix IDs at best solution to pick the most accurate one
    best = best_mix_scan(flames[0][0], flames[0][2], flames[0][3], target)
    return make_result(best, target, curve)

# ── GA+MFO Hybrid ─────────────────────────────────────────────────────────────
def run_hybrid(target, n_gen=80, n_iter=70, pop=200, n_moths=200,
               top_k=20, b=1.0, seed=42):
    random.seed(seed); np.random.seed(seed)
    for a in ['_FHYV2','_IHYV2']:
        if hasattr(creator, a): delattr(creator, a)
    creator.create('_FHYV2', base.Fitness, weights=(-1.0,))
    creator.create('_IHYV2', list, fitness=creator._FHYV2)

    tb = base.Toolbox()
    tb.register('ind', lambda: creator._IHYV2([
        random.uniform(SUPER_MIN, SUPER_MAX),
        float(random.randint(0, 5)),
        float(random.randint(0, 7)),
        random.uniform(AGE_MIN, AGE_MAX),
    ]))
    tb.register('pop_', tools.initRepeat, list, tb.ind)
    tb.register('evaluate', lambda ind: (obj4(ind[0],ind[1],ind[2],ind[3],target),))

    def cx(a, b_, alpha=0.5):
        for i, (lo_b, hi_b) in [(0, (SUPER_MIN, SUPER_MAX)), (3, (AGE_MIN, AGE_MAX))]:
            lo = min(a[i],b_[i]) - alpha*abs(a[i]-b_[i])
            hi = max(a[i],b_[i]) + alpha*abs(a[i]-b_[i])
            a[i]  = max(lo_b, min(hi_b, random.uniform(lo, hi)))
            b_[i] = max(lo_b, min(hi_b, random.uniform(lo, hi)))
        a[1], b_[1] = b_[1], a[1]
        a[2], b_[2] = b_[2], a[2]
        return a, b_

    def mut(ind, sig=0.15, p=0.3):
        if random.random() < p:
            ind[0] += random.gauss(0, sig*(SUPER_MAX-SUPER_MIN))
            ind[0] = max(SUPER_MIN, min(SUPER_MAX, ind[0]))
        if random.random() < p: ind[1] = float(random.randint(0, 5))
        if random.random() < p: ind[2] = float(random.randint(0, 7))
        if random.random() < p:
            ind[3] += random.gauss(0, sig*(AGE_MAX-AGE_MIN))
            ind[3] = max(AGE_MIN, min(AGE_MAX, ind[3]))
        return (ind,)

    tb.register('mate',   cx)
    tb.register('mutate', mut)
    tb.register('select', tools.selTournament, tournsize=3)

    population = tb.pop_(n=pop); hof = tools.HallOfFame(top_k)
    for ind in population: ind.fitness.values = tb.evaluate(ind)
    hof.update(population); ga_curve = []

    for _ in range(n_gen):
        offs = algorithms.varAnd(population, tb, cxpb=0.7, mutpb=0.3)
        for ind in offs: ind.fitness.values = tb.evaluate(ind)
        population = tb.select(offs+population, k=pop)
        hof.update(population)
        ga_curve.append(min(ind.fitness.values[0] for ind in population))

    # MFO phase seeded from GA elite
    seed_pos = np.array([[float(x) for x in ind] for ind in hof])
    extra    = np.random.uniform(LB4, UB4, (n_moths-top_k, 4))
    extra[:,1] = np.round(extra[:,1]).clip(0,5)
    extra[:,2] = np.round(extra[:,2]).clip(0,7)
    moths    = np.clip(np.vstack([seed_pos, extra]), LB4, UB4)
    moths[:,1] = np.round(moths[:,1]).clip(0,5)
    moths[:,2] = np.round(moths[:,2]).clip(0,7)

    fitness  = np.array([obj4(m[0],m[1],m[2],m[3],target) for m in moths])
    si       = np.argsort(fitness)
    flames   = moths[si].copy(); ff = fitness[si].copy()
    mfo_curve = []

    for it in range(n_iter):
        nf = max(1, round(n_moths - it*(n_moths-1)/n_iter))
        for i in range(n_moths):
            fp = flames[min(i, nf-1)]
            t  = np.random.uniform(-1, 1, 4)
            D  = np.abs(fp - moths[i])
            moths[i] = np.clip(D*np.exp(b*t)*np.cos(2*np.pi*t)+fp, LB4, UB4)
            moths[i,1] = round(moths[i,1])
            moths[i,2] = round(moths[i,2])

        fitness = np.array([obj4(m[0],m[1],m[2],m[3],target) for m in moths])
        cp = np.vstack([flames, moths]); cf = np.concatenate([ff, fitness])
        si2 = np.argsort(cf)
        flames = cp[si2[:n_moths]].copy(); ff = cf[si2[:n_moths]].copy()
        mfo_curve.append(float(ff[0]))

    # Scan all mix IDs at best solution to pick the most accurate one
    best = best_mix_scan(flames[0][0], flames[0][2], flames[0][3], target)
    return make_result(best, target, ga_curve+mfo_curve, ga_gens=n_gen)

# ── HTML ──────────────────────────────────────────────────────────────────────
PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OPS-OPBC Mix Design Optimiser</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');
:root{
  --bg:#FFFFFF;--panel:#F8FAFC;--card:#FFFFFF;--b1:#E5E7EB;--b2:#D1D5DB;
  --txt:#111827;--muted:#6B7280;--acc:#2563EB;--good:#16A34A;
  --warn:#D97706;--ga:#D97706;--mfo:#059669;--hyb:#7C3AED;--err:#DC2626;
  --note:#1E40AF;--mono:'IBM Plex Mono',monospace;--sans:'IBM Plex Sans',sans-serif;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--txt);font-family:var(--sans);min-height:100vh;padding:28px 18px 60px}
body::before{content:'';position:fixed;inset:0;z-index:0;
  background-image:linear-gradient(var(--b1) 1px,transparent 1px),linear-gradient(90deg,var(--b1) 1px,transparent 1px);
  background-size:36px 36px;opacity:.55;pointer-events:none}
.wrap{position:relative;z-index:1;max-width:980px;margin:0 auto}
.hdr{margin-bottom:28px}
.hdr-row{display:flex;align-items:center;gap:14px;margin-bottom:8px}
.hdr-logo{width:46px;height:46px;border-radius:12px;background:linear-gradient(135deg,#2563EB,#7C3AED);
  display:flex;align-items:center;justify-content:center;font-size:1.4rem;flex-shrink:0;
  box-shadow:0 8px 24px rgba(37,99,235,.12)}
h1{font-family:var(--mono);font-size:1.25rem;font-weight:700;letter-spacing:-.5px}
.hdr p{color:var(--muted);font-size:.8rem;margin-left:60px;margin-bottom:10px}
.pills{display:flex;gap:7px;flex-wrap:wrap;margin-left:60px}
.pill{font-family:var(--mono);font-size:.63rem;padding:3px 8px;border-radius:4px;border:1px solid;font-weight:700}
.pb{border-color:#BFDBFE;color:#1D4ED8;background:#EFF6FF}
.pg{border-color:#BBF7D0;color:#15803D;background:#F0FDF4}
.py{border-color:#FDE68A;color:#B45309;background:#FFFBEB}
.grid{display:grid;grid-template-columns:300px 1fr;gap:14px;align-items:start}
.panel{background:var(--panel);border:1px solid var(--b2);border-radius:14px;padding:20px}
.ptitle{font-family:var(--mono);font-size:.63rem;font-weight:700;color:var(--muted);
  letter-spacing:1.5px;text-transform:uppercase;margin-bottom:16px;
  display:flex;align-items:center;gap:7px}
.ptitle::before{content:'';display:block;width:6px;height:6px;border-radius:50%;
  background:var(--acc);box-shadow:0 0 8px var(--acc)}
.lbl{font-size:.73rem;color:var(--muted);margin-bottom:5px;font-weight:500;display:block}
.cs-row{display:flex;margin-bottom:6px}
.cs-input{flex:1;background:var(--card);border:1px solid var(--b2);border-right:none;
  color:var(--txt);padding:10px 13px;font-size:1.55rem;font-family:var(--mono);
  font-weight:700;border-radius:9px 0 0 9px;outline:none;transition:border-color .2s;
  -moz-appearance:textfield}
.cs-input::-webkit-inner-spin-button,.cs-input::-webkit-outer-spin-button{display:none}
.cs-input:focus{border-color:var(--acc)}
.unit-tag{background:var(--b1);border:1px solid var(--b2);padding:10px 11px;
  border-radius:0 9px 9px 0;font-family:var(--mono);font-size:.8rem;
  color:var(--muted);font-weight:700;display:flex;align-items:center}
.range-hint{font-size:.68rem;color:var(--muted);margin-bottom:14px}
.slider{-webkit-appearance:none;width:100%;height:4px;border-radius:4px;
  background:#E5E7EB;outline:none;margin-bottom:4px}
.slider::-webkit-slider-thumb{-webkit-appearance:none;width:15px;height:15px;
  border-radius:50%;background:var(--acc);cursor:pointer;box-shadow:0 2px 8px rgba(37,99,235,.3)}
.slider-row{display:flex;justify-content:space-between;font-size:.63rem;
  color:var(--muted);font-family:var(--mono);margin-bottom:16px}
.algo-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:16px}
.abtn{background:var(--card);border:1.5px solid var(--b2);border-radius:10px;
  padding:12px 6px;cursor:pointer;text-align:center;transition:all .18s;color:var(--muted)}
.abtn:hover{border-color:#9CA3AF;color:var(--txt)}
.aicon{font-size:1.3rem;display:block;margin-bottom:5px}
.aname{font-family:var(--mono);font-size:.7rem;font-weight:700;display:block;margin-bottom:2px}
.asub{font-size:.6rem;color:var(--muted)}
.sel-GA{border-color:var(--ga)!important;color:var(--ga)!important;background:#FFFBEB!important;box-shadow:0 0 0 3px #FDE68A44}
.sel-MFO{border-color:var(--mfo)!important;color:var(--mfo)!important;background:#F0FDF4!important;box-shadow:0 0 0 3px #86EFAC44}
.sel-Hybrid{border-color:var(--hyb)!important;color:var(--hyb)!important;background:#F5F3FF!important;box-shadow:0 0 0 3px #C4B5FD44}
.run{width:100%;padding:12px;background:linear-gradient(135deg,#2563EB,#7C3AED);
  border:none;border-radius:9px;color:#fff;font-family:var(--mono);
  font-size:.8rem;font-weight:700;cursor:pointer;letter-spacing:.4px;
  transition:opacity .2s,transform .1s;box-shadow:0 4px 16px rgba(37,99,235,.2)}
.run:hover:not(:disabled){opacity:.88;transform:translateY(-1px)}
.run:disabled{opacity:.3;cursor:not-allowed}
.sbar{display:flex;align-items:center;gap:7px;margin-top:11px;
  font-size:.7rem;color:var(--muted);font-family:var(--mono)}
.sdot{width:7px;height:7px;border-radius:50%;background:#D1D5DB;flex-shrink:0}
.sdot.ready{background:var(--good);box-shadow:0 0 6px var(--good)}
.sdot.run{background:var(--warn);animation:blink 1s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.1}}
.ph{background:var(--panel);border:1px dashed var(--b2);border-radius:14px;
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  gap:12px;padding:40px;text-align:center;min-height:360px}
.ph-ico{font-size:2.6rem;opacity:.4}
.ph-txt{color:var(--muted);font-size:.82rem;line-height:1.7;max-width:260px}
.mrow{display:grid;grid-template-columns:1fr 1fr 1fr;gap:9px;margin-bottom:12px}
.mc{background:var(--panel);border:1px solid var(--b2);border-radius:11px;padding:12px 14px}
.mc.hi{border-color:#86EFAC;background:#F0FDF4}
.mc-lbl{font-size:.63rem;color:var(--muted);margin-bottom:4px;font-weight:500;
  text-transform:uppercase;letter-spacing:.5px}
.mc-val{font-family:var(--mono);font-size:1.3rem;font-weight:700;color:var(--txt)}
.mc-val.g{color:var(--good)}
.mc-unit{font-size:.65rem;color:var(--muted);margin-left:3px}
.rtable-wrap{background:var(--panel);border:1px solid var(--b2);border-radius:13px;
  overflow:hidden;margin-bottom:10px}
.rt-head{font-family:var(--mono);font-size:.63rem;font-weight:700;
  color:var(--muted);letter-spacing:1.2px;text-transform:uppercase;
  padding:12px 16px 9px;border-bottom:1px solid var(--b1);
  display:flex;align-items:center;gap:7px}
.rt-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
table{width:100%;border-collapse:collapse}
th{font-size:.63rem;color:var(--muted);font-weight:600;padding:8px 15px;
  text-align:left;background:#F8FAFC;text-transform:uppercase;letter-spacing:.5px}
td{padding:10px 15px;border-top:1px solid var(--b1);font-size:.86rem}
tr:hover td{background:#F9FAFB}
.feat{font-weight:600;color:var(--txt)}
.fval{font-family:var(--mono);font-weight:700;font-size:.97rem;color:var(--txt)}
.funit{color:var(--muted);font-size:.73rem}
.ctag{display:inline-block;padding:3px 10px;border-radius:5px;
  font-size:.73rem;font-weight:700;font-family:var(--mono)}
.mix-badge{display:inline-block;padding:2px 8px;border-radius:4px;
  font-family:var(--mono);font-size:.72rem;font-weight:700;
  background:#EFF6FF;color:#1D4ED8;border:1px solid #BFDBFE;margin-left:6px}
/* constant note box */
.note-box{background:#EFF6FF;border:1px solid #BFDBFE;border-radius:10px;
  padding:11px 14px;margin-bottom:10px;display:flex;align-items:flex-start;gap:9px}
.note-icon{font-size:1rem;flex-shrink:0;margin-top:1px}
.note-content{font-size:.78rem;color:var(--note);line-height:1.55}
.note-content strong{font-weight:700}
.note-vals{font-family:var(--mono);font-size:.75rem;margin-top:4px;
  display:flex;gap:14px;flex-wrap:wrap}
.nval{background:#DBEAFE;padding:2px 7px;border-radius:4px;color:#1E40AF;font-weight:600}
.ccard{background:var(--panel);border:1px solid var(--b2);border-radius:13px;padding:15px 15px 11px}
.ctitle{font-family:var(--mono);font-size:.63rem;font-weight:700;
  color:var(--muted);letter-spacing:1.2px;text-transform:uppercase;margin-bottom:11px}
.chartbox{position:relative;height:150px}
.lbox{background:var(--panel);border:1px solid var(--b2);border-radius:14px;
  padding:44px 20px;display:flex;flex-direction:column;align-items:center;gap:13px;text-align:center}
.spin{width:42px;height:42px;border-radius:50%;border:3px solid var(--b1);
  animation:spin .85s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.lmsg{font-family:var(--mono);font-size:.76rem;color:var(--muted)}
.lsub{font-size:.71rem;color:#9CA3AF}
.ebox{background:#FEF2F2;border:1px solid #FECACA;border-radius:9px;
  padding:10px 13px;font-size:.78rem;color:#DC2626;font-family:var(--mono);margin-top:9px}
@media(max-width:700px){.grid{grid-template-columns:1fr}.mrow{grid-template-columns:1fr 1fr}h1{font-size:1rem}}
</style>
</head>
<body>
<div class="wrap">
  <div class="hdr">
    <div class="hdr-row">
      <div class="hdr-logo">🏗️</div>
      <h1>OPS-OPBC Mix Design Optimiser For Target Compressive Strength</h1>
    </div>
    <p>XGBoost surrogate model &nbsp;·&nbsp; inverse prediction &nbsp;·&nbsp; metaheuristic search</p>
    <div class="pills">
      <span class="pill pb">R² = {{ r2 }}</span>
      <span class="pill pg">RMSE = {{ rmse }} MPa</span>
      <span class="pill py">n = {{ n }} samples</span>
      <span class="pill pb">CS {{ cs_min }} – {{ cs_max }} MPa</span>
    </div>
  </div>

  <div class="grid">
    <div class="panel">
      <div class="ptitle">Configuration</div>
      <label class="lbl">Target compressive strength</label>
      <div class="cs-row">
        <input type="number" class="cs-input" id="csIn" value="35"
               min="{{ cs_min }}" max="{{ cs_max }}" step="0.5"
               oninput="document.getElementById('csSlide').value=this.value">
        <div class="unit-tag">MPa</div>
      </div>
      <input type="range" class="slider" id="csSlide"
             min="{{ cs_min }}" max="{{ cs_max }}" step="0.5" value="35"
             oninput="document.getElementById('csIn').value=this.value">
      <div class="slider-row">
        <span>{{ cs_min }}</span><span>{{ cs_mid }}</span><span>{{ cs_max }}</span>
      </div>
      <div class="range-hint">Dataset range: {{ cs_min }} – {{ cs_max }} MPa</div>

      <label class="lbl">Optimisation algorithm</label>
      <div class="algo-grid">
        <button class="abtn" id="bGA" onclick="pick('GA')">
          <span class="aicon">🧬</span>
          <span class="aname">GA</span>
          <span class="asub">Genetic Algorithm</span>
        </button>
        <button class="abtn" id="bMFO" onclick="pick('MFO')">
          <span class="aicon">🦋</span>
          <span class="aname">MFO</span>
          <span class="asub">Moth-Flame</span>
        </button>
        <button class="abtn" id="bHybrid" onclick="pick('Hybrid')">
          <span class="aicon">⚡</span>
          <span class="aname">Hybrid</span>
          <span class="asub">GA + MFO</span>
        </button>
      </div>

      <button class="run" id="runBtn" onclick="go()" disabled>Select an algorithm</button>
      <div class="sbar">
        <div class="sdot" id="sdot"></div>
        <span id="stxt">Awaiting configuration</span>
      </div>
      <div class="ebox" id="errBox" style="display:none"></div>

      <!-- Constant materials note — always visible -->
      <div class="note-box" style="margin-top:16px">
        <div class="note-icon">📌</div>
        <div class="note-content">
          <strong>Note:</strong> The following mix constituents remain
          constant throughout the mix design and are not subject to optimisation:
          <div class="note-vals">
            <span class="nval">Cement = {{ cement }} kg/m³</span>
            <span class="nval">Water = {{ water }} kg/m³</span>
            <span class="nval">Sand = {{ sand }} kg/m³</span>
          </div>
        </div>
      </div>
    </div>

    <div id="rArea">
      <div class="ph">
        <div class="ph-ico">🔬</div>
        <div class="ph-txt">
          Enter a target CS, choose an algorithm,<br>and click <strong>Run Optimisation</strong>.<br><br>
          The XGBoost surrogate will find the optimal
          Mix ID (OPS/OPBC ratio) · Superplasticiser · Age · Curing Method
          without running a single experiment.
        </div>
      </div>
    </div>
  </div>
</div>

<script>
let algo=null,chart=null;
const COL={GA:'#D97706',MFO:'#059669',Hybrid:'#7C3AED'};
const CTAG={FW:'#3B82F6',AC:'#EF4444','3W':'#10B981','5W':'#6366F1',
            '7W':'#8B5CF6','2T2D':'#F59E0B','2T6D':'#EC4899',PS:'#14B8A6'};

function pick(a){
  algo=a;
  ['GA','MFO','Hybrid'].forEach(x=>document.getElementById('b'+x).className='abtn'+(x===a?' sel-'+x:''));
  document.getElementById('runBtn').disabled=false;
  document.getElementById('runBtn').textContent='▶  Run optimisation with '+a;
}
function setStatus(msg,state='idle'){
  document.getElementById('stxt').textContent=msg;
  document.getElementById('sdot').className='sdot'+(state!=='idle'?' '+state:'');
}
async function go(){
  const target=parseFloat(document.getElementById('csIn').value);
  const errEl=document.getElementById('errBox');
  errEl.style.display='none';
  if(!algo){errEl.style.display='block';errEl.textContent='Please select an algorithm.';return;}
  if(isNaN(target)||target<5||target>70){errEl.style.display='block';errEl.textContent='Target out of range.';return;}

  document.getElementById('runBtn').disabled=true;
  setStatus('Running '+algo+'…','run');
  const msgs={
    GA:    ['Initialising population…','Evaluating fitness…','Crossover + mutation…','Converging…'],
    MFO:   ['Releasing moths…','Spiral search…','Flames updating…','Converging…'],
    Hybrid:['GA Phase 1: evolving…','Selecting elites…','MFO Phase 2: refining…','Converging…'],
  }[algo];
  let mi=0;
  document.getElementById('rArea').innerHTML=`
    <div class="lbox">
      <div class="spin" style="border-top-color:${COL[algo]}"></div>
      <div class="lmsg" id="lmsg">${msgs[0]}</div>
      <div class="lsub">XGBoost evaluating candidates — please wait…</div>
    </div>`;
  const iv=setInterval(()=>{mi++;const e=document.getElementById('lmsg');if(e)e.textContent=msgs[mi%msgs.length];},1600);
  try{
    const res=await fetch('/opt',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({target,algo})});
    clearInterval(iv);
    const d=await res.json();
    if(d.server_error){showErr(d.server_error);return;}
    render(d,algo);
    setStatus('Done  ·  |pred − target| = '+d.pred_error+' MPa','ready');
  }catch(e){clearInterval(iv);showErr('Network error: '+e.message);}
  document.getElementById('runBtn').disabled=false;
}
function showErr(msg){
  document.getElementById('rArea').innerHTML='';
  const eb=document.getElementById('errBox');
  eb.style.display='block';eb.textContent='Error: '+msg;
  setStatus('Error');document.getElementById('runBtn').disabled=false;
}
function render(d,a){
  const col=COL[a];
  const ctcol=CTAG[d.cure_short]||col;
  document.getElementById('rArea').innerHTML=`
  <div style="display:flex;flex-direction:column;gap:11px">

    <div class="mrow">
      <div class="mc hi">
        <div class="mc-lbl">Predicted CS</div>
        <div class="mc-val g">${d.predicted_cs}<span class="mc-unit">MPa</span></div>
      </div>
      <div class="mc">
        <div class="mc-lbl">Target CS</div>
        <div class="mc-val">${d.target}<span class="mc-unit">MPa</span></div>
      </div>
      <div class="mc">
        <div class="mc-lbl">|Pred − Target|</div>
        <div class="mc-val" style="color:${d.pred_error<0.5?'var(--good)':'var(--warn)'}">
          ${d.pred_error}<span class="mc-unit">MPa</span></div>
      </div>
    </div>

    <!-- Optimal mix design table -->
    <div class="rtable-wrap">
      <div class="rt-head">
        <div class="rt-dot" style="background:${col};box-shadow:0 0 6px ${col}"></div>
        Optimal mix design &nbsp;·&nbsp; Algorithm: <span style="color:${col};margin-left:3px">${a}</span>
      </div>
      <table>
        <thead><tr>
          <th>Input parameter</th><th>Optimal value</th><th>Unit</th><th>Dataset range</th>
        </tr></thead>
        <tbody>
          </tr>
          <tr>
            <td class="feat">OPS aggregate</td>
            <td class="fval">${d.ops}</td>
            <td class="funit">kg/m³</td>
            <td class="funit">{{ ops_range }}</td>
          </tr>
          <tr>
            <td class="feat">OPBC aggregate</td>
            <td class="fval">${d.opbc}</td>
            <td class="funit">kg/m³</td>
            <td class="funit">{{ opbc_range }}</td>
          </tr>
          <tr>
            <td class="feat">Superplasticiser</td>
            <td class="fval">${d.superplasticiser}</td>
            <td class="funit">kg/m³</td>
            <td class="funit">{{ super_range }}</td>
          </tr>
          <tr>
            <td class="feat">Specimen age</td>
            <td class="fval">${d.age}</td>
            <td class="funit">days</td>
            <td class="funit">{{ age_range }}</td>
          </tr>
          <tr>
            <td class="feat">Curing method</td>
            <td colspan="3">
              <span class="ctag" style="background:${ctcol}18;color:${ctcol};border:1px solid ${ctcol}40">
                ${d.cure_name}
              </span>
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <div class="ccard">
      <div class="ctitle">Convergence — |predicted − target| per iteration</div>
      <div class="chartbox"><canvas id="cv"></canvas></div>
    </div>
  </div>`;

  if(chart) chart.destroy();
  chart=new Chart(document.getElementById('cv').getContext('2d'),{
    type:'line',
    data:{labels:d.curve.map((_,i)=>i+1),datasets:[{label:a,data:d.curve,
      borderColor:col,backgroundColor:col+'18',borderWidth:2,pointRadius:0,fill:true,tension:0.4}]},
    options:{animation:{duration:420},responsive:true,maintainAspectRatio:false,
      scales:{
        x:{ticks:{color:'#6B7280',maxTicksLimit:8,font:{family:'IBM Plex Mono',size:9}},
           grid:{color:'#E5E7EB'},title:{display:true,text:'Iteration / Generation',color:'#9CA3AF',font:{size:9}}},
        y:{ticks:{color:'#6B7280',font:{family:'IBM Plex Mono',size:9}},
           grid:{color:'#E5E7EB'},title:{display:true,text:'|Error| (MPa)',color:'#9CA3AF',font:{size:9}}}
      },
      plugins:{legend:{labels:{color:'#6B7280',font:{family:'IBM Plex Mono',size:9}}},
               tooltip:{callbacks:{label:c=>' Error: '+c.parsed.y.toFixed(4)+' MPa'}}}}
  });
}
</script>
</body>
</html>"""

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    from sklearn.metrics import mean_squared_error as mse
    r2   = round(xgb.score(X_sc, y), 4)
    rmse = round(float(np.sqrt(mse(y, xgb.predict(X_sc)))), 2)
    all_ops  = sorted(df_raw['OPS (kg/m3)'].unique())
    all_opbc = sorted(df_raw['OPBC (kg/m3)'].unique())
    return render_template_string(PAGE,
        r2=r2, rmse=rmse, n=len(df_raw),
        cs_min=round(float(y.min()),1),
        cs_max=round(float(y.max()),1),
        cs_mid=round(float((y.min()+y.max())/2),1),
        super_range=f"{SUPER_MIN:.2f} – {SUPER_MAX:.2f}",
        ops_range  =f"{min(all_ops):.0f} – {max(all_ops):.0f}",
        opbc_range =f"{min(all_opbc):.0f} – {max(all_opbc):.0f}",
        age_range  =f"{AGE_MIN:.0f} – {AGE_MAX:.0f}",
        cement=CEMENT, water=WATER, sand=SAND,
    )

@app.route('/opt', methods=['POST'])
def optimise():
    try:
        d      = request.get_json(force=True)
        target = float(d['target'])
        algo   = str(d['algo'])
        if   algo == 'GA':     r = run_ga(target)
        elif algo == 'MFO':    r = run_mfo(target)
        elif algo == 'Hybrid': r = run_hybrid(target)
        else: return jsonify({'server_error': f'Unknown algorithm: {algo}'})
        return jsonify(r)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'server_error': str(e)})

if __name__ == '__main__':
    print('\n' + '='*55)
    print('  OPS/OPBC Mix Design Optimiser  v2')
    print('='*55)
    print(f'  Samples    : {len(df_raw)}')
    print(f'  CS range   : {y.min():.2f} – {y.max():.2f} MPa')
    print(f'  Mix IDs    : {MIX_IDS}')
    print(f'  Constants  : Cement={CEMENT}  Water={WATER}  Sand={SAND}')
    print('='*55)
    print('  → Open:  http://127.0.0.1:5000')
    print('  → Stop:  Ctrl+C')
    print('='*55 + '\n')
    app.run(debug=False, port=5000)
