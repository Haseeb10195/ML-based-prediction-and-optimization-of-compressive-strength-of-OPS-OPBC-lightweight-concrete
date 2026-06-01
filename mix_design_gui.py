"""
OPS/OPBC Concrete Mix Design Optimiser
XGBoost surrogate + GA / MFO / GA+MFO Hybrid

HOW TO RUN:
  1. Put this file and Input_output_dataset.csv in the SAME folder
  2. Install: pip install flask xgboost scikit-learn deap numpy pandas
  3. Run:     python mix_design_gui.py
  4. Open:    http://127.0.0.1:5000
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
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Input_output_dataset.csv'),
        os.path.join(os.getcwd(), 'Input_output_dataset.csv'),
        '/mnt/user-data/uploads/Input_output_dataset.csv',
    ]
    for p in candidates:
        if os.path.exists(p):
            print(f"  CSV found: {p}")
            return p
    raise FileNotFoundError(
        "Cannot find Input_output_dataset.csv\n"
        "Place it in the same folder as mix_design_gui.py"
    )

# ── Load data ─────────────────────────────────────────────────────────────────
df           = pd.read_csv(find_csv())
feature_cols = df.columns[3:15].tolist()
target_col   = df.columns[15]
X = df[feature_cols].values
y = df[target_col].values

IDX_SUPER = 0; IDX_OPS = 1; IDX_OPBC = 2
IDX_CURE  = list(range(3, 11)); IDX_AGE  = 11

CURE_NAMES = [
    'Full Water (FW)', 'Air Curing (AC)', '2-Day Water (3W)',
    '5-Day Water (5W)', '7-Day Water (7W)',
    'Watered Twice a Day for 2 Days (2T2D)', 'Watered Twice a Day for 6 Days (2T6D)', 'Wrapped in 4 Layers of a 1 mm Thick Plastic Sheet (PS)'
]
CURE_SHORT = ['FW','AC','2W','5W','7W','2T2D','2T6D','PS']

B = {
    'super': (float(df[feature_cols[IDX_SUPER]].min()), float(df[feature_cols[IDX_SUPER]].max())),
    'ops':   (float(df[feature_cols[IDX_OPS]].min()),   float(df[feature_cols[IDX_OPS]].max())),
    'opbc':  (float(df[feature_cols[IDX_OPBC]].min()),  float(df[feature_cols[IDX_OPBC]].max())),
    'age':   (float(df[feature_cols[IDX_AGE]].min()),   float(df[feature_cols[IDX_AGE]].max())),
}
LB = np.array([B['super'][0], B['ops'][0], B['opbc'][0], 0.0, B['age'][0]])
UB = np.array([B['super'][1], B['ops'][1], B['opbc'][1], 7.0, B['age'][1]])

scaler = StandardScaler()
X_sc   = scaler.fit_transform(X)
xgb    = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5,
                       subsample=0.8, colsample_bytree=0.8,
                       random_state=42, verbosity=0)
xgb.fit(X_sc, y)
print(f"  XGBoost R² = {xgb.score(X_sc, y):.4f}")
print(f"  CS range   = {y.min():.2f} – {y.max():.2f} MPa")

# ── Surrogate ─────────────────────────────────────────────────────────────────
def predict_cs(sv, ops, opbc, ci, age):
    vec = np.zeros(12)
    vec[IDX_SUPER] = sv; vec[IDX_OPS] = ops; vec[IDX_OPBC] = opbc
    ci  = int(round(np.clip(ci, 0, 7)))
    for i, c in enumerate(IDX_CURE):
        vec[c] = 1.0 if i == ci else 0.0
    vec[IDX_AGE] = age
    return float(xgb.predict(scaler.transform(vec.reshape(1,-1)))[0])

def obj(sv, ops, opbc, ci, age, target):
    return abs(predict_cs(sv, ops, opbc, ci, age) - target)

def make_result(pos, target, curve, **extra):
    ci   = int(round(np.clip(float(pos[3]), 0, 7)))
    pred = predict_cs(float(pos[0]), float(pos[1]), float(pos[2]), ci, float(pos[4]))
    return dict(
        superplasticiser = round(float(pos[0]), 3),
        ops              = round(float(pos[1]), 2),
        opbc             = round(float(pos[2]), 2),
        age              = round(float(pos[4]), 1),
        cure_idx         = ci,
        cure_name        = CURE_NAMES[ci],
        cure_short       = CURE_SHORT[ci],
        predicted_cs     = round(pred, 3),
        pred_error       = round(abs(pred - target), 4),
        target           = target,
        curve            = [round(v, 4) for v in curve],
        **extra
    )

# ── GA ────────────────────────────────────────────────────────────────────────
def run_ga(target, n_gen=150, pop_size=200, seed=42):
    random.seed(seed); np.random.seed(seed)
    for a in ['_FGA','_IGA']:
        if hasattr(creator, a): delattr(creator, a)
    creator.create('_FGA', base.Fitness, weights=(-1.0,))
    creator.create('_IGA', list, fitness=creator._FGA)

    tb = base.Toolbox()
    tb.register('ind', lambda: creator._IGA([
        random.uniform(*B['super']), random.uniform(*B['ops']),
        random.uniform(*B['opbc']), float(random.randint(0,7)),
        random.uniform(*B['age']),
    ]))
    tb.register('pop', tools.initRepeat, list, tb.ind)
    tb.register('evaluate', lambda ind: (obj(ind[0],ind[1],ind[2],ind[3],ind[4],target),))

    def cx(a,b,alpha=0.5):
        for i in [0,1,2,4]:
            lo=min(a[i],b[i])-alpha*abs(a[i]-b[i]); hi=max(a[i],b[i])+alpha*abs(a[i]-b[i])
            a[i]=random.uniform(lo,hi); b[i]=random.uniform(lo,hi)
        a[3],b[3]=b[3],a[3]; return a,b

    def mut(ind, sig=0.15, p=0.3):
        for i,k in enumerate(['super','ops','opbc',None,'age']):
            if k and random.random()<p:
                sp=B[k][1]-B[k][0]; ind[i]+=random.gauss(0,sig*sp)
                ind[i]=max(B[k][0],min(B[k][1],ind[i]))
        if random.random()<p: ind[3]=float(random.randint(0,7))
        return (ind,)

    tb.register('mate',cx); tb.register('mutate',mut)
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

    return make_result(list(hof[0]), target, curve)

# ── MFO ───────────────────────────────────────────────────────────────────────
def run_mfo(target, n=200, iters=150, b=1.0, seed=42):
    np.random.seed(seed)
    moths   = np.random.uniform(LB, UB, (n, 5))
    fitness = np.array([obj(*m[:3], m[3], m[4], target) for m in moths])
    si      = np.argsort(fitness)
    flames  = moths[si].copy(); ff = fitness[si].copy()
    curve   = []
    for it in range(iters):
        nf = max(1, round(n - it*(n-1)/iters))
        for i in range(n):
            fp = flames[min(i, nf-1)]
            t  = np.random.uniform(-1, 1, 5)
            D  = np.abs(fp - moths[i])
            moths[i] = np.clip(D*np.exp(b*t)*np.cos(2*np.pi*t)+fp, LB, UB)
        fitness = np.array([obj(*m[:3], m[3], m[4], target) for m in moths])
        cp = np.vstack([flames, moths]); cf = np.concatenate([ff, fitness])
        si2 = np.argsort(cf)
        flames = cp[si2[:n]].copy(); ff = cf[si2[:n]].copy()
        curve.append(float(ff[0]))
    return make_result(flames[0], target, curve)

# ── GA+MFO Hybrid ─────────────────────────────────────────────────────────────
def run_hybrid(target, n_gen=80, n_iter=70, pop=200, n_moths=200, top_k=20, b=1.0, seed=42):
    random.seed(seed); np.random.seed(seed)
    for a in ['_FHY','_IHY']:
        if hasattr(creator, a): delattr(creator, a)
    creator.create('_FHY', base.Fitness, weights=(-1.0,))
    creator.create('_IHY', list, fitness=creator._FHY)

    tb = base.Toolbox()
    tb.register('ind', lambda: creator._IHY([
        random.uniform(*B['super']), random.uniform(*B['ops']),
        random.uniform(*B['opbc']), float(random.randint(0,7)),
        random.uniform(*B['age']),
    ]))
    tb.register('pop_', tools.initRepeat, list, tb.ind)
    tb.register('evaluate', lambda ind: (obj(ind[0],ind[1],ind[2],ind[3],ind[4],target),))

    def cx(a,b_,alpha=0.5):
        for i in [0,1,2,4]:
            lo=min(a[i],b_[i])-alpha*abs(a[i]-b_[i]); hi=max(a[i],b_[i])+alpha*abs(a[i]-b_[i])
            a[i]=random.uniform(lo,hi); b_[i]=random.uniform(lo,hi)
        a[3],b_[3]=b_[3],a[3]; return a,b_

    def mut(ind, sig=0.15, p=0.3):
        for i,k in enumerate(['super','ops','opbc',None,'age']):
            if k and random.random()<p:
                sp=B[k][1]-B[k][0]; ind[i]+=random.gauss(0,sig*sp)
                ind[i]=max(B[k][0],min(B[k][1],ind[i]))
        if random.random()<p: ind[3]=float(random.randint(0,7))
        return (ind,)

    tb.register('mate',cx); tb.register('mutate',mut)
    tb.register('select', tools.selTournament, tournsize=3)

    population = tb.pop_(n=pop); hof = tools.HallOfFame(top_k)
    for ind in population: ind.fitness.values = tb.evaluate(ind)
    hof.update(population); ga_curve = []

    for _ in range(n_gen):
        offs = algorithms.varAnd(population, tb, cxpb=0.7, mutpb=0.3)
        for ind in offs: ind.fitness.values = tb.evaluate(ind)
        population = tb.select(offs+population, k=pop); hof.update(population)
        ga_curve.append(min(ind.fitness.values[0] for ind in population))

    # Seed MFO from GA elite
    seed_pos = np.array([[float(x) for x in ind] for ind in hof])
    extra    = np.random.uniform(LB, UB, (n_moths-top_k, 5))
    moths    = np.clip(np.vstack([seed_pos, extra]), LB, UB)
    fitness  = np.array([obj(*m[:3], m[3], m[4], target) for m in moths])
    si       = np.argsort(fitness)
    flames   = moths[si].copy(); ff = fitness[si].copy()
    mfo_curve = []
    for it in range(n_iter):
        nf = max(1, round(n_moths - it*(n_moths-1)/n_iter))
        for i in range(n_moths):
            fp = flames[min(i, nf-1)]
            t  = np.random.uniform(-1, 1, 5)
            D  = np.abs(fp - moths[i])
            moths[i] = np.clip(D*np.exp(b*t)*np.cos(2*np.pi*t)+fp, LB, UB)
        fitness = np.array([obj(*m[:3], m[3], m[4], target) for m in moths])
        cp = np.vstack([flames, moths]); cf = np.concatenate([ff, fitness])
        si2 = np.argsort(cf)
        flames = cp[si2[:n_moths]].copy(); ff = cf[si2[:n_moths]].copy()
        mfo_curve.append(float(ff[0]))

    return make_result(flames[0], target, ga_curve+mfo_curve, ga_gens=n_gen)

# ── HTML ──────────────────────────────────────────────────────────────────────
PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OPS-OPBC Mix Design Optimiser For Target Compressive Strength</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');
:root{
  --bg:#FFFFFF; --panel:#F8FAFC; --card:#FFFFFF; --b1:#E5E7EB; --b2:#D1D5DB;
  --txt:#111827; --muted:#6B7280; --acc:#2563EB; --good:#16A34A;
  --warn:#F59E0B; --ga:#F59E0B; --mfo:#10B981; --hyb:#A78BFA; --err:#F87171;
  --mono:'IBM Plex Mono',monospace; --sans:'IBM Plex Sans',sans-serif;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--txt);font-family:var(--sans);min-height:100vh;padding:28px 18px 60px}
/* grid bg */
body::before{content:'';position:fixed;inset:0;z-index:0;
  background-image:linear-gradient(var(--b1) 1px,transparent 1px),linear-gradient(90deg,var(--b1) 1px,transparent 1px);
  background-size:36px 36px;opacity:.55;pointer-events:none}
.wrap{position:relative;z-index:1;max-width:960px;margin:0 auto}

/* header */
.hdr{margin-bottom:30px}
.hdr-row{display:flex;align-items:center;gap:14px;margin-bottom:8px}
.hdr-logo{width:46px;height:46px;border-radius:12px;background:linear-gradient(135deg,#2563EB,#7C3AED);
  display:flex;align-items:center;justify-content:center;font-size:1.4rem;flex-shrink:0;box-shadow:0 8px 24px rgba(37,99,235,.12)}
h1{font-family:var(--mono);font-size:1.3rem;font-weight:700;letter-spacing:-.5px}
.hdr p{color:var(--muted);font-size:.82rem;margin-left:60px;margin-bottom:12px}
.pills{display:flex;gap:8px;flex-wrap:wrap;margin-left:60px}
.pill{font-family:var(--mono);font-size:.65rem;padding:3px 9px;border-radius:4px;border:1px solid;font-weight:700}
.pb{border-color:#BFDBFE;color:#2563EB;background:#EFF6FF}
.pg{border-color:#BBF7D0;color:#16A34A;background:#F0FDF4}
.py{border-color:#FDE68A;color:#D97706;background:#FFFBEB}

/* layout */
.grid{display:grid;grid-template-columns:300px 1fr;gap:14px;align-items:start}

/* panels */
.panel{background:var(--panel);border:1px solid var(--b2);border-radius:14px;padding:20px}
.ptitle{font-family:var(--mono);font-size:.65rem;font-weight:700;color:var(--muted);
  letter-spacing:1.5px;text-transform:uppercase;margin-bottom:18px;
  display:flex;align-items:center;gap:7px}
.ptitle::before{content:'';display:block;width:6px;height:6px;border-radius:50%;
  background:var(--acc);box-shadow:0 0 8px var(--acc)}

/* input */
.lbl{font-size:.75rem;color:var(--muted);margin-bottom:6px;font-weight:500;display:block}
.cs-row{display:flex;margin-bottom:6px}
.cs-input{flex:1;background:var(--card);border:1px solid var(--b2);border-right:none;
  color:var(--txt);padding:11px 14px;font-size:1.6rem;font-family:var(--mono);
  font-weight:700;border-radius:9px 0 0 9px;outline:none;transition:border-color .2s;
  -moz-appearance:textfield}
.cs-input::-webkit-outer-spin-button,.cs-input::-webkit-inner-spin-button{display:none}
.cs-input:focus{border-color:var(--acc)}
.unit-tag{background:var(--b2);border:1px solid var(--b2);padding:11px 12px;
  border-radius:0 9px 9px 0;font-family:var(--mono);font-size:.82rem;
  color:var(--muted);font-weight:700;display:flex;align-items:center}
.range-hint{font-size:.7rem;color:var(--muted);margin-bottom:16px}

/* slider */
.slider{-webkit-appearance:none;width:100%;height:4px;border-radius:4px;
  background:#E5E7EB;outline:none;margin-bottom:4px}
.slider::-webkit-slider-thumb{-webkit-appearance:none;width:16px;height:16px;
  border-radius:50%;background:var(--acc);cursor:pointer;box-shadow:0 0 10px var(--acc)}
.slider-row{display:flex;justify-content:space-between;font-size:.65rem;
  color:var(--muted);font-family:var(--mono);margin-bottom:18px}

/* algo btns */
.algo-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:18px}
.abtn{background:var(--card);border:1.5px solid var(--b2);border-radius:10px;
  padding:13px 6px;cursor:pointer;text-align:center;transition:all .18s;color:var(--muted)}
.abtn:hover{border-color:var(--b2);color:var(--txt)}
.aicon{font-size:1.35rem;display:block;margin-bottom:5px}
.aname{font-family:var(--mono);font-size:.72rem;font-weight:700;display:block;margin-bottom:2px}
.asub{font-size:.62rem;color:var(--muted)}
.sel-GA   {border-color:var(--ga)!important;color:var(--ga)!important;background:#FFFBEB!important;box-shadow:0 0 0 3px #FDE68A33}
.sel-MFO  {border-color:var(--mfo)!important;color:var(--mfo)!important;background:#F0FDF4!important;box-shadow:0 0 0 3px #BBF7D033}
.sel-Hybrid{border-color:var(--hyb)!important;color:var(--hyb)!important;background:#F5F3FF!important;box-shadow:0 0 0 3px #DDD6FE66}

/* run btn */
.run{width:100%;padding:12px;background:linear-gradient(135deg,#2563EB,#7C3AED);
  border:none;border-radius:9px;color:#fff;font-family:var(--mono);
  font-size:.82rem;font-weight:700;cursor:pointer;letter-spacing:.4px;
  transition:opacity .2s,transform .1s;box-shadow:0 4px 20px #3B82F625}
.run:hover:not(:disabled){opacity:.88;transform:translateY(-1px)}
.run:disabled{opacity:.32;cursor:not-allowed}
.sbar{display:flex;align-items:center;gap:7px;margin-top:12px;
  font-size:.72rem;color:var(--muted);font-family:var(--mono)}
.sdot{width:7px;height:7px;border-radius:50%;background:#D1D5DB;flex-shrink:0}
.sdot.ready{background:var(--good);box-shadow:0 0 7px var(--good)}
.sdot.run{background:var(--warn);animation:blink 1s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.15}}

/* placeholder */
.ph{background:var(--panel);border:1px dashed var(--b2);border-radius:14px;
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  gap:12px;padding:40px;text-align:center;min-height:360px}
.ph-ico{font-size:2.8rem;opacity:.45}
.ph-txt{color:var(--muted);font-size:.83rem;line-height:1.65;max-width:240px}

/* metrics */
.mrow{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-bottom:12px}
.mc{background:var(--panel);border:1px solid var(--b2);border-radius:11px;padding:13px 15px}
.mc.hi{border-color:#86EFAC;background:#F0FDF4}
.mc-lbl{font-size:.65rem;color:var(--muted);margin-bottom:4px;font-weight:500;
  text-transform:uppercase;letter-spacing:.5px}
.mc-val{font-family:var(--mono);font-size:1.35rem;font-weight:700;color:var(--txt)}
.mc-val.g{color:var(--good)}
.mc-unit{font-size:.68rem;color:var(--muted);margin-left:3px}

/* results table */
.rtable-wrap{background:var(--panel);border:1px solid var(--b2);border-radius:13px;
  overflow:hidden;margin-bottom:12px}
.rt-head{font-family:var(--mono);font-size:.65rem;font-weight:700;
  color:var(--muted);letter-spacing:1.2px;text-transform:uppercase;
  padding:13px 16px 9px;border-bottom:1px solid var(--b1);
  display:flex;align-items:center;gap:7px}
.rt-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
table{width:100%;border-collapse:collapse}
th{font-size:.65rem;color:var(--muted);font-weight:600;padding:9px 16px;
  text-align:left;background:#F8FAFC;text-transform:uppercase;letter-spacing:.5px}
td{padding:11px 16px;border-top:1px solid var(--b1);font-size:.87rem}
tr:hover td{background:#F8FAFC}
.feat{font-weight:600;color:var(--txt)}
.fval{font-family:var(--mono);font-weight:700;font-size:.98rem;color:var(--txt)}
.funit{color:var(--muted);font-size:.75rem}
.ctag{display:inline-block;padding:4px 11px;border-radius:5px;
  font-size:.75rem;font-weight:700;font-family:var(--mono)}

/* chart */
.ccard{background:var(--panel);border:1px solid var(--b2);border-radius:13px;padding:16px 16px 12px}
.ctitle{font-family:var(--mono);font-size:.65rem;font-weight:700;
  color:var(--muted);letter-spacing:1.2px;text-transform:uppercase;margin-bottom:12px}
.chartbox{position:relative;height:155px}

/* loading */
.lbox{background:var(--panel);border:1px solid var(--b2);border-radius:14px;
  padding:48px 20px;display:flex;flex-direction:column;align-items:center;gap:14px;text-align:center}
.spin{width:44px;height:44px;border-radius:50%;border:3px solid var(--b2);
  animation:spin .85s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.lmsg{font-family:var(--mono);font-size:.78rem;color:var(--muted)}
.lsub{font-size:.73rem;color:#9CA3AF}

/* error */
.ebox{background:#FEF2F2;border:1px solid #FECACA;border-radius:9px;
  padding:11px 14px;font-size:.8rem;color:#B91C1C;font-family:var(--mono);margin-top:10px}

@media(max-width:700px){
  .grid{grid-template-columns:1fr}
  .mrow{grid-template-columns:1fr 1fr}
  h1{font-size:1.05rem}
}
</style>
</head>
<body>
<div class="wrap">

  <!-- header -->
  <div class="hdr">
    <div class="hdr-row">
      <div class="hdr-logo">🏗️</div>
      <h1>OPS-OPBC Mix Design Optimiser For Target Compressive Strength</h1>
    </div>
    <p>XGBoost surrogate model · inverse prediction · metaheuristic search</p>
    <div class="pills">
      <span class="pill pb">R² = {{ r2 }}</span>
      <span class="pill pg">RMSE = {{ rmse }} MPa</span>
      <span class="pill py">n = {{ n }} samples</span>
      <span class="pill pb">CS {{ cs_min }} – {{ cs_max }} MPa</span>
    </div>
  </div>

  <div class="grid">

    <!-- control panel -->
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
        <span>{{ cs_min }}</span>
        <span>{{ cs_mid }}</span>
        <span>{{ cs_max }}</span>
      </div>
      <div class="range-hint">Dataset range: {{ cs_min }} – {{ cs_max }} MPa</div>

      <label class="lbl">Optimisation algorithm</label>
      <div class="algo-grid">
        <button class="abtn" id="bGA"  onclick="pick('GA')">
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

      <button class="run" id="runBtn" onclick="go()" disabled>
        Select an algorithm
      </button>

      <div class="sbar">
        <div class="sdot" id="sdot"></div>
        <span id="stxt">Awaiting configuration</span>
      </div>
      <div class="ebox" id="errBox" style="display:none"></div>
    </div>

    <!-- results -->
    <div id="rArea">
      <div class="ph">
        <div class="ph-ico">🔬</div>
        <div class="ph-txt">
          Enter a target CS, choose an algorithm,<br>and click <strong>Run Optimisation</strong>.<br><br>
          The XGBoost surrogate will find the optimal
          Superplasticiser · OPS · OPBC · Age · Curing Method
          combination without running a single experiment.
        </div>
      </div>
    </div>

  </div>
</div>

<script>
let algo=null, chart=null;
const COL={GA:'#F59E0B',MFO:'#10B981',Hybrid:'#A78BFA'};
const CTAG={
  FW:'#3B82F6',AC:'#EF4444','2W':'#10B981','5W':'#6366F1',
  '7W':'#8B5CF6','2T2D':'#F59E0B','2T6D':'#EC4899',PS:'#14B8A6'
};

function pick(a){
  algo=a;
  ['GA','MFO','Hybrid'].forEach(x=>{
    document.getElementById('b'+x).className='abtn'+(x===a?' sel-'+x:'');
  });
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
  if(isNaN(target)||target<5||target>70){
    errEl.style.display='block';errEl.textContent='Target out of range.';return;
  }

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
    const res=await fetch('/opt',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({target,algo})
    });
    clearInterval(iv);
    const d=await res.json();
    if(d.server_error){showErr(d.server_error);return;}
    render(d,algo);
    setStatus('Done  ·  error = '+d.pred_error+' MPa','ready');
  }catch(e){
    clearInterval(iv);
    showErr('Network error: '+e.message);
  }
  document.getElementById('runBtn').disabled=false;
}

function showErr(msg){
  document.getElementById('rArea').innerHTML='';
  const eb=document.getElementById('errBox');
  eb.style.display='block'; eb.textContent='Error: '+msg;
  setStatus('Error');
  document.getElementById('runBtn').disabled=false;
}

function render(d,a){
  const col=COL[a];
  const ctcol=CTAG[d.cure_short]||col;
  const shortName=a;
  const errorValue = d.pred_error ?? d.error;

  document.getElementById('rArea').innerHTML=`
  <div style="display:flex;flex-direction:column;gap:12px">

    <!-- metrics -->
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
        <div class="mc-lbl">Error between predicted and target value</div>
        <div class="mc-val" style="color:${errorValue<0.5?'var(--good)':'var(--warn)'}">${errorValue}<span class="mc-unit">MPa</span></div>
      </div>
    </div>

    <!-- optimal mix design table -->
    <div class="rtable-wrap">
      <div class="rt-head">
        <div class="rt-dot" style="background:${col};box-shadow:0 0 7px ${col}"></div>
        Optimal mix design &nbsp;·&nbsp;
        <span style="color:${col}">${shortName}</span>
      </div>
      <table>
        <thead><tr>
          <th>Input parameter</th>
          <th>Optimal value</th>
          <th>Unit</th>
          <th>Dataset range</th>
        </tr></thead>
        <tbody>
          <tr>
            <td class="feat">Superplasticiser</td>
            <td class="fval">${d.superplasticiser}</td>
            <td class="funit">kg/m³</td>
            <td class="funit">{{ super_range }}</td>
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
            <td class="feat">Specimen age</td>
            <td class="fval">${d.age}</td>
            <td class="funit">days</td>
            <td class="funit">{{ age_range }}</td>
          </tr>
          <tr>
            <td class="feat">Curing method</td>
            <td colspan="3">
              <span class="ctag" style="background:${ctcol}1A;color:${ctcol};border:1px solid ${ctcol}40">
                ${d.cure_name}
              </span>
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <!-- convergence chart -->
    <div class="ccard">
      <div class="ctitle">Convergence — |predicted − target| per iteration</div>
      <div class="chartbox"><canvas id="cv"></canvas></div>
    </div>

  </div>`;

  // Chart
  if(chart) chart.destroy();
  chart=new Chart(document.getElementById('cv').getContext('2d'),{
    type:'line',
    data:{
      labels:d.curve.map((_,i)=>i+1),
      datasets:[{
        label:a,data:d.curve,
        borderColor:col,backgroundColor:col+'1A',
        borderWidth:2,pointRadius:0,fill:true,tension:0.4
      }]
    },
    options:{
      animation:{duration:450},responsive:true,maintainAspectRatio:false,
      scales:{
        x:{ticks:{color:'#6B7280',maxTicksLimit:8,font:{family:'IBM Plex Mono',size:9}},
           grid:{color:'#E5E7EB'},
           title:{display:true,text:'Iteration / Generation',color:'#6B7280',font:{size:9}}},
        y:{ticks:{color:'#6B7280',font:{family:'IBM Plex Mono',size:9}},
           grid:{color:'#E5E7EB'},
           title:{display:true,text:'|Error| (MPa)',color:'#6B7280',font:{size:9}}}
      },
      plugins:{
        legend:{labels:{color:'#6B7280',font:{family:'IBM Plex Mono',size:9}}},
        tooltip:{callbacks:{label:c=>' Error: '+c.parsed.y.toFixed(4)+' MPa'}}
      }
    }
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
    return render_template_string(PAGE,
        r2   = r2,   rmse = rmse,
        n    = len(df),
        cs_min = round(float(y.min()), 1),
        cs_max = round(float(y.max()), 1),
        cs_mid = round(float((y.min()+y.max())/2), 1),
        super_range = f"{B['super'][0]:.2f} – {B['super'][1]:.2f}",
        ops_range   = f"{B['ops'][0]:.0f} – {B['ops'][1]:.0f}",
        opbc_range  = f"{B['opbc'][0]:.0f} – {B['opbc'][1]:.0f}",
        age_range   = f"{B['age'][0]:.0f} – {B['age'][1]:.0f}",
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
        else: return jsonify({'server_error': 'Unknown algorithm: '+algo})
        return jsonify(r)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'server_error': str(e)})

# ── Start ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('\n' + '='*52)
    print('  OPS/OPBC Mix Design Optimiser')
    print('='*52)
    print(f'  Samples  : {len(df)}')
    print(f'  CS range : {y.min():.2f} – {y.max():.2f} MPa')
    print('='*52)
    print('  → Open:  http://127.0.0.1:5000')
    print('  → Stop:  Ctrl+C')
    print('='*52 + '\n')
    app.run(debug=False, port=5000)
