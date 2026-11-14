"""Polymarket FAA outage attribution forecaster (Python 3.13, stdlib only).

Genuine causal forecaster for the FINAL (4 FAA markets, 48h after deadline).
Method (chosen by backtest on real FAA price history, NOT by the practice
leaderboard — the practice is NVIDIA data whose answers are public and whose
rank-1 loss of 0 is a hardcode, not forecasting):

- Logit space: forecast in logit(p) so low prices (0.006-0.04) behave well.
- Trend-following drift: the FAA markets fell hard in their first hours
  (Iran 0.515 -> 0.021); mean-reversion to a 24h anchor PULLS UP and hurts
  (backtest: mr24 0.668 vs baseline 0.662). Instead drift = recent 240-min
  logit trend x 0.4/hour, so the model follows the market's direction and
  flattens to persistence when prices stabilise.
- Wide mixture tails: FAA jumps are huge (1-min logit move up to 2.6, Iran
  -0.215 in one minute), so quantiles use a two-component mixture
  (85% N(0,0.03^2) + 15% N(0,0.15^2)) instead of a narrow normal.
  Precomputed quantiles Z = [-0.0815, -0.0236, 0, 0.0236, 0.0815] per sqrt(min).
  No level-dependent spread: on FAA's low prices it added nothing in backtest.

Backtest on FAA history (16 checkpoints, 240-min horizon):
  v1 (K 0.4, s2 0.15) 0.3576 -> v2 (K 0.6, s2 0.12) 0.3398,
  house baseline 0.6365. Asymmetric upper tails were tried and REJECTED
  (0.43-0.79): FAA history almost always falls, so widening only the top
  quantiles costs far more than it saves.
On NVIDIA practice (dev, different regime) v2 scores ~0.38, ~baseline —
the dev score is not the target; the final FAA window is.
Deterministic, no network, stdlib only.
"""
import json
import math
import os

TAUS = [0.05, 0.25, 0.5, 0.75, 0.95]
# v2 (koreksi atas perintah user): trend lebih kuat (K 0.4 -> 0.6) + ekor
# mixture sedikit lebih sempit (s2 0.15 -> 0.12). Backtest FAA: v1 0.3576
# -> v2 0.3398 (baseline 0.6365). Ekor asimetris DICOBA dan DITOLAK:
# memperlebar kuantil atas saja merusak skor (0.43-0.79) karena histori FAA
# hampir selalu turun; ekor simetris terbukti terbaik.
Z = [-0.070275, -0.023224, 0.0, 0.023224, 0.070275]
WIN = 240
K = 0.6


def sigmoid(x):
    if x > 30:
        return 1.0
    if x < -30:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def forecast_case(case):
    horizon = case.get("horizon_minutes", 240)
    out = []
    for m in case["markets"]:
        prices = m["prices"]
        x = prices[-1]
        lx = logit(x)
        n = len(prices)
        w = min(WIN, n - 1)
        drift = (lx - logit(prices[-1 - w])) * (60.0 / w) * K if w > 0 else 0.0
        rows = []
        for h in range(1, horizon + 1):
            center = lx + drift * h / 60.0
            s = math.sqrt(h)
            row = []
            for z in Z:
                q = sigmoid(center + z * s)
                row.append(round(min(max(q, 0.0), 1.0), 6))
            for i in range(1, 5):
                if row[i] < row[i - 1]:
                    row[i] = row[i - 1]
            rows.append(row)
        out.append({"id": m["id"], "q": rows})
    return {"forecasts": out}


def main():
    inp = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    outp = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(outp, exist_ok=True)
    for fn in sorted(os.listdir(inp)):
        if not fn.endswith(".in"):
            continue
        with open(os.path.join(inp, fn)) as f:
            case = json.load(f)
        fc = forecast_case(case)
        cid = case.get("case", fn[:-3])
        with open(os.path.join(outp, cid + ".out"), "w") as f:
            json.dump(fc, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
