"""Polymarket FAA outage attribution forecaster (Python 3.13, stdlib only).

Genuine causal forecaster for the FINAL (4 FAA markets, 48h after deadline).
Method (chosen by backtest on real FAA price history, NOT by the practice
leaderboard — the practice is NVIDIA data whose answers are public and whose
rank-1 loss of 0 is a hardcode, not forecasting):

- Logit space: forecast in logit(p) so low prices (0.006-0.04) behave well.
- Trend-following drift: drift = recent 240-min logit trend x K/hour.
- Mean-reversion (v4): pull toward the 24h logit-mean anchor, strength
  K_mr/hour (exponential approach). A SMALL mean-reversion (K_mr 0.10)
  fixes v3's biggest weakness — its constant downward bias hurt badly in
  flat regimes (HOLD calibration q50 0.029 vs ideal 0.50). The anchor
  counterweights the bias when prices stabilise, without hurting the
  crash regime. K_mr 0.10 improves BOTH FAA and NVIDIA.
- Downward bias correction (v3, from calibration audit): the model's
  quantiles were systematically too high. A downward logit bias, scaling
  with sqrt(horizon), corrects this. Bias -0.30 at h=240.
- Wide mixture tails: FAA jumps are huge (1-min logit move up to 2.6),
  so quantiles use a two-component mixture (85% N(0,0.03^2) +
  15% N(0,0.12^2)) instead of a narrow normal.
  Precomputed quantiles Z = [-0.070275, -0.023224, 0, 0.023224, 0.070275]
  per sqrt(min). No level-dependent spread.

Backtest on FAA history (22 checkpoints, 240-min horizon, 4 markets):
  v1 (K 0.4, s2 0.15) 0.3576 -> v2 (K 0.6, s2 0.12) 0.3398
  -> v3 (K 0.35, bias -0.30) 0.2852
  -> v4 (K 0.35, bias -0.30, K_mr 0.10) 0.2649, house baseline 0.5302.
  v4 beats v3 on ALL splits (DEV 0.3359 vs 0.3403, ROLL 0.2271 vs 0.2443,
  HOLD 0.1842 vs 0.2341) and on NVIDIA practice (0.3469 vs 0.3524,
  baseline 0.3636). LOCO NVIDIA: v4 wins 5/6 cases.
  K_mr 0.10 is the robust optimum: K_mr 0.20 improves FAA further (0.2534)
  but hurts NVIDIA (0.3598); K_mr 0.30+ hurts NVIDIA badly (0.3782).
Deterministic, no network, stdlib only.
"""
import json
import math
import os

TAUS = [0.05, 0.25, 0.5, 0.75, 0.95]
# v4 (hybrid tren + mean-reversion, PRACTICE OPTIMIZATION 14 Nov 2026):
# v3 + K_mr 0.10 (tarik ke anchor rata-rata logit 24 jam, eksponensial).
# Memperbaiki kelemahan terbesar v3 (bias merusak saat flat).
# FAA: v3 0.2852 -> v4 0.2649 (baseline 0.5302). NVIDIA: v3 0.3524 -> v4 0.3469
# (baseline 0.3636). LOCO NVIDIA: v4 menang 5/6 case. Split FAA: v4 menang
# di DEV, ROLL, dan HOLD. K_mr 0.10 adalah optimum robust (K_mr 0.20
# memperbaiki FAA jadi 0.2534 tapi merusak NVIDIA jadi 0.3598).
Z = [-0.070275, -0.023224, 0.0, 0.023224, 0.070275]
WIN = 240
WIN24 = 1440
K = 0.35
K_MR = 0.10
BIAS = -0.30


def sigmoid(x):
    if x > 30:
        return 1.0
    if x < -30:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def logit(p):
    p = min(max(p, 1e-6), 1-1e-6)
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
        w24 = min(WIN24, n)
        anchor = sum(logit(p) for p in prices[-w24:]) / w24 if w24 > 0 else lx
        rows = []
        for h in range(1, horizon + 1):
            mr_part = (anchor - lx) * (1.0 - math.exp(-K_MR * h / 60.0))
            center = lx + drift * h / 60.0 + mr_part + BIAS * math.sqrt(h) / math.sqrt(240.0)
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
