# PSG

Path-Stability Generalization

Path-Stability Generalization (PSG) — toy experiments on two-moons.

## Setup

- Create env and install deps: `pip install -r requirements.txt`

## Experiments

- Exp-A (Correlation sweep): `python exp_a.py`
- Exp-B (Intervention via path regularizer): `python exp_b.py`
- Exp-C (Stress tests): `python exp_c.py`
- Exp-D (One-shot path vs gap, conditions A/B/C): `python exp_d.py`

Outputs are written under `runs/exp_{a,b,c,d}_<timestamp>/`.
