"""Equation check for the final Chapter 3 (eqs [1]-[16]) and the two-stage lexicographic search.

For every small case in verification_cases.fixtures() and block length m = 1, 2, 3:
  - the objective coefficients of model.py equal those rebuilt from the chapter (PaperModel);
  - stage 1 (min F) gives the same optimal excess waiting F* in both models;
  - stage 2 (min Z subject to F <= F*) gives the same optimal cost Z*;
  - each optimal solution is feasible in the other model.
Run:  python3 verify_ch3_final.py   ->  validation/ch3_final_check.json
"""
import copy, json, time
from pathlib import Path
import numpy as np
import verification_cases as q
from paper_model_check import PaperModel, code_violation

EPS = 1e-6
OUT = Path(__file__).resolve().parent / 'validation'
OUT.mkdir(exist_ok=True)


def compare(label, W, m):
    M = q.WindowModel(W, m=m)
    P = PaperModel(M)
    assert np.allclose(M.cost, P.c, rtol=0, atol=EPS), (label, 'cost coefficients differ')
    M.to_highs()
    M.set_stage(1)
    c1, p1 = M.solve(time_limit=10), P.solve(time_limit=10, stage=1)
    assert c1['status'] == p1['status'] == 'Optimal', (label, m, c1['status'], p1['status'])
    assert abs(c1['obj'] - p1['obj']) < EPS, (label, m, 'F*', c1['obj'], p1['obj'])
    Fstar = float(M.F(c1['x']))
    assert abs(Fstar - c1['obj']) < EPS
    M.set_stage(2, Fstar)
    c2, p2 = M.solve(time_limit=10), P.solve(time_limit=10, stage=2, fcap=Fstar + 1e-6)
    assert c2['status'] == p2['status'] == 'Optimal', (label, m, c2['status'], p2['status'])
    assert abs(c2['obj'] - p2['obj']) < 1e-4, (label, m, 'Z*', c2['obj'], p2['obj'])
    for name, x in (('code', c2['x']), ('reference', p2['x'])):
        pv, _ = P.violation(x)
        assert pv < 1e-5 and code_violation(M, x) < 1e-5, (label, m, name, pv, code_violation(M, x))
    return dict(label=label, m=m, F_star=Fstar, Z_star=c2['obj'], code_rows=M.nrow, reference_rows=len(P.rows))


def extra_cases():
    """Cases for the features added on 2026-11-16."""
    cases = []
    # (a) internal work interrupted at 18:00 and resumed at 09:00 (p_ijt), window 17:00 -> next day 11:00
    v = q.vd(0, tasks=['maint', 'insp'], modes=[q.md(0, p=14), q.md(1, p=1, kind='Q')], hold=204, due=288 + 112)
    cases.append(('overnight_interruption', q.window([v], t0=204, t1=288 + 132)))
    # (b) a recovery trip that needs a technician: one general staff member, no towing supply
    v = q.vd(0, tasks=['insp'], loc='OUT', due=130)
    W = q.window([v], gen=1)
    W.inst.caps['sup_tow'] = 0
    cases.append(('technician_joins_trip', W))
    # (c) a new non-final alternative that ends after the planning horizon (last term of eq. [15])
    v = q.vd(0, tasks=['maint', 'insp'], modes=[q.md(0, p=30), q.md(1, p=1, kind='Q')], due=118)
    cases.append(('alternative_crossing_horizon', q.window([v])))
    return cases


def main():
    t = time.time()
    rows = []
    for label, W, _ in q.fixtures():
        for m in (1, 2, 3):
            rows.append(compare(label, copy.deepcopy(W), m))
    for label, W in extra_cases():
        for m in (1, 2, 3):
            rows.append(compare(label, copy.deepcopy(W), m))
    # the added cases must really exercise the new features
    M = q.WindowModel(extra_cases()[0][1]); M.to_highs(); M.set_stage(2, 0.0); x = M.solve(time_limit=10)['x']
    assert any(M.xp[j] > M.W.vehs[M.xv[j]]['modes'][M.xb[j]].p for j in np.nonzero(x[M.x0:M.x0 + M.xv.size] > 0.5)[0]), 'no interrupted task chosen'
    M = q.WindowModel(extra_cases()[1][1]); M.to_highs(); M.set_stage(2, 0.0); x = M.solve(time_limit=10)['x']
    assert M.decode(x)['crew'], 'no technician in the recovery trip'
    res = dict(pairs=len(rows), with_excess=sum(r['F_star'] > 0 for r in rows), seconds=round(time.time() - t, 2), rows=rows)
    (OUT / 'ch3_final_check.json').write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print('equation check PASS: %d pairs (%d with F* > 0), %.1fs' % (res['pairs'], res['with_excess'], res['seconds']))


if __name__ == '__main__':
    main()
