"""Small-instance comparison: full MILP (two-stage lexicographic optimum) vs rule-based plan vs LNS
on the first replanning window.  python3 small_opt.py <daily> <cond> <seed> [budget] [milp_limit] [outdir]"""
import sys, json, time, os
import numpy as np
from instance import generate
from simulate import Rolling
from model import WindowModel
from run_main import resources, HOLD_COST_DAY, LNS_SEED


def main(daily, cond, seed, budget=120.0, tl=600.0, outdir='results_small'):
    caps, park_in, park_out, rent_max = resources(daily, cond)
    def inst():
        I = generate(daily, seed, park_in=park_in, park_out=park_out, rent_max=rent_max, caps=caps)
        I.hold_cost_day = HOLD_COST_DAY
        return I
    out = dict(daily=daily, cond=cond, seed=seed, budget=budget, milp_limit=tl)
    R = Rolling(inst(), policy='lns', m=1, budget=budget, seed=LNS_SEED, verbose=False)
    ts = R.replan_times(); t0, tn = ts[0], ts[1]
    rec = R.replan(t0, tn)
    out.update(n_veh=rec['n_veh'], ncol=rec['stats']['ncol'], nrow=rec['stats'].get('nrow'),
               rule_F=rec['init_F'], rule_Z=rec['init_obj'], lns_F=rec['final_F'], lns_Z=rec['final_obj'],
               lns_iter=rec['lns_iter'], lns_s=rec['total_s'])
    # full model, same window and same bounds
    R2 = Rolling(inst(), policy='init', m=1, budget=budget, seed=LNS_SEED, verbose=False)
    t1 = R2.window_end(t0)
    W = R2.window_state(t0, t1)
    M = WindowModel(W, m=1, elastic=True)
    lo, hi = M.lo.copy(), M.hi.copy()
    M.to_highs()
    M.set_stage(1)
    t = time.time()
    r1 = M.solve(time_limit=tl, lo=lo, hi=hi)
    out.update(s1_status=r1['status'], s1_obj=r1['obj'], s1_bound=r1['bound'], s1_s=time.time() - t)
    if r1['x'] is not None:
        Fs = float(M.F(r1['x']))
        M.set_stage(2, Fs)
        t = time.time()
        r2 = M.solve(time_limit=tl, start=r1['x'], lo=lo, hi=hi)
        out.update(s2_status=r2['status'], s2_obj=r2['obj'], s2_bound=r2['bound'], s2_gap=r2['gap'], s2_s=time.time() - t,
                   milp_F=Fs, milp_Z=None if r2['x'] is None else float(M.Z(M.normalize(r2['x']))))
    os.makedirs(outdir, exist_ok=True)
    json.dump(out, open(f'{outdir}/small_d{daily}_{cond}_s{seed}.json', 'w'), indent=1, default=float)
    print(json.dumps(out, default=float))


if __name__ == '__main__':
    a = sys.argv
    main(int(a[1]), a[2], int(a[3]), float(a[4]) if len(a) > 4 else 120.0, float(a[5]) if len(a) > 5 else 600.0,
         a[6] if len(a) > 6 else 'results_small')
