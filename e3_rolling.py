"""E3: daily-inflow operating pilot (prompt 9.3).  14 calendar days, re-planning every business day
08:00 and at fault notices / decisions, 4-day look-ahead window, 120 s total per re-plan.

  python3 e3_rolling.py one DAILY SEED POLICY M SCALE   -> results/e3/<name>.json
  python3 e3_rolling.py all                             -> runs the fixed pilot plan
Resource scale: SCALE multiplies every staff / equipment / supply count and both parking areas of
the v5 synthetic base (15 staff, IN 12, OUT 18).  SCALE is an experiment assumption.
"""
import sys, os, json, time, subprocess, resource
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results', 'e3')

PLAN = [  # (daily, seed, policy, m, scale)
    (10, 1, 'init', 1, 1), (10, 1, 'lns', 1, 1), (10, 1, 'lns', 2, 1), (10, 1, 'lns', 3, 1),
    (40, 1, 'init', 1, 4), (40, 1, 'lns', 1, 4), (40, 1, 'lns', 3, 4),
]


def name(d, s, p, m, sc):
    return f'd{d}_s{s}_{p}_m{m}_x{sc}'


def conv(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, np.ndarray): return o.tolist()
    return str(o)


def one(daily, seed, policy, m, scale):
    from instance import generate, DEFAULT_CAPS
    from simulate import Rolling
    caps = {k: v * scale for k, v in DEFAULT_CAPS.items()}
    inst = generate(daily, seed, park_in=12 * scale, park_out=18 * scale, rent_max=15 * scale, caps=caps)
    t = time.time()
    R = Rolling(inst, policy=policy, m=m, budget=120.0, seed=11, verbose=True)
    chk, ev = R.run()
    log = []
    for r in R.log:
        r = dict(r); r.pop('history', None)
        log.append(r)
    res = dict(name=name(daily, seed, policy, m, scale), daily=daily, seed=seed, policy=policy, m=m,
               scale=scale, n_vehicles=len(inst.vehicles), meta=inst.meta,
               n_new=sum(v.cohort == 'new' for v in inst.vehicles),
               n_init=sum(v.cohort == 'init' for v in inst.vehicles),
               business_days=sum(1 for d in range(14) if d % 7 < 5),
               checker_ok=chk['ok'], n_viol=chk['n_viol'], violations=chk['violations'][:10],
               staff_persons=chk['staff_persons'], eval=ev, replans=len(R.log),
               replans_valid=sum(r['valid'] for r in R.log),
               replans_over_budget=sum(r['total_s'] > 121 for r in R.log),
               cum_plan_s=sum(r['total_s'] for r in R.log), wall_s=time.time() - t,
               maxrss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, log=log,
               trajectory=dict(modes=[(vid, off, md.label, md.f, md.g, s) for (vid, off, md, s) in R.traj['modes']],
                               trips=R.traj['trips'], exits=R.traj['exits'],
                               rent={str(k): v for k, v in R.traj['rent'].items()}))
    return res


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    if sys.argv[1] == 'one':
        d, s, p, m, sc = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], int(sys.argv[5]), int(sys.argv[6])
        r = one(d, s, p, m, sc)
        json.dump(r, open(os.path.join(OUT, r['name'] + '.json'), 'w'), default=conv, indent=1)
        print('DONE', r['name'], r['checker_ok'], r['eval']['econ'])
    else:
        sel = PLAN if len(sys.argv) < 3 else [PLAN[int(i)] for i in sys.argv[2].split(',')]
        for (d, s, p, m, sc) in sel:
            fn = os.path.join(OUT, name(d, s, p, m, sc) + '.json')
            if os.path.exists(fn):
                continue
            with open(os.path.join(OUT, name(d, s, p, m, sc) + '.log'), 'w') as lf:
                subprocess.run([sys.executable, os.path.abspath(__file__), 'one', str(d), str(s), p, str(m), str(sc)],
                               stdout=lf, stderr=subprocess.STDOUT, cwd=HERE)
