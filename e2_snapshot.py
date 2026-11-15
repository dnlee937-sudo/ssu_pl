"""E2: controlled comparison of parking representations (5 / 10 / 15 min, 30 min stats only) on
v5-like snapshot inputs (n vehicles over 14 calendar days, all known at d0 08:00).
Same input / construction / LNS / seed / 1 thread / total budget; only the parking block length m
changes.  Every returned plan is re-checked by the independent 5-minute checker.

  python3 e2_snapshot.py one N SEED M BUDGET      -> prints one JSON line
  python3 e2_snapshot.py all                      -> results/e2_snapshot.jsonl (subprocess per run)
"""
import sys, json, time, os, resource, subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results')


def one(N, seed, m, budget, stats_only=False, scale=1.0):
    from instance import generate_snapshot, SLOTS_PER_DAY, REPLAN_SLOT, pickup_slot
    from simulate import Rolling
    from model import Window, WindowModel
    import checker
    t_start = time.perf_counter()
    from instance import DEFAULT_CAPS
    if scale != 1.0:
        caps = {k: max(1, int(round(v * scale))) for k, v in DEFAULT_CAPS.items()}
        inst = generate_snapshot(N, seed, park_in=int(round(12 * scale)), park_out=int(round(18 * scale)),
                                 rent_max=int(round(12 * scale)), caps=caps)
    else:
        inst = generate_snapshot(N, seed)
    t0, t1 = REPLAN_SLOT, inst.horizon
    vehs = [dict(vid=v.vid, entry_slot=v.arrival, entry_loc='IN', tasks=v.tasks, modes=v.modes,
                 hold=t0, deadline=v.deadline, pickup_req=None, offset=0) for v in inst.vehicles]
    W = Window(inst, t0, t1, vehs, {}, np.zeros(t1 - t0, int))
    if stats_only:
        M = WindowModel(W, m=m, elastic=True)
        st = M.stats()
        return dict(N=N, seed=seed, m=m, stats=st, maxrss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    R = Rolling(inst, policy='lns', m=m, budget=budget - (time.perf_counter() - t_start), seed=11)
    R.window = t1 - t0                      # plan check over the whole 14-day snapshot
    R.window_state = lambda a, b: W
    R.known_ids = lambda a: {v.vid for v in inst.vehicles}
    rec = R.replan(t0, t1)
    chk = checker.check(inst, R.traj, t_end=t1)
    ev = checker.evaluate(inst, R.traj, chk, t1)
    total = time.perf_counter() - t_start
    return dict(N=N, seed=seed, m=m, scale=scale, budget=budget, stats=rec['stats'], build_s=rec['build_s'],
                init_s=rec['init_s'], construct_overflow=rec['construct_overflow'],
                conflict_rounds=rec['conflict_rounds'], init_valid=rec['init_check'] and rec['sigma0'] <= 1e-6,
                valid=rec['valid'], init_obj=rec['init_obj'], final_obj=rec['final_obj'],
                lns_iter=rec['lns_iter'], lns_acc=rec['lns_acc'], history=rec['history'],
                checker_5min_ok=chk['ok'], checker_viol=chk['violations'][:3],
                econ=ev['econ'], late_min=ev['late_min'], work_outsourced=ev['work_outsourced'],
                staff_trips=ev['n_staff_trips'], tow=ev['n_tow'], rent=ev['rent'],
                completed=ev['completed'], max_in=ev['max_in'],
                total_s=total, within_budget=total <= budget + 1.0,
                maxrss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)


def conv(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    return str(o)


if __name__ == '__main__':
    if sys.argv[1] == 'one':
        N, seed, m, budget = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
        so = len(sys.argv) > 6 and sys.argv[6] == 'stats'
        sc = float(sys.argv[6][5:]) if len(sys.argv) > 6 and sys.argv[6].startswith('scale') else 1.0
        try:
            r = one(N, seed, m, budget, stats_only=so, scale=sc)
        except Exception as e:  # keep failures in the record
            import traceback
            r = dict(N=N, seed=seed, m=m, error=repr(e), tb=traceback.format_exc()[-800:])
        print('RESULT ' + json.dumps(r, default=conv))
    else:
        os.makedirs(OUT, exist_ok=True)
        fn = os.path.join(OUT, 'e2_snapshot.jsonl')
        done = set()
        if os.path.exists(fn):
            for line in open(fn):
                d = json.loads(line); done.add((d['N'], d['seed'], d['m'], d.get('kind', 'run')))
        plan = []
        for N in (20, 40, 60, 80, 100):
            for m in (1, 2, 3, 6):
                plan.append((N, 1, m, 'stats'))
        for N in (20, 40, 60, 80, 100):
            for m in (1, 2, 3):
                plan.append((N, 1, m, 'run'))
        for N in (40, 60):
            for m in (1, 2, 3):
                plan.append((N, 2, m, 'run'))
        for N in (60, 80, 100):
            for m in (1, 2, 3):
                plan.append((N, 1, m, 'runS'))
        for (N, seed, m, kind) in plan:
            if (N, seed, m, kind) in done:
                continue
            args = [sys.executable, os.path.abspath(__file__), 'one', str(N), str(seed), str(m), '120']
            if kind == 'stats':
                args.append('stats')
            if kind == 'runS':
                args.append(f'scale{N / 40}')
            t = time.time()
            p = subprocess.run(args, capture_output=True, text=True, timeout=900, cwd=HERE)
            line = [l for l in p.stdout.splitlines() if l.startswith('RESULT ')]
            d = json.loads(line[0][7:]) if line else dict(N=N, seed=seed, m=m, error='no result',
                                                          stderr=p.stderr[-800:], rc=p.returncode)
            d['kind'] = kind; d['wall_s'] = time.time() - t
            with open(fn, 'a') as f:
                f.write(json.dumps(d, default=conv) + '\n')
            print(kind, N, seed, m, 'valid', d.get('valid'), 'chk', d.get('checker_5min_ok'),
                  'obj', d.get('final_obj'), 'cols', d.get('stats', {}).get('ncol'), 'rss', d.get('maxrss_mb'),
                  'wall', round(d['wall_s'], 1), d.get('error', ''), flush=True)
