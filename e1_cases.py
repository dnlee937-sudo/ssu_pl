"""E1: small hand-built verification cases (prompt 9.1) + checker mutation tests.
Run:  python3 e1_cases.py  -> results/e1_cases.json
"""
import json, copy, sys, os
import numpy as np
from instance import (Instance, Vehicle, build_modes, make_trip_types, DEFAULT_CAPS, STAFF,
                      SLOTS_PER_DAY, OPEN_SLOT, INF, pickup_slot, collect_slot, SLOT_MIN)
from simulate import Rolling
from model import Window, WindowModel
import checker
from construct import construct

OUT = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(OUT, exist_ok=True)


def hm(day, h, mi=0):
    return day * SLOTS_PER_DAY + (h * 60 + mi) // SLOT_MIN


def make(vehicles, days=7, caps=None, park_in=12, park_out=18, rent_max=0):
    c = dict(DEFAULT_CAPS)
    c.update(caps or {})
    return Instance(days * SLOTS_PER_DAY, vehicles, c, set(STAFF), dict(IN=park_in, OUT=park_out),
                    rent_max, 5000.0, make_trip_types(), name='case')


def veh(vid, arrival, tasks, deadline, **kw):
    return Vehicle(vid, arrival, kw.pop('init_loc', None), tasks, build_modes(tasks) if tasks else [],
                   deadline, **kw)


def run_roll(inst, days, budget=20, m=1, policy='lns'):
    R = Rolling(inst, policy=policy, m=m, budget=budget, t_obs=days * SLOTS_PER_DAY)
    chk, ev = R.run()
    return R, chk, ev


def fmt_events(R, vid):
    out = []
    for e in sorted(R.by_v[vid], key=lambda e: e[1]):
        t = e[1]
        stamp = f'd{t // SLOTS_PER_DAY} {(t % SLOTS_PER_DAY) * 5 // 60:02d}:{(t % SLOTS_PER_DAY) * 5 % 60:02d}'
        if e[0] == 'mode':
            out.append(f"{stamp} start {e[3].label} ({e[3].p * 5} min)")
        elif e[0] == 'trip':
            out.append(f"{stamp} trip {e[2].src}->{e[2].dst} {e[2].mode} q={e[2].q}")
        else:
            out.append(f"{stamp} release")
    return out


results = {}

# ---------------------------------------------------------------- C1 normal flow with outsourcing
v = veh(0, hm(0, 9), ['maint', 'wash', 'insp'], hm(2, 18), response_delay=12)
inst = make([v], caps=dict(washbay=0))          # internal wash impossible -> outsourced wash
R, chk, ev = run_roll(inst, 4)
r = chk['vrec'][0]
results['C1_normal_outsourced'] = dict(
    check_ok=chk['ok'], violations=chk['violations'], events=fmt_events(R, 0),
    completion=r['completion'], pickup_booking=pickup_slot(v, r['completion']) if r['completion'] else None,
    release=r['exit'],
    assert_outsourced_before_inspection=any('wash:O' in e for e in fmt_events(R, 0)),
    assert_release_after_booking=(r['exit'] is not None and r['exit'] >= pickup_slot(v, r['completion'])))

# ---------------------------------------------------------------- C2 customer approval hold
v = veh(0, hm(0, 9), ['maint', 'insp'], hm(1, 18), approval_at=hm(1, 14), response_delay=12)
inst = make([v])
R, chk, ev = run_roll(inst, 4)
first = min(e[1] for e in R.by_v[0] if e[0] == 'mode')
results['C2_approval_hold'] = dict(check_ok=chk['ok'], events=fmt_events(R, 0),
                                   first_start=first, approval=hm(1, 14),
                                   assert_no_start_before_approval=first >= hm(1, 14),
                                   lateness_min=ev['late_min'],
                                   note='deadline extended by the customer-side wait (approval - arrival)')

# ---------------------------------------------------------------- C3 completed, customer never replies
v = veh(0, None, [], 0, init_loc='IN', response_delay=None, done=2, cohort='init')
inst = make([v])
R, chk, ev = run_roll(inst, 7)
occ_nights = [int(chk['occ_in'][hm(d, 2)] + chk['occ_out'][hm(d, 2)]) for d in range(7)]
results['C3_no_reply_storage'] = dict(check_ok=chk['ok'], events=fmt_events(R, 0),
                                      occupied_at_02h_each_day=occ_nights,
                                      released=chk['vrec'][0]['exit'],
                                      storage_billed_won=ev['storage_billed'],
                                      assert_occupies_nights_and_weekend=all(o == 1 for o in occ_nights))

# ---------------------------------------------------------------- C4 shop-fault event -> settle, later return
v = veh(0, hm(0, 9), ['body_h', 'paint_h', 'wash', 'insp'], hm(6, 18), response_delay=12,
        fault=dict(notice=hm(0, 12), decision=hm(1, 10), outcome='return', extend_slots=0,
                   collect_delay=SLOTS_PER_DAY))
inst = make([v], days=7)
R, chk, ev = run_roll(inst, 7)
r = chk['vrec'][0]
results['C4_fault_return'] = dict(check_ok=chk['ok'], events=fmt_events(R, 0),
                                  settlement=hm(1, 10), collect=collect_slot(v), release=r['exit'],
                                  tasks_done=r['next_task'], tasks_total=r['ntask'],
                                  status=ev['status'],
                                  assert_release_after_settlement=r['exit'] is not None and r['exit'] >= collect_slot(v),
                                  assert_no_work_started_after_notice=all(e[1] < hm(0, 12) for e in R.by_v[0] if e[0] == 'mode'))

# ---------------------------------------------------------------- C5 end of observation with open work
v1 = veh(0, hm(0, 9), ['engine', 'wash', 'insp'], hm(0, 18), response_delay=12)
v2 = veh(1, hm(0, 9), ['engine', 'insp'], hm(0, 18), response_delay=12)
v3 = veh(2, hm(0, 9), ['engine', 'insp'], hm(0, 18), response_delay=12)
inst = make([v1, v2, v3], caps=dict(mech=1, lift=1, sup_engine=0), days=3)
R, chk, ev = run_roll(inst, 1)
results['C5_open_at_end'] = dict(check_ok=chk['ok'], events={i: fmt_events(R, i) for i in range(3)},
                                 status=ev['status'], residual_work_lb=ev['residual_work_lb'],
                                 residual_late_lb=ev['residual_late_lb'],
                                 note='no forced completion; remaining obligation valued separately')

# ---------------------------------------------------------------- C6 block-boundary cases (static window)
def static_window(inst, t0, t1):
    vehs = []
    for v in inst.vehicles:
        es = None if v.arrival is None else v.arrival
        vehs.append(dict(vid=v.vid, entry_slot=es if es is None or es > t0 else None,
                         entry_loc=v.init_loc or 'IN', tasks=v.tasks, modes=v.modes,
                         hold=v.approval_at or t0, deadline=v.deadline,
                         pickup_req=pickup_slot(v, 0) if not v.tasks else None, offset=0))
    return Window(inst, t0, t1, vehs, {}, np.zeros(t1 - t0, int))


# (a) car A (completed, booking d1 09:00) leaves at 09:00, car B arrives 09:00 / 09:05 ; IN capacity 1
case6 = {}
for arr_min in (0, 5, 10, 15):
    a = veh(0, None, [], 0, init_loc='IN', response_delay=0, done=1)
    b = veh(1, hm(1, 9, arr_min), ['wash', 'insp'], hm(3, 18), response_delay=12)
    inst = make([a, b], park_in=1, park_out=0, rent_max=0, days=3, caps=dict(supportcar=0, sup_tow=0))
    # booking of A: pickup_slot(a,0) = next business day 10:00 + offset -> override for the test
    a.response_delay = 0
    W = static_window(inst, hm(1, 8), hm(3, 0))
    W.vehs[0]['pickup_req'] = hm(1, 9)
    row = {}
    for m in (1, 2, 3):
        M = WindowModel(W, m=m)
        res = M.solve(time_limit=30)
        row[f'{5 * m}min'] = res['status'] + ('' if res['x'] is None else f" obj={res['obj']:.0f}")
    case6[f'B_arrives_09:{arr_min:02d}'] = row
results['C6a_departure_then_arrival_capacity1'] = dict(
    table=case6, note='5-min model admits A leaving at 09:00 and B entering at 09:00/09:05; '
                      'the conservative 10/15-min blocks reject B arriving inside the block where A leaves')

# (b) a short overflow strictly inside a block: sampling block ends misses it, checker and models catch it
a = veh(0, None, ['wash', 'insp'], hm(2, 18), init_loc='IN', response_delay=12)
b = veh(1, hm(0, 9, 5), ['insp'], hm(2, 18), response_delay=12)
inst = make([a, b], park_in=1, park_out=0, days=3, caps=dict(supportcar=0, sup_tow=0))
md_b = b.modes[0]
traj = dict(modes=[(1, 0, md_b, hm(0, 9, 10))], trips=[], exits=[], rent={})   # B waits 09:05-09:10
chk = checker.check(inst, traj, t_end=hm(0, 9, 15), include={0, 1})
occ = chk['occ_in'][hm(0, 9):hm(0, 9, 15)]
results['C6b_short_overflow_inside_block'] = dict(
    occupancy_09_00_to_09_15=occ.tolist(), checker_ok=chk['ok'], checker_violations=chk['violations'],
    block_end_sampling_15min=int(chk['occ_in'][hm(0, 9, 15) - 1]),
    note='sampling only the state at 15-min block ends would see 1 car (<= capacity) and miss the overflow')

# ---------------------------------------------------------------- C7 small direct MILP vs construction+LNS
rng = np.random.default_rng(7)
from instance import draw_tasks, add_business_days
vs = []
for i in range(6):
    tasks = draw_tasks(rng)
    arr = hm(0, 9 + int(rng.integers(0, 6)))
    vs.append(veh(i, arr, tasks, add_business_days(arr, 2), response_delay=12))
inst = make(vs, park_in=3, park_out=4, rent_max=2, days=3, caps=dict(gen=3, mech=1, body=1, insp=1))
W = static_window(inst, hm(0, 8), hm(3, 0))
out7 = {}
for m in (1, 2, 3):
    M = WindowModel(W, m=m, elastic=False)
    res = M.solve(time_limit=120)
    plan = M.decode(res['x']) if res['x'] is not None else None
    ok = None
    if plan:
        traj = dict(modes=[(vid, 0, W.vehs[vid]['modes'][b], s) for (vid, b, s) in plan['modes']],
                    trips=plan['trips'], exits=plan['exits'], rent=plan['rent'])
        ok = checker.check(inst, traj, t_end=hm(3, 0))['ok']
    out7[f'direct_MILP_{5 * m}min'] = dict(status=res['status'], obj=res['obj'], bound=res['bound'],
                                          gap=res['gap'], time_s=round(res['time'], 2), checker_ok=ok,
                                          stats=M.stats())
# construction + LNS (same window, 5-min)
R = Rolling(inst, policy='lns', m=1, budget=60)
R.known_ids = lambda t0: {v.vid for v in inst.vehicles}
R.window_state = lambda t0, t1: W
rec = R.replan(hm(0, 8), hm(3, 0))
chk = checker.check(inst, R.traj, t_end=hm(3, 0))
out7['construction_plus_LNS_5min'] = dict(init_obj=rec['init_obj'], final_obj=rec['final_obj'],
                                         lns_iter=rec['lns_iter'], lns_acc=rec['lns_acc'],
                                         checker_ok=chk['ok'], violations=chk['violations'][:3])
results['C7_direct_MILP_vs_heuristic'] = out7

# ---------------------------------------------------------------- checker mutation tests
v0 = veh(0, hm(0, 9), ['maint', 'wash', 'insp'], hm(2, 18), approval_at=hm(0, 11), response_delay=12)
v1 = veh(1, hm(0, 9), ['maint', 'insp'], hm(2, 18), response_delay=12)
v2 = veh(2, hm(0, 9), ['insp'], hm(2, 18), response_delay=12)
inst = make([v0, v1, v2], park_in=2, park_out=5, days=3, caps=dict(mech=1, lift=1))
L = lambda vv, lab: [m for m in vv.modes if m.label == lab][0]
base = dict(modes=[(0, 0, L(v0, 'maint:I'), hm(0, 11)), (0, 0, L(v0, 'wash:I'), hm(0, 13)),
                   (0, 0, L(v0, 'insp:I'), hm(0, 14)),
                   (1, 0, L(v1, 'maint:I'), hm(0, 9)), (1, 0, L(v1, 'insp:I'), hm(0, 15))],
            trips=[(2, 0, hm(0, 9))], exits=[], rent={})
T_END = hm(1, 0)
mut = {}
c0 = checker.check(inst, base, t_end=T_END)
mut['baseline_valid'] = dict(ok=c0['ok'], violations=c0['violations'])
def mk(f):
    t = copy.deepcopy(base); f(t); return checker.check(inst, t, t_end=T_END + SLOTS_PER_DAY)['violations'][:3]
mut['a_start_before_approval'] = mk(lambda t: t['modes'].__setitem__(0, (0, 0, L(v0, 'maint:I'), hm(0, 10, 30))))
mut['b_remove_move_out_overfills_IN'] = mk(lambda t: t.__setitem__('trips', []))
mut['c_skip_inspection_and_release'] = mk(lambda t: (t['modes'].pop(2), t['exits'].append((0, hm(1, 10)))))
mut['d_mechanic_double_booked'] = mk(lambda t: t['modes'].__setitem__(3, (1, 0, L(v1, 'maint:I'), hm(0, 11))))
mut['e_work_while_car_at_OUT'] = mk(lambda t: t['modes'].append((2, 0, L(v2, 'insp:I'), hm(0, 10))))
mut['f_release_before_customer_booking'] = mk(lambda t: t['exits'].append((1, hm(0, 16))))
mut['g_trip_cars_not_multiple_of_q'] = mk(lambda t: t.__setitem__('trips', [(2, 4, hm(0, 9))]))
results['checker_mutations'] = mut


def conv(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


json.dump(results, open(os.path.join(OUT, 'e1_cases.json'), 'w'), ensure_ascii=False, indent=1, default=conv)
print(json.dumps(results, ensure_ascii=False, indent=1, default=conv)[:6000])
