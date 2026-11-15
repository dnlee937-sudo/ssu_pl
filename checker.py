"""Independent checker.  Uses only the instance data and the event list of a trajectory
(no model matrices).  Works on the exact 5-minute grid and additionally sweeps the change points
of parking occupancy (event-based check).

Trajectory format
  modes : list of (vid, offset, Mode, s)   offset = original index of task Mode.f
  trips : list of (vid, kid, s)
  exits : list of (vid, s)
  rent  : {day: spaces}
  t_end : end slot of the checked period (exclusive)
"""
from __future__ import annotations
from collections import defaultdict
import heapq
import numpy as np
from instance import (SLOT_MIN, SLOTS_PER_DAY, REST_SLOTS, INF, business_mask, pickup_slot,
                      collect_slot, mode_span, mode_profile)


def vehicle_rules(v):
    """Exogenous facts the checker derives itself from the instance."""
    entry = (0, v.init_loc) if v.arrival is None else (v.arrival, 'IN')
    hold_until = 0
    if v.approval_at is not None:
        hold_until = v.approval_at
    fault_hold = None
    if v.fault is not None:
        fault_hold = (v.fault['notice'], v.fault['decision'])
    return entry, hold_until, fault_hold


def check(inst, traj, t_end=None, strict_release=True, include=None, info_time=None):
    """info_time: check a plan made at that time -> faults announced later are not yet known."""
    t_end = t_end or inst.horizon
    viol = []
    V = {v.vid: v for v in inst.vehicles}
    bm = business_mask(t_end + 2000)
    by_v = defaultdict(list)
    for (vid, off, md, s) in traj['modes']:
        by_v[vid].append(('mode', s, off, md))
    for (vid, kid, s) in traj['trips']:
        by_v[vid].append(('trip', s, kid))
    for (vid, s) in traj['exits']:
        by_v[vid].append(('exit', s))
    trips = {tt.kid: tt for tt in inst.trip_types}

    wait_in = np.zeros(t_end, int); wait_out = np.zeros(t_end, int)
    hand_in = np.zeros(t_end, int)
    use = defaultdict(lambda: np.zeros(t_end + 2000, int))
    staff_intervals = defaultdict(list)   # group -> list of (a, b)
    vrec = {}
    for vid, v in V.items():
        if include is not None and vid not in include:
            continue
        entry_t, entry_loc = (0, v.init_loc) if v.arrival is None else (v.arrival, 'IN')
        if entry_t >= t_end:
            continue
        evs = sorted(by_v.get(vid, []), key=lambda e: (e[1], 0 if e[0] == 'exit' else 1))
        state, loc, free_at = 'wait', entry_loc, entry_t
        next_task = 0
        ntask = len(v.tasks)
        completion = 0 if ntask == 0 else None
        waived = False
        hold = v.approval_at if v.approval_at is not None else 0
        fault = v.fault
        if info_time is not None and fault is not None and fault['notice'] > info_time:
            fault = None
        exit_t = None
        seg_start = entry_t
        segs = []           # waiting segments (loc, a, b)
        for e in evs:
            kind, s = e[0], e[1]
            if s < free_at:
                viol.append(f'v{vid}: event {kind}@{s} before vehicle free at {free_at}')
            if exit_t is not None:
                viol.append(f'v{vid}: event after exit')
            if fault and fault['outcome'] == 'return' and s >= fault['decision'] and kind == 'mode':
                viol.append(f'v{vid}: work started after return decision')
            if kind == 'mode':
                off, md = e[2], e[3]
                if loc != 'IN':
                    viol.append(f'v{vid}: mode start while at {loc}')
                if off + md.f != next_task:
                    viol.append(f'v{vid}: task order broken (task {off + md.f} expected {next_task})')
                if s < hold:
                    viol.append(f'v{vid}: started before customer approval {hold}')
                if fault and fault['notice'] <= s < fault['decision']:
                    viol.append(f'v{vid}: started during fault consultation')
                if md.kind == 'O' and (off + md.g) == ntask - 1:
                    viol.append(f'v{vid}: outsourced mode covers final inspection')
                if v.tasks[off + md.g] == 'insp' and md.kind != 'Q':
                    viol.append(f'v{vid}: inspection not internal')
                if md.kind == 'O':
                    if md.staff_slots is not None and md.staff_slots.size and not bm[s + md.staff_slots].all():
                        viol.append(f'v{vid}: staff needed outside business hours')
                elif not bm[s]:
                    viol.append(f'v{vid}: internal work started outside business hours')
                segs.append((loc, seg_start, s))
                sp_, res_, park_ = mode_profile(md, s)
                for r, prof in res_.items():
                    e_ = max(0, min(sp_, use[r].size - s))
                    use[r][s:s + e_] += prof[:e_]
                    if r in ('gen', 'mech', 'body', 'insp'):
                        nz = np.nonzero(prof)[0]
                        if nz.size:
                            runs = np.split(nz, np.nonzero(np.diff(nz) > 1)[0] + 1)
                            for run in runs:
                                for _ in range(int(prof[run[0]])):
                                    staff_intervals[r].append((s + run[0], s + run[-1] + 1))
                for h in np.nonzero(park_)[0]:
                    if s + h < t_end:
                        hand_in[s + h] += 1
                next_task = off + md.g + 1
                free_at = s + sp_
                seg_start = free_at
                loc = 'IN'
                if next_task == ntask:
                    completion = free_at
            elif kind == 'trip':
                tt = trips[e[2]]
                if loc != tt.src:
                    viol.append(f'v{vid}: trip from {tt.src} while at {loc}')
                if tt.src == 'OUT' and seg_start > s - tt.pre:
                    viol.append(f'v{vid}: not at OUT when staff left for recovery')
                segs.append((loc, seg_start, s))
                if tt.mode == 'tow':
                    h = s if tt.src == 'IN' else s + tt.move - 1
                    if h < t_end:
                        hand_in[h] += 1
                free_at = s + tt.move
                seg_start = free_at
                loc = tt.dst
            elif kind == 'exit':
                if fault and fault['notice'] <= s < fault['decision']:
                    viol.append(f'v{vid}: released during fault consultation')
                if loc != 'IN':
                    viol.append(f'v{vid}: released while at {loc}')
                if fault and fault['outcome'] == 'return' and fault['decision'] <= s:
                    waived = True
                    ok_t = collect_slot(v)
                else:
                    if completion is None:
                        viol.append(f'v{vid}: released before completion')
                        ok_t = INF
                    else:
                        ok_t = pickup_slot(v, completion)
                        if ok_t is None:
                            viol.append(f'v{vid}: released without customer reply')
                            ok_t = INF
                if strict_release and s < ok_t:
                    viol.append(f'v{vid}: released at {s} before customer pickup {ok_t}')
                segs.append((loc, seg_start, s))
                exit_t = s
                loc = None
        if exit_t is None:
            segs.append((loc, seg_start, t_end))
        for (l, a, b) in segs:
            a, b = max(a, 0), min(b, t_end)
            if b > a:
                (wait_in if l == 'IN' else wait_out)[a:b] += 1
        vrec[vid] = dict(completion=completion, exit=exit_t, next_task=next_task, ntask=ntask,
                         waived=waived)

    # trips: participants per (kid, s) multiple of q;  staff and support-car usage
    grp = defaultdict(int)
    for (vid, kid, s) in traj['trips']:
        grp[(kid, s)] += 1
    for (kid, s), c in grp.items():
        tt = trips[kid]
        if c % tt.q:
            viol.append(f'trip {kid}@{s}: {c} cars for q={tt.q}')
        w = c // tt.q
        if tt.mode == 'staff':
            a, b = s - tt.pre, s + tt.back + REST_SLOTS
            if not bm[a:s + tt.back + REST_SLOTS].all():
                viol.append(f'trip {kid}@{s}: staff outside business hours')
            cr = (traj.get('crew') or {}).get((kid, s), {})
            ntech = sum(cr.values())
            if ntech > w * (tt.q + 1):
                viol.append(f'trip {kid}@{s}: crew {ntech} above {w * (tt.q + 1)}')
            use['gen'][a:b] += w * (tt.q + 1) - ntech
            for g_, n_ in cr.items():
                use[g_][a:b] += n_
                for _ in range(n_):
                    staff_intervals[g_].append((a, b))
            use['supportcar'][a:s + tt.back] += w
            for _ in range(w * (tt.q + 1) - ntech):
                staff_intervals['gen'].append((a, b))
        else:
            use['sup_tow'][s:s + tt.move] += w
            h = s if tt.src == 'IN' else s + tt.move - 1
            use['gen'][h] += w
            staff_intervals['gen'].append((h, h + 1))

    # parking capacity (slot check)
    rent = np.zeros(t_end, int)
    for d, n in traj.get('rent', {}).items():
        a = d * SLOTS_PER_DAY
        rent[max(0, a):max(0, min(a + SLOTS_PER_DAY, t_end))] = n
        if n > inst.rent_max:
            viol.append(f'rent {n} on day {d} above max')
    occ_in = wait_in + hand_in
    occ_out = wait_out
    over_in = np.nonzero(occ_in > inst.park_cap['IN'])[0]
    over_out = np.nonzero(occ_out > inst.park_cap['OUT'] + rent)[0]
    # internal excess waiting is part of the model (E_b): recorded, not a violation
    excess_in = int(np.maximum(occ_in - inst.park_cap['IN'], 0).sum())
    if over_out.size:
        viol.append(f'OUT parking over capacity at {over_out.size} slots')
    # event-based sweep (independent second computation of IN occupancy)
    chg = np.nonzero(np.diff(np.r_[0, occ_in]))[0]
    sweep_max = max([occ_in[c] for c in chg] + [0])
    if sweep_max != (occ_in.max() if occ_in.size else 0):
        viol.append('event sweep disagrees with slot check')
    # resources
    for r, arr in use.items():
        cap = inst.caps[r] * (bm[:t_end + 2000].astype(int) if r in inst.staff_resources
                              else np.ones(t_end + 2000, int))
        bad = np.nonzero(arr[:t_end + 2000] > cap)[0]
        bad = bad[bad < t_end + 2000]
        if bad.size:
            viol.append(f'resource {r} over capacity at {bad.size} slots (first {bad[0]})')
    # individual assignment by interval partitioning (identical calendars inside a group)
    assign_ok = {}
    for r, ivs in staff_intervals.items():
        ivs.sort()
        heap = []
        npers = 0
        okr = True
        for (a, b) in ivs:
            if heap and heap[0][0] <= a:
                _, pid = heapq.heappop(heap)
            else:
                pid = npers; npers += 1
            heapq.heappush(heap, (b, pid))
        if npers > inst.caps[r]:
            okr = False
            viol.append(f'staff group {r}: needs {npers} persons > {inst.caps[r]}')
        assign_ok[r] = npers
    return dict(ok=not viol, violations=viol[:50], n_viol=len(viol), vrec=vrec,
                occ_in=occ_in, occ_out=occ_out, rent=rent, staff_persons=assign_ok,
                excess_in_slots=int(over_in.size), excess_in_space_min=excess_in * SLOT_MIN,
                max_in=int(occ_in.max()) if occ_in.size else 0,
                max_out=int(occ_out.max()) if occ_out.size else 0)


def evaluate(inst, traj, chk, t_obs):
    """Cost and service metrics of a trajectory over the observation window [0, t_obs)."""
    V = {v.vid: v for v in inst.vehicles}
    trips = {tt.kid: tt for tt in inst.trip_types}
    work = sum(md.cost for (_, _, md, s) in traj['modes'] if s < t_obs)
    work_I = sum(md.cost for (_, _, md, s) in traj['modes'] if s < t_obs and md.kind != 'O')
    work_O = work - work_I
    grp = defaultdict(int)
    for (vid, kid, s) in traj['trips']:
        if s - trips[kid].pre < t_obs:
            grp[(kid, s)] += 1
    trip_cost = sum((c // trips[k].q) * trips[k].cost for (k, s), c in grp.items())
    n_staff_trips = sum((c // trips[k].q) for (k, s), c in grp.items() if trips[k].mode == 'staff')
    n_tow = sum(c for (k, s), c in grp.items() if trips[k].mode == 'tow')
    rent = sum(n * inst.rent_cost_day for d, n in traj.get('rent', {}).items() if d * SLOTS_PER_DAY < t_obs)
    late_min, late_cnt, ontime, comp_n = 0.0, 0, 0, 0
    pick_wait = 0.0
    storage_days = 0.0
    status = defaultdict(int)
    residual_lb = 0.0
    residual_late_lb = 0.0
    unfinished_days = 0.0
    from instance import remaining_relaxation
    for vid, r in chk['vrec'].items():
        v = V[vid]
        eff_dl = v.deadline
        if v.approval_at is not None:
            if v.approval_at >= INF:
                eff_dl = INF
            else:
                base = v.arrival if v.arrival is not None else 0
                eff_dl = v.deadline + max(0, v.approval_at - base)
        returned = v.fault is not None and v.fault['outcome'] == 'return' and v.fault['decision'] < t_obs
        comp = r['completion']
        # unfinished vehicle-time inside the observation window (holding cost basis)
        if v.tasks:
            st = max(0, v.arrival if v.arrival is not None else 0)
            en = t_obs if comp is None else min(comp, t_obs)
            if returned:
                en = min(en, v.fault['decision'])
            unfinished_days += max(0, en - st) / SLOTS_PER_DAY
        if returned and (comp is None or comp > v.fault['decision']):
            status['returned_settled' if r['exit'] is None or r['exit'] >= t_obs else 'returned_released'] += 1
            continue
        if comp is not None and comp <= t_obs and v.tasks:
            comp_n += 1
            lm = max(0, SLOT_MIN * (comp - eff_dl)) if eff_dl < INF else 0
            late_min += lm; late_cnt += lm > 0; ontime += lm == 0
        if r['exit'] is not None and r['exit'] < t_obs:
            status['released'] += 1
            ps = pickup_slot(v, comp if comp is not None else 0)
            if ps is not None:
                pick_wait += SLOT_MIN * max(0, r['exit'] - ps)
        elif comp is not None and comp <= t_obs:
            status['done_waiting_customer'] += 1
        else:
            # unfinished at t_obs: classify
            if v.approval_at is not None and v.approval_at > t_obs:
                status['wait_approval'] += 1
            elif v.fault is not None and v.fault['notice'] <= t_obs < v.fault['decision']:
                status['wait_fault_decision'] += 1
            elif eff_dl < t_obs:
                status['open_overdue'] += 1
            else:
                status['open_in_time'] += 1
            # Work is charged when its mode starts; an ongoing mode is already paid.
            # Its outstanding duration must precede all unstarted work. Bundle modes
            # are included by a task-path relaxation; do not sum stand-alone prices.
            started = [(off, md, s) for (vv, off, md, s) in traj['modes']
                       if vv == vid and s < t_obs]
            done = {off + k for off, md, s in started for k in range(md.f, md.g + 1)}
            rem = [v.tasks[k] for k in range(len(v.tasks)) if k not in done]
            cmin, pmin = remaining_relaxation(rem)
            residual_lb += float(cmin[0])
            ready = max([t_obs] + [s + mode_span(md, s) for off, md, s in started])
            if eff_dl < INF:
                earliest = ready + pmin[0]
                residual_late_lb += inst.lateness_cost_min * SLOT_MIN * max(0, earliest - eff_dl)
        # customer-side storage billing (completion notice + grace -> pickup booking / exit / t_obs)
        if comp is not None and comp < t_obs:
            ps = pickup_slot(v, comp)
            end = min(x for x in [ps, r['exit'], t_obs] if x is not None)
            start = comp + inst.storage_grace_days * SLOTS_PER_DAY
            if end > start:
                storage_days += (end - start) / SLOTS_PER_DAY
    late_cost = inst.lateness_cost_min * late_min
    pick_cost = inst.pickup_wait_cost_min * pick_wait
    econ = work + trip_cost + rent + late_cost + pick_cost
    occ = chk['occ_in'][:t_obs]
    days = t_obs // SLOTS_PER_DAY
    daily_inv = [int(chk['occ_in'][d * SLOTS_PER_DAY + 17 * 12] + chk['occ_out'][d * SLOTS_PER_DAY + 17 * 12])
                 for d in range(days)]
    return dict(econ=econ, work=work, work_internal=work_I, work_outsourced=work_O, trips=trip_cost,
                n_staff_trips=n_staff_trips, n_tow=n_tow, rent=rent, late_cost=late_cost,
                late_min=late_min, late_cnt=late_cnt, completed=comp_n, ontime=ontime,
                pickup_wait_cost=pick_cost, storage_billed=storage_days * inst.storage_fee_day,
                status=dict(status), residual_work_lb=residual_lb, residual_late_lb=residual_late_lb,
                max_in=int(occ.max()), max_out=int(chk['occ_out'][:t_obs].max()),
                daily_inventory_17h=daily_inv, unfinished_vehicle_days=unfinished_days,
                hold_cost=unfinished_days * getattr(inst, 'hold_cost_day', 0.0))
