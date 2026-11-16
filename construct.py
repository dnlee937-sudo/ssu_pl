"""Rule-based construction for one planning window (= the '초기 구성' / rule-based operation).

R1  work  : vehicles in (deadline, entry) order; each task gets the earliest resource-feasible
            start of its modes; the internal mode is kept if it still meets the deadline, otherwise
            the mode with the earliest completion (ties: lower cost).
R2  return: a vehicle waiting at OUT is brought back by a single-car staff trip (or towing if staff
            is unavailable) arriving before its next task / customer pickup.
R3  space : while IN parking is over capacity at some slot, the vehicle with the longest remaining
            wait at that slot is moved out (single-car staff trip, else towing) and brought back
            before its next event; OUT spaces are rented per day when OUT is full.
R4  release: a vehicle with a known pickup booking leaves at the first release slot >= booking.
Everything uses the candidate sets of the window model, so the result can be encoded exactly.
If R3 cannot remove an overflow the plan is returned with overflow (flagged invalid).
"""
from __future__ import annotations
from collections import defaultdict
import numpy as np
from instance import REST_SLOTS, SLOTS_PER_DAY, INF
from instance import mode_span, mode_profile


def construct(W, M, rent_lb=None):
    inst, t0, t1 = W.inst, W.t0, W.t1
    T = t1 - t0
    tX = M.tX                                   # resources are checked over the extended range T+
    resid = {r: np.array(M.Qx[r], float).copy() for r in inst.caps}
    trips = {tt.kid: tt for tt in inst.trip_types}
    cand = defaultdict(list)
    for j in range(M.xv.size):
        cand[(int(M.xv[j]), int(M.xb[j]))].append(int(M.xs[j]))
    tcand = defaultdict(list)
    for k, s in zip(M.wk, M.ws):
        tcand[int(k)].append(int(s))
    ecand = defaultdict(list)
    for i, s in zip(M.ev, M.es):
        ecand[int(i)].append(int(s))
    kid = {}
    for tt in inst.trip_types:
        if tt.q == 1:
            kid[(tt.src, tt.dst, tt.mode)] = tt.kid
    plan_modes, plan_trips, plan_exits = [], [], []
    rent = np.zeros(len(M.days), int)
    for q, d in enumerate(M.days):
        rent[q] = (rent_lb or {}).get(int(d), 0)
    hand_in = W.fixed_park.copy().astype(int)

    TECH = [g for g in ('mech', 'body', 'insp') if g in resid]
    crew = {}

    def crew_split(a, b, need):
        """general staff first; technicians join only when general staff are short (R2/R3)"""
        out, left = {}, need
        for g in ['gen'] + TECH:
            if left == 0:
                break
            av = int(resid[g][a:b].min()) if b > a else 0
            n = min(left, max(0, av))
            if n:
                out[g] = n; left -= n
        return out if left == 0 else None

    def trip_ok(k, s):
        tt = trips[k]
        if tt.mode == 'staff':
            a, b = s - tt.pre - t0, s + tt.back + REST_SLOTS - t0
            return crew_split(a, b, tt.q + 1) is not None and (resid['supportcar'][a:s + tt.back - t0] >= 1).all()
        h = (s if tt.src == 'IN' else s + tt.move - 1) - t0
        return (resid['sup_tow'][s - t0:s + tt.move - t0] >= 1).all() and resid['gen'][h] >= 1

    def trip_take(k, s):
        tt = trips[k]
        if tt.mode == 'staff':
            a, b = s - tt.pre - t0, s + tt.back + REST_SLOTS - t0
            sp_ = crew_split(a, b, tt.q + 1)
            for g, n in sp_.items():
                resid[g][a:b] -= n
                if g != 'gen':
                    d_ = crew.setdefault((int(k), int(s)), {})
                    d_[g] = d_.get(g, 0) + n
            resid['supportcar'][a:s + tt.back - t0] -= 1
        else:
            h = (s if tt.src == 'IN' else s + tt.move - 1) - t0
            resid['sup_tow'][s - t0:s + tt.move - t0] -= 1
            resid['gen'][h] -= 1
            hand_in[h] += 1

    m = M.m

    def out_ready(arrival_at_out):
        """earliest staff-departure slot for a recovery from OUT that the block model accepts"""
        b = (arrival_at_out - t0) // m
        return t0 + (b + 1) * m

    def find_trip(src, dst, lo_dep, hi_arr=INF, latest=False, vehicle_entry=None, hi_dep=INF,
                  out_since=None):
        """single-car trip src->dst departing in [lo_dep, hi_dep] and arriving <= hi_arr."""
        for mode in ('staff', 'tow'):
            k = kid[(src, dst, mode)]
            tt = trips[k]
            ss = [s for s in tcand[k] if lo_dep <= s <= hi_dep and s + tt.move <= hi_arr
                  and (vehicle_entry is None or s >= vehicle_entry)
                  and (out_since is None or s - tt.pre >= out_ready(out_since))]
            if latest:
                ss = ss[::-1]
            for s in ss:
                if trip_ok(k, s):
                    return k, s
        return None

    order = sorted(range(len(W.vehs)), key=lambda i: (W.vehs[i]['deadline'], W.vehs[i]['entry_slot'] or t0))
    # per vehicle timeline of physical states inside the window: list of (loc, a, b)
    life = {}
    for i in order:
        v = W.vehs[i]
        ent = v['entry_slot'] if v['entry_slot'] is not None else t0
        loc = v['entry_loc']
        segs = []
        cur_t = ent
        ready = max(ent, v['hold'])
        nk = len(v['tasks'])
        first_event = None
        # R2: bring back from OUT before first work (reserve the earliest return)
        if loc == 'OUT' and nk and ready < t1:
            got = find_trip('OUT', 'IN', max(ready, t0), t1, vehicle_entry=ent,
                            out_since=None if v['entry_slot'] is None else ent)
            if got is None:
                life[i] = [('OUT', ent, t1)]
                continue
            k, s = got
            trip_take(k, s); plan_trips.append((v['vid'], k, s))
            segs.append(('OUT', cur_t, s))
            cur_t = s + trips[k].move
            loc = 'IN'
            ready = max(ready, cur_t)
        kk = 0
        while kk < nk and ready < t1:
            opts = []
            for b, md in enumerate(v['modes']):
                if md.f != kk:
                    continue
                for s in cand.get((i, b), []):
                    if s < ready:
                        continue
                    sp_, res_, _pk = mode_profile(md, s)
                    e = min(sp_, tX - s)
                    if all((resid[r][s - t0:s - t0 + e] >= prof[:e]).all() for r, prof in res_.items()):
                        opts.append((s, b, md)); break
            if not opts:
                break
            rem_after = lambda md: M.remaining_duration[i][md.g + 1]
            internal = [o for o in opts if o[2].kind != 'O']
            pick = None
            if internal:
                s, b, md = internal[0]
                if s + mode_span(md, s) + rem_after(md) <= v['deadline']:
                    pick = internal[0]
            if pick is None:
                pick = min(opts, key=lambda o: (o[0] + mode_span(o[2], o[0]) + rem_after(o[2]), o[2].cost))
            s, b, md = pick
            sp_, res_, park_ = mode_profile(md, s)
            for r, prof in res_.items():
                e = min(sp_, tX - s)
                resid[r][s - t0:s - t0 + e] -= prof[:e]
            for h in np.nonzero(park_)[0]:
                if s + h < t1:
                    hand_in[s + h - t0] += 1
            plan_modes.append((v['vid'], b, s))
            segs.append(('IN', cur_t, s))
            cur_t = s + sp_
            ready = cur_t
            kk = md.g + 1
        # R4: release for vehicles whose pickup is known (no remaining tasks)
        if nk == 0 and v['pickup_req'] is not None and ecand.get(i):
            # if waiting at OUT bring back before pickup
            if loc == 'OUT':
                cands = [s for s in ecand[i]]
                done = False
                for se in cands:
                    got = find_trip('OUT', 'IN', max(t0, ent), se, latest=True, vehicle_entry=ent,
                                    out_since=None if v['entry_slot'] is None else ent)
                    if got:
                        k, s = got
                        trip_take(k, s); plan_trips.append((v['vid'], k, s))
                        segs.append(('OUT', cur_t, s)); cur_t = s + trips[k].move; loc = 'IN'
                        plan_exits.append((v['vid'], se)); segs.append(('IN', cur_t, se)); cur_t = None
                        done = True
                        break
                if not done:
                    segs.append(('OUT', cur_t, t1)); cur_t = None
            else:
                se = next((s for s in ecand[i] if s >= cur_t), None)
                if se is not None:
                    plan_exits.append((v['vid'], se)); segs.append(('IN', cur_t, se)); cur_t = None
        if cur_t is not None:
            if cur_t < t1:
                segs.append((loc, cur_t, t1))
        # zero-length waits are kept: the m>1 capacity bound counts every arrival in a block
        life[i] = [sg for sg in segs if sg[2] > sg[1] or (M.m > 1 and sg[2] == sg[1] and sg[1] < t1)]

    # R3: IN overflow resolution, counted exactly like the capacity rows of the window model:
    #   m == 1 : a waiting segment [a,b) occupies slots a..b-1
    #   m >  1 : it occupies blocks blk(a)..blk(b) (conservative bound y[b-1] + arrivals in block)
    NB = M.NB
    blk = lambda t: (t - t0) // m

    def seg_blocks(a, b):
        a = max(a, t0); b = min(b, t1)
        if m == 1:
            return range(a - t0, b - t0) if b > a else range(0)
        if b < a or a >= t1:
            return range(0)
        return range(blk(a), blk(min(b, t1 - 1)) + 1)

    fp = np.zeros(NB * m, int); fp[:T] = W.fixed_park
    fixed_blk = fp.reshape(NB, m).max(axis=1)
    fpo = np.zeros(NB * m, int); fpo[:T] = W.fixed_out_park
    fixed_out_blk = fpo.reshape(NB, m).max(axis=1)
    hand_ev = []                     # list of arrays of blocks occupied by handover events
    for (vid_, b_, s_) in plan_modes:
        md = next(v_ for v_ in W.vehs if v_['vid'] == vid_)['modes'][b_]
        nz = np.nonzero(md.park)[0]
        if nz.size:
            ts_ = s_ + nz
            ts_ = ts_[(ts_ >= t0) & (ts_ < t1)]
            hand_ev.append(np.unique((ts_ - t0) // m))
    tow_ev = []

    def occupancy():
        occ = {'IN': fixed_blk.copy(), 'OUT': fixed_out_blk.copy()}
        for arr in hand_ev:
            occ['IN'][arr] += 1
        for h in tow_ev:
            occ['IN'][(h - t0) // m] += 1
        for i, segs in life.items():
            for (l, a, b) in segs:
                rg = seg_blocks(a, b)
                if len(rg):
                    occ[l][rg.start:rg.stop] += 1
        return occ

    def note_tow(k, s):
        tt = trips[k]
        if tt.mode == 'tow':
            tow_ev.append(s if tt.src == 'IN' else s + tt.move - 1)

    for (vid_, k_, s_) in plan_trips:
        note_tow(k_, s_)
    capin = inst.park_cap['IN']
    day_positions = [np.searchsorted(M.days, dates) for dates in M.block_days]
    blocks_by_day = [[] for _ in M.days]
    for b, positions in enumerate(day_positions):
        for q in positions:
            blocks_by_day[q].append(b)
    occ0 = occupancy()['OUT']
    for q in range(len(M.days)):
        blocks = blocks_by_day[q]
        if blocks:
            rent[q] = min(inst.rent_max, max(rent[q], int(occ0[blocks].max()) - inst.park_cap['OUT']))
    tried = set()
    skipped = set()
    occ = occupancy()
    guard = 0
    while guard < 6000:
        guard += 1
        over = [b for b in np.nonzero(occ['IN'] > capin)[0] if b not in skipped]
        if not over:
            break
        bstar = int(over[0])
        bs_t = t0 + bstar * m                 # first slot of the overflow block
        hi_dep = bs_t if m == 1 else bs_t - 1
        lo_arr_back = bs_t + 1 if m == 1 else bs_t + m
        best = None
        for i, segs in life.items():
            for q, (l, a, b) in enumerate(segs):
                if l != 'IN' or (i, a) in tried:
                    continue
                rg = seg_blocks(a, b)
                if bstar not in rg or a > hi_dep:
                    continue
                if best is None or b > best[3]:
                    best = (i, q, a, b)
        if best is None:
            skipped.add(bstar)
            continue
        i, q, a, b = best
        tried.add((i, a))
        v = W.vehs[i]
        ent = v['entry_slot'] if v['entry_slot'] is not None else t0
        go = find_trip('IN', 'OUT', a, hi_dep=hi_dep, vehicle_entry=ent)
        if go is None:
            continue
        k1, s1 = go
        arr1 = s1 + trips[k1].move
        back = None
        if b < t1:
            back = find_trip('OUT', 'IN', arr1, b, latest=True, out_since=arr1)
            if back is None or back[1] + trips[back[0]].move < lo_arr_back:
                continue
        end_out = back[1] if back else t1
        rg = seg_blocks(arr1, end_out)
        add = defaultdict(int)
        for bb in rg:
            for pos in day_positions[bb]:
                nd = occ['OUT'][bb] + 1 - inst.park_cap['OUT'] - rent[pos]
                if nd > 0:
                    add[pos] = max(add[pos], nd)
        if any(rent[p_] + nd > inst.rent_max for p_, nd in add.items()):
            continue
        for p_, nd in add.items():
            rent[p_] += nd
        trip_take(k1, s1); plan_trips.append((v['vid'], k1, s1)); note_tow(k1, s1)
        new = [('IN', a, s1), ('OUT', arr1, end_out)]
        if back:
            trip_take(*back); plan_trips.append((v['vid'], back[0], back[1])); note_tow(*back)
            new.append(('IN', back[1] + trips[back[0]].move, b))
        segs = life[i]
        life[i] = segs[:q] + [sg for sg in new if sg[2] > sg[1] or (m > 1 and sg[2] == sg[1])] + segs[q + 1:]
        occ = occupancy()
    overflow = int(np.maximum(occ['IN'] - capin, 0).sum())
    return dict(modes=plan_modes, trips=plan_trips, exits=plan_exits,
                rent={int(d): int(rent[q]) for q, d in enumerate(M.days)}, overflow=overflow, crew=crew)
