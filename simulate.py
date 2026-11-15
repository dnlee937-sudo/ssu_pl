"""Rolling (daily / event-driven) re-planning over a 14-calendar-day observation window.

Policies
  'init' : rule-based construction + parking/move repair MILP (+ conflict repair)   = 초기 구성만
  'lns'  : 'init' followed by fixed-weight LNS with MILP repair inside the same time budget

Only information revealed up to the re-planning time is used (arrivals of the current day,
customer approvals / replies already received, fault notices already issued).
"""
from __future__ import annotations
import time, json, math, os, resource
from collections import defaultdict, Counter
import numpy as np

from instance import (SLOT_MIN, SLOTS_PER_DAY, REPLAN_SLOT, REST_SLOTS, INF, business_mask,
                      build_modes, pickup_slot, reply_slot, collect_slot, is_business_day, CLOSE_SLOT, mode_span, mode_profile)
from model import Window, WindowModel, LOC
import checker


class ExecutionGateError(RuntimeError):
    """No executable validated prefix is available; committed history is preserved."""


# ---------------------------------------------------------------------------- state
def replay(v, traj_by_v, t0):
    """State of vehicle v at t0 from executed events with start < t0."""
    entry_t, entry_loc = (0, v.init_loc) if v.arrival is None else (v.arrival, 'IN')
    st = dict(next_task=0, loc=entry_loc, free_at=entry_t, completion=0 if not v.tasks else None,
              exited=False, busy_until=None, moving_to=None)
    for e in sorted(traj_by_v.get(v.vid, []), key=lambda e: e[1]):
        kind, s = e[0], e[1]
        if s >= t0:
            break
        if kind == 'mode':
            off, md = e[2], e[3]
            st['next_task'] = off + md.g + 1
            st['free_at'] = s + mode_span(md, s)
            st['loc'] = 'IN'
            if st['next_task'] == len(v.tasks):
                st['completion'] = s + mode_span(md, s)
        elif kind == 'trip':
            tt = e[2]
            st['free_at'] = s + tt.move
            st['loc'] = tt.dst
        elif kind == 'exit':
            st['exited'] = True
    return st


class Rolling:
    def __init__(self, inst, policy='lns', m=1, budget=120.0, window_days=4, t_obs=None,
                 repair_limit=4.0, seed=11, log=None, verbose=False, max_iter=10 ** 6,
                 carry=False, stage1_limit=40.0):
        self.inst, self.policy, self.m, self.budget = inst, policy, m, budget
        self.window = window_days * SLOTS_PER_DAY
        self.t_obs = t_obs or inst.horizon
        self.repair_limit = repair_limit
        self.rng = np.random.default_rng(seed)
        self.traj = dict(modes=[], trips=[], exits=[], rent={}, crew={})
        if policy == 'lns_carry':       # optional variant: start from the carried previous plan when better
            self.policy, carry = 'lns', True
        self.carry, self.stage1_limit = carry, stage1_limit
        self.prev = None
        self.by_v = defaultdict(list)
        self.log = log if log is not None else []
        self.verbose = verbose
        self.max_iter = max_iter
        self.V = {v.vid: v for v in inst.vehicles}
        self.trip_by_id = {tt.kid: tt for tt in inst.trip_types}

    # ------------------------------------------------------------------ replanning times
    def replan_times(self):
        ts = set()
        for d in range(self.t_obs // SLOTS_PER_DAY):
            if is_business_day(d):
                ts.add(d * SLOTS_PER_DAY + REPLAN_SLOT)
        for v in self.inst.vehicles:
            if v.fault:
                for key in ('notice', 'decision'):
                    if v.fault[key] < self.t_obs:
                        ts.add(v.fault[key])
        return sorted(t for t in ts if t < self.t_obs)

    # ------------------------------------------------------------------ window state
    def window_state(self, t0, t1):
        inst = self.inst
        today = t0 // SLOTS_PER_DAY
        vehs, fixed_res, fixed_park = [], defaultdict(lambda: np.zeros(t1 - t0, int)), np.zeros(t1 - t0, int)
        fixed_out_park = np.zeros(t1 - t0, int)
        # fixed usage of executed modes / trips that overlap the window
        for (vid, off, md, s) in self.traj['modes']:
            sp_, res_, park_ = mode_profile(md, s)
            if s + sp_ > t0:
                for r, prof in res_.items():
                    a = max(t0, s); b = min(t1, s + sp_)
                    if b > a:
                        fixed_res[r][a - t0:b - t0] += prof[a - s:b - s]
                for h in np.nonzero(park_)[0]:
                    if t0 <= s + h < t1:
                        fixed_park[s + h - t0] += 1
        grp = Counter((kid, s) for (_, kid, s) in self.traj['trips'])
        for (kid, s), c in grp.items():
            tt = self.trip_by_id[kid]
            w = c // tt.q
            if tt.mode == 'staff':
                cr = self.traj['crew'].get((kid, s), {})
                ntech = sum(cr.values())
                for t in range(s - tt.pre, s + tt.back + REST_SLOTS):
                    if t0 <= t < t1:
                        fixed_res['gen'][t - t0] += w * (tt.q + 1) - ntech
                        for g_, n_ in cr.items():
                            fixed_res[g_][t - t0] += n_
                for t in range(s - tt.pre, s + tt.back):
                    if t0 <= t < t1:
                        fixed_res['supportcar'][t - t0] += w
            else:
                for t in range(s, s + tt.move):
                    if t0 <= t < t1:
                        fixed_res['sup_tow'][t - t0] += w
                h = s if tt.src == 'IN' else s + tt.move - 1
                if t0 <= h < t1:
                    fixed_res['gen'][h - t0] += w
                    fixed_park[h - t0] += 1
        for v in inst.vehicles:
            if v.arrival is not None and (v.arrival // SLOTS_PER_DAY > today or v.arrival >= t1):
                continue            # not yet known
            st = replay(v, self.by_v, t0)
            # A recovery is committed when staff leave, before the vehicle departs.
            # Keep its source waiting occupancy fixed until departure, and expose
            # the vehicle to new decisions only at the fixed arrival at IN.
            pending = [(self.trip_by_id[k], s) for (vid, k, s) in self.traj['trips']
                       if vid == v.vid and s - self.trip_by_id[k].pre < t0 <= s]
            if pending:
                if len(pending) != 1:
                    raise ValueError('overlapping committed predeparture trips')
                tt, s = pending[0]
                if tt.src != 'OUT':
                    raise ValueError('unsupported predeparture source')
                fixed_out_park[:max(0, min(s, t1) - t0)] += 1
                st['loc'], st['free_at'] = tt.dst, s + tt.move
            if st['exited']:
                continue
            entry_slot, entry_loc = None, st['loc']
            if st['free_at'] > t0:
                entry_slot = st['free_at']
                if entry_slot >= t1:
                    continue
            nt = st['next_task']
            tasks = v.tasks[nt:]
            hold = t0
            dl = v.deadline
            pickup = None
            if v.approval_at is not None:
                if v.approval_at <= t0:
                    hold = max(hold, v.approval_at)
                    base = v.arrival if v.arrival is not None else 0
                    dl = v.deadline + max(0, v.approval_at - base)
                else:
                    hold, dl = INF, INF
            if v.fault:
                f = v.fault
                if f['notice'] <= t0 < f['decision']:
                    hold = INF
                elif f['decision'] <= t0 and f['outcome'] == 'return':
                    tasks = []
                    pickup = collect_slot(v)
            consulting = v.fault is not None and v.fault['notice'] <= t0 < v.fault['decision']
            if not tasks and pickup is None and not consulting:
                comp = st['completion'] if st['completion'] is not None else 0
                rp = reply_slot(v, comp)
                if rp is not None and rp <= t0:
                    pickup = pickup_slot(v, comp)
            modes = build_modes(tasks) if tasks else []
            vehs.append(dict(vid=v.vid, entry_slot=entry_slot, entry_loc=entry_loc, tasks=tasks,
                             modes=modes, hold=hold, deadline=dl, pickup_req=pickup, offset=nt,
                             hold_start=max(t0, v.arrival if v.arrival is not None else 0)))
        return Window(inst, t0, t1, vehs, dict(fixed_res), fixed_park, fixed_out_park=fixed_out_park)

    # ------------------------------------------------------------------ rule-based construction
    def construct(self, W: Window, M: WindowModel):
        from construct import construct
        return construct(W, M, rent_lb=self.traj['rent'])

    # ------------------------------------------------------------------ helpers on column groups
    def col_groups(self, M):
        V = M.W.vehs
        g = defaultdict(list)
        for j in range(M.xv.size):
            g[int(M.xv[j])].append(M.x0 + j)
        for j in range(M.zv.size):
            g[int(M.zv[j])].append(M.z0 + j)
        for j in range(M.ev.size):
            g[int(M.ev[j])].append(M.e0 + j)
        for j, i in enumerate(M.ne_v):
            g[int(i)].append(M.ne0 + j)
        for j in range(M.ov.size):
            g[int(M.ov[j])].append(M.o0 + j)
        return {i: np.array(c, int) for i, c in g.items()}

    def bounds_fixing(self, M, sol, free_vehicles, base_lo, base_hi, fix_all_x=False, elastic=False):
        lo, hi = base_lo.copy(), base_hi.copy()
        groups = self.cg
        for i, cols in groups.items():
            if i in free_vehicles:
                continue
            v = np.round(sol[cols])
            lo[cols] = v; hi[cols] = v
        if fix_all_x:
            for i, cols in groups.items():
                xo = cols[(cols < M.x0 + M.xv.size) & (cols >= M.x0) | ((cols >= M.o0) & (cols < M.o0 + M.ov.size))]
                v = np.round(sol[xo]); lo[xo] = v; hi[xo] = v
        hi[M.sg0:M.sg0 + M.NB] = np.inf            # E_b (internal excess waiting)
        hi[M.sg0 + M.NB:M.sg0 + 2 * M.NB] = 0      # external lot: hard capacity
        return lo, hi

    def plan_check(self, M, sol, t0, t1):
        plan = M.decode(sol)
        traj = self.compose(M, plan, upto=None)
        known = {v['vid'] for v in M.W.vehs} | {vid for (vid, _, _, _) in self.traj['modes']}
        chk = checker.check(self.inst, traj, t_end=t1, include=self.known_ids(t0), info_time=t0)
        return chk, plan

    def known_ids(self, t0):
        today = t0 // SLOTS_PER_DAY
        return {v.vid for v in self.inst.vehicles if v.arrival is None or v.arrival // SLOTS_PER_DAY <= today}

    def compose(self, M, plan, upto=None):
        """executed history + planned window decisions (optionally only those starting < upto)."""
        V = M.W.vehs
        vmap = {v['vid']: v for v in V}
        traj = dict(modes=list(self.traj['modes']), trips=list(self.traj['trips']),
                    exits=list(self.traj['exits']), rent=dict(self.traj['rent']),
                    crew=dict(self.traj['crew']))
        for (vid, b, s) in plan['modes']:
            if upto is None or s < upto:
                traj['modes'].append((vid, vmap[vid]['offset'], vmap[vid]['modes'][b], s))
        for (vid, k, s) in plan['trips']:
            # Staff departure is the first irreversible action of a recovery.
            if upto is None or s - self.trip_by_id[k].pre < upto:
                traj['trips'].append((vid, k, s))
                if (k, s) in plan.get('crew', {}):
                    traj['crew'][(k, s)] = dict(plan['crew'][(k, s)])
        for (vid, s) in plan['exits']:
            if upto is None or s < upto:
                traj['exits'].append((vid, s))
        for d, n in plan['rent'].items():
            if upto is None or d * SLOTS_PER_DAY < upto:
                traj['rent'][d] = max(traj['rent'].get(d, 0), n)
        return traj

    # ------------------------------------------------------------------ one re-plan
    def window_end(self, t0):
        """end of business on the fourth business day counted from the planning day"""
        d, n = t0 // SLOTS_PER_DAY, 0
        while True:
            if is_business_day(d):
                n += 1
                if n == 4:
                    return d * SLOTS_PER_DAY + CLOSE_SLOT
            d += 1

    def better(self, a, b):
        """lexicographic comparison of (F, Z) pairs"""
        return a[0] < b[0] - 1e-6 or (a[0] <= b[0] + 1e-6 and a[1] < b[1] - 1e-6)

    def carry_plans(self, M, plan0):
        """Plans built from the unexecuted part of the previous plan (Section 4.3.2)."""
        prev, V, t0 = self.prev, M.W.vehs, M.t0
        vid2i = {v['vid']: i for i, v in enumerate(V)}
        xkey = set(zip(M.xv.tolist(), M.xb.tolist(), M.xs.tolist()))
        zkey = set(zip(M.zv.tolist(), M.zk.tolist(), M.zs.tolist()))
        ekey = set(zip(M.ev.tolist(), M.es.tolist()))
        bad, modes, trips, exits = set(), [], [], []
        for (vid, b, s) in prev['plan']['modes']:
            if s < t0:
                continue
            i = vid2i.get(vid)
            if i is None:
                bad.add(vid); continue
            off0, mds0 = prev['vinfo'][vid]
            md0 = mds0[b]
            off1 = V[i]['offset']
            nb = [q for q, md in enumerate(V[i]['modes'])
                  if md.f + off1 == md0.f + off0 and md.g + off1 == md0.g + off0
                  and md.kind == md0.kind and md.label == md0.label]
            if nb and (i, nb[0], s) in xkey:
                modes.append((vid, nb[0], s))
            else:
                bad.add(vid)
        for (vid, k, s) in prev['plan']['trips']:
            if s - self.trip_by_id[k].pre < t0:
                continue
            i = vid2i.get(vid)
            if i is not None and (i, k, s) in zkey:
                trips.append((vid, k, s))
            else:
                bad.add(vid)
        for (vid, s) in prev['plan']['exits']:
            if s < t0:
                continue
            i = vid2i.get(vid)
            if i is not None and (i, s) in ekey:
                exits.append((vid, s))
            else:
                bad.add(vid)
        changed = True                     # a shared trip is kept only with all its members
        while changed:
            changed = False
            grp = defaultdict(set)
            for (vid, k, s) in prev['plan']['trips']:
                if s - self.trip_by_id[k].pre >= t0:
                    grp[(k, s)].add(vid)
            for mem in grp.values():
                if mem & bad and not mem <= bad:
                    bad |= mem; changed = True
        modes = [x for x in modes if x[0] not in bad]
        trips = [x for x in trips if x[0] not in bad]
        exits = [x for x in exits if x[0] not in bad]
        kept = {x[0] for x in modes} | {x[0] for x in trips} | {x[0] for x in exits}
        tk = {(k, s) for (_, k, s) in trips}
        crew = {ks: dict(c) for ks, c in prev['plan'].get('crew', {}).items() if ks in tk}
        lb = self.traj['rent']
        rent = {int(d): max(prev['plan']['rent'].get(int(d), 0), lb.get(int(d), 0)) for d in M.days}
        planA = dict(modes=modes, trips=trips, exits=exits, rent=rent, crew=crew)
        crewB = {ks: dict(c) for ks, c in crew.items()}
        tripsB = [x for x in plan0['trips'] if x[0] not in kept]
        for (k, s) in {(k, s) for (_, k, s) in tripsB}:
            for g, n in plan0.get('crew', {}).get((k, s), {}).items():
                crewB.setdefault((k, s), {}); crewB[(k, s)][g] = crewB[(k, s)].get(g, 0) + n
        planB = dict(modes=modes + [x for x in plan0['modes'] if x[0] not in kept],
                     trips=trips + tripsB,
                     exits=exits + [x for x in plan0['exits'] if x[0] not in kept],
                     rent={d: max(rent[d], plan0['rent'].get(d, 0)) for d in rent}, crew=crewB)
        self._kept = kept
        return [('carry+construct', planB), ('carry', planA)]

    def feasible_vector(self, M, plan, base_lo):
        try:
            sol = M.normalize(M.encode(plan))
        except KeyError:
            return None
        w = sol[M.w0:M.w0 + M.wk.size]
        if np.abs(w - np.round(w)).max(initial=0) > 1e-6 or M.max_violation(sol) > 1e-5:
            return None
        if (sol < base_lo - 1e-6).any():
            return None
        return sol

    # ------------------------------------------------------------------ one re-plan
    def replan(self, t0, t_next):
        t_start = time.perf_counter()
        deadline = t_start + self.budget
        t1 = self.window_end(t0)
        W = self.window_state(t0, t1)
        rec = dict(t0=t0, t1=t1, day=t0 // SLOTS_PER_DAY, n_veh=len(W.vehs))
        M = WindowModel(W, m=self.m, elastic=True)
        rec['stats'] = M.stats()
        base_lo, base_hi = M.lo.copy(), M.hi.copy()
        for q, d in enumerate(M.days):
            base_lo[M.u0 + q] = self.traj['rent'].get(int(d), 0)
        M.to_highs()
        self.cg = self.col_groups(M)
        rec['build_s'] = time.perf_counter() - t_start
        # ---- initial solution: rule-based plan and (optionally) the carried previous plan
        plan0 = self.construct(W, M)
        cands = [('construct', plan0)]
        if self.carry and self.prev is not None and self.policy == 'lns':
            cands += self.carry_plans(M, plan0)
        cands.append(('defer_all', dict(modes=[], trips=[], exits=[], crew={},
                                        rent={int(d): self.traj['rent'].get(int(d), 0) for d in M.days})))
        scored = []
        for name, pl in cands:
            sol = self.feasible_vector(M, pl, base_lo)
            rec['cand_' + name] = None if sol is None else [M.F(sol), M.Z(sol)]
            if sol is not None:
                scored.append(((M.F(sol), M.Z(sol)), name, sol))
        # carried plan completed by a partial MILP: kept decisions fixed, all other vehicles re-decided
        carried = [x for x in scored if x[1] == 'carry']
        if carried:
            cur = carried[0][2]
            vid2i = {v['vid']: i for i, v in enumerate(M.W.vehs)}
            free = {i for i in self.cg if M.W.vehs[i]['vid'] not in self._kept}
            lo, hi = self.bounds_fixing(M, cur, free, base_lo, base_hi)
            tl = max(0.5, min(5.0, 0.05 * self.budget))
            if M.F(cur) > 1e-6:
                M.set_stage(1)
                r = M.solve(time_limit=tl, start=cur, lo=lo, hi=hi)
                if r['x'] is not None and M.F(r['x']) < M.F(cur) - 1e-6:
                    cur = M.normalize(r['x'])
            M.set_stage(2, M.F(cur))
            r = M.solve(time_limit=tl, start=cur, lo=lo, hi=hi)
            if r['x'] is not None:
                c2 = M.normalize(r['x'])
                if self.better((M.F(c2), M.Z(c2)), (M.F(cur), M.Z(cur))):
                    cur = c2
            rec['cand_carry+repair'] = [M.F(cur), M.Z(cur)]
            scored.append(((M.F(cur), M.Z(cur)), 'carry+repair', cur))
        scored.sort(key=lambda x: (round(x[0][0], 6), x[0][1]))
        best = None
        for val, name, sol in scored:
            chk, _ = self.plan_check(M, sol, t0, t1)
            if chk['ok']:
                best, best_val, rec['init_source'] = sol.copy(), val, name
                break
            rec.setdefault('init_rejected', []).append([name, chk['violations'][:3]])
        rec['init_s'] = time.perf_counter() - t_start
        if best is None:
            rec['execution_stopped'] = True
            rec['stop_reason'] = 'no_validated_plan'
            rec['total_s'] = time.perf_counter() - t_start
            self.log.append(rec)
            raise ExecutionGateError('No validated plan at slot %s; history preserved' % t0)
        rec['init_F'], rec['init_obj'] = best_val
        rec['init_parts'] = {k: float(v) for k, v in M.cost_parts(best).items()}
        hist = [(round(rec['init_s'], 2), best_val[0], best_val[1])]
        it = acc = 0
        op_use = Counter(); op_acc = Counter()
        freeable = [i for i, c in self.cg.items() if c.size > 0]

        def lns(stage, ops, t_stop):
            nonlocal best, best_val, it, acc
            while time.perf_counter() < t_stop and it < self.max_iter:
                if stage == 1 and best_val[0] <= 1e-6:
                    break
                it += 1
                op = ops[self.rng.integers(len(ops))]
                op_use[op] += 1
                k = int(self.rng.integers(2, min(10, len(freeable)) + 1)) if len(freeable) >= 2 else 1
                pick = self.select(op, M, best, freeable, k)
                pick = self.trip_closure(M, best, pick)
                lo, hi = self.bounds_fixing(M, best, pick, base_lo, base_hi)
                tl = min(self.repair_limit, deadline - time.perf_counter() - 5)
                if tl < 0.5:
                    break
                r = M.solve(time_limit=tl, start=best, lo=lo, hi=hi)
                if r['x'] is None:
                    continue
                cand = M.normalize(r['x'])
                val = (M.F(cand), M.Z(cand))
                ok = val[0] < best_val[0] - 1e-6 if stage == 1 else self.better(val, best_val)
                if ok:
                    chk2, _ = self.plan_check(M, cand, t0, t1)
                    if chk2['ok']:
                        best, best_val = cand.copy(), val
                        acc += 1; op_acc[op] += 1
                        hist.append((round(time.perf_counter() - t_start, 2), val[0], val[1]))
                    else:
                        rec['rejected_by_check'] = rec.get('rejected_by_check', 0) + 1

        if self.policy == 'lns' and freeable:
            if best_val[0] > 1e-6:            # stage 1: minimise the excess waiting F
                M.set_stage(1)
                lns(1, ['random', 'overflow', 'related', 'trip'],
                    min(deadline - 6, t_start + self.stage1_limit))
            rec['stage1_F'], rec['stage1_iter'], rec['stage1_s'] = best_val[0], it, time.perf_counter() - t_start
            M.set_stage(2, best_val[0])       # stage 2: minimise Z subject to F <= F*
            lns(2, ['random', 'cost', 'related', 'trip'], deadline - 6)
        rec['op_use'] = dict(op_use); rec['op_acc'] = dict(op_acc)
        rec['lns_iter'] = it; rec['lns_acc'] = acc
        rec['final_F'], rec['final_obj'] = best_val
        rec['final_parts'] = {k: float(v) for k, v in M.cost_parts(best).items()}
        rec['history'] = hist
        rec['valid'] = True
        plan = M.decode(best)
        self.prev = dict(plan=plan, vinfo={v['vid']: (v['offset'], v['modes']) for v in M.W.vehs})
        # Independently validate the commitments made before the next replan.
        new = self.compose(M, plan, upto=t_next)
        prefix = checker.check(self.inst, new, t_end=t_next,
                               include=self.known_ids(t0), info_time=t0)
        rec['prefix_check'] = bool(prefix['ok'])
        if not prefix['ok']:
            rec['execution_stopped'] = True
            rec['stop_reason'] = 'invalid_execution_prefix'
            rec['prefix_violations'] = prefix['violations'][:10]
            rec['total_s'] = time.perf_counter() - t_start
            self.log.append(rec)
            raise ExecutionGateError('Invalid execution prefix at slot %s; history preserved' % t0)
        n_old = (len(self.traj['modes']), len(self.traj['trips']), len(self.traj['exits']))
        for rec_ in new['modes'][n_old[0]:]:
            self.traj['modes'].append(rec_); self.by_v[rec_[0]].append(('mode', rec_[3], rec_[1], rec_[2]))
        for rec_ in new['trips'][n_old[1]:]:
            self.traj['trips'].append(rec_); self.by_v[rec_[0]].append(('trip', rec_[2], self.trip_by_id[rec_[1]]))
        for rec_ in new['exits'][n_old[2]:]:
            self.traj['exits'].append(rec_); self.by_v[rec_[0]].append(('exit', rec_[1]))
        self.traj['crew'] = new['crew']
        for d, n in plan['rent'].items():
            if d * SLOTS_PER_DAY < t_next:
                self.traj['rent'][d] = max(self.traj['rent'].get(d, 0), n)
        rec['total_s'] = time.perf_counter() - t_start
        rec['maxrss_mb'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        self.log.append(rec)
        if self.verbose:
            print(f"day {rec['day']} t0={t0} n={rec['n_veh']} cols={rec['stats']['ncol']} "
                  f"build={rec['build_s']:.1f}s init={rec['init_s']:.1f}s src={rec['init_source']} "
                  f"F {rec['init_F']:.0f}->{best_val[0]:.0f} Z {rec['init_obj']:.0f}->{best_val[1]:.0f} "
                  f"it={it} acc={acc} total={rec['total_s']:.1f}s", flush=True)
        return rec

    def overflow_vehicles(self, M, sol):
        NB = M.NB
        blocks = np.nonzero(M.excess(sol) > 1e-6)[0]
        y = sol[M.y0:M.y0 + len(M.W.vehs) * 2 * NB].reshape(len(M.W.vehs), 2, NB)
        cand = set()
        for b in blocks[:20]:
            cand |= set(np.nonzero(y[:, 0, max(0, b - 1):b + 1].max(axis=1) > 0.5)[0].tolist())
        return [i for i in cand if i in self.cg]

    def trip_closure(self, M, sol, pick):
        zact = np.nonzero(sol[M.z0:M.z0 + M.zv.size] > 0.5)[0]
        by_trip = defaultdict(set)
        for j in zact:
            by_trip[(int(M.zk[j]), int(M.zs[j]))].add(int(M.zv[j]))
        changed = True
        pick = set(pick)
        while changed:
            changed = False
            for members in by_trip.values():
                if members & pick and not members <= pick:
                    pick |= members; changed = True
        return pick

    def select(self, op, M, sol, freeable, k):
        rng = self.rng
        if op == 'random' or len(freeable) <= k:
            return set(rng.choice(freeable, size=min(k, len(freeable)), replace=False).tolist())
        V = M.W.vehs
        if op == 'overflow':
            cand = self.overflow_vehicles(M, sol)
            if not cand:
                return set(rng.choice(freeable, size=min(k, len(freeable)), replace=False).tolist())
            return set(rng.choice(cand, size=min(k, len(cand)), replace=False).tolist())
        if op == 'cost':
            sc = np.zeros(len(V))
            xa = np.nonzero(sol[M.x0:M.x0 + M.xv.size] > 0.5)[0]
            for j in xa:
                sc[M.xv[j]] += M.cost[M.x0 + j]
            sc += M.cost[M.L0:M.L0 + len(V)] * sol[M.L0:M.L0 + len(V)]
            for j in np.nonzero(sol[M.o0:M.o0 + M.ov.size] > 0.5)[0]:
                sc[M.ov[j]] += M.ocost[j]
            cand = [i for i in np.argsort(-sc) if i in self.cg][:2 * k]
            return set(rng.choice(cand, size=min(k, len(cand)), replace=False).tolist())
        if op == 'related':
            seed = int(rng.choice(freeable))
            xa = np.nonzero(sol[M.x0:M.x0 + M.xv.size] > 0.5)[0]
            st = defaultdict(list)
            for j in xa:
                st[int(M.xv[j])].append(int(M.xs[j]))
            ref = st.get(seed, [M.t0])
            dist = {i: min(abs(a - b) for a in ref for b in st.get(i, [M.t1])) for i in freeable if i != seed}
            near = sorted(dist, key=dist.get)[:k - 1]
            return {seed, *near}
        if op == 'trip':
            zact = np.nonzero(sol[M.z0:M.z0 + M.zv.size] > 0.5)[0]
            if zact.size == 0:
                return set(rng.choice(freeable, size=min(k, len(freeable)), replace=False).tolist())
            j = int(rng.choice(zact))
            pick = {int(M.zv[j])}
            rest = [i for i in freeable if i not in pick]
            pick |= set(rng.choice(rest, size=min(k - 1, len(rest)), replace=False).tolist())
            return pick
        return set()

    # ------------------------------------------------------------------ main loop
    def run(self):
        times = self.replan_times()
        for q, t0 in enumerate(times):
            t_next = times[q + 1] if q + 1 < len(times) else self.t_obs
            self.replan(t0, t_next)
        chk = checker.check(self.inst, self.traj, t_end=self.t_obs)
        ev = checker.evaluate(self.inst, self.traj, chk, self.t_obs)
        return chk, ev
