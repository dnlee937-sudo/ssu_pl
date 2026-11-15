"""Independent check that the formulation of Chapter 3 (final version, eqs [1]-[16]) equals the implemented
MILP (model.py).  Tags such as '3.2' below are the equation numbers [2], [3], ... of the final chapter.

The paper model is rebuilt here row by row from the printed equations.  Only the candidate index
sets (T_ij, T^D_i, T_a and the trip participation candidates), i.e. the *inputs* of the model, are
taken from model.WindowModel so that both models share the same column layout.  No row of model.py
is reused.  Checks per instance and block length m:
  (a) the optimal / best solution of the implemented model is feasible in the paper model and has
      the same objective value;
  (b) the solution of the paper model is feasible in the implemented model and has the same value;
  (c) where both are solved to optimality, the optimal values coincide.
Run:  python3 paper_model_check.py  -> results/paper_model_check.json
"""
import json, os, sys, time
from collections import defaultdict
import numpy as np
import scipy.sparse as sp
import highspy

from instance import SLOT_MIN, SLOTS_PER_DAY, REST_SLOTS, INF, business_mask
from model import WindowModel

LOCS = ('IN', 'OUT')


class PaperModel:
    def __init__(self, M: WindowModel):
        W, m = M.W, M.m
        inst, t0, t1 = W.inst, W.t0, W.t1
        V = W.vehs
        # Independently evaluate the suffix DP printed in the 261101 definition.
        # Do not reuse the production helper or the production row coefficients.
        suffix_duration = []
        for v in V:
            n_tasks = len(v['tasks'])
            suffix = [0.0] * (n_tasks + 1)
            for k in range(n_tasks - 1, -1, -1):
                suffix[k] = min(md.p + suffix[md.g + 1]
                                for md in v['modes'] if md.f == k)
            suffix_duration.append(suffix)
        # V_ik: minimum direct cost of the remaining suffix (independent DP), used for c^rem_ik
        suffix_cost = []
        for v in V:
            n_tasks = len(v['tasks'])
            suffix = [0.0] * (n_tasks + 1)
            for k in range(n_tasks - 1, -1, -1):
                suffix[k] = min(md.cost + suffix[md.g + 1] for md in v['modes'] if md.f == k)
            suffix_cost.append(suffix)
        # p_ijt: internal work stops at the end of business and resumes at the next opening
        bm_ = business_mask(t1 + 30 * SLOTS_PER_DAY)
        STAFF_ = ('gen', 'mech', 'body', 'insp')
        _ws = {}

        def wslots(md, t):
            key = (id(md), t)
            if key not in _ws:
                if md.kind == 'O':
                    _ws[key] = list(range(t, t + md.p))
                else:
                    out, u = [], t
                    while len(out) < md.p:
                        if bm_[u]:
                            out.append(u)
                        u += 1
                    _ws[key] = out
            return _ws[key]

        pend = lambda md, t: wslots(md, t)[-1] + 1            # t + p_ijt

        def occupancy(md, t, prof, staff):
            ws, occ = wslots(md, t), {}
            for q_, val in enumerate(prof):
                if not val:
                    continue
                if staff or md.kind == 'O':
                    occ[ws[q_]] = occ.get(ws[q_], 0) + val
                else:                                         # bay / equipment stay occupied while interrupted
                    nxt = ws[q_ + 1] if q_ + 1 < len(ws) else ws[q_] + 1
                    for u in range(ws[q_], nxt):
                        occ[u] = val
            return occ
        self.M = M
        n = M.ncol
        rows = []                                      # (dict col->coef, lo, hi, tag)
        add = lambda d, lo, hi, tag: rows.append((d, lo, hi, tag))
        NB = M.NB
        blk = lambda t: (t - t0) // m                  # b(t)
        Tb = lambda b: range(t0 + b * m, min(t0 + (b + 1) * m, t1))
        dayof = lambda t: t // SLOTS_PER_DAY
        trips = {tt.kid: tt for tt in inst.trip_types}

        # ---- columns (same layout as model.py)
        X = [(int(M.xv[j]), int(M.xb[j]), int(M.xs[j]), M.x0 + j) for j in range(M.xv.size)]
        O = {(int(M.ov[j]), int(M.ok[j])): M.o0 + j for j in range(M.ov.size)}
        E = defaultdict(list)                          # i -> [(t, col)] incl. t1 (no handover)
        for j in range(M.ev.size):
            E[int(M.ev[j])].append((int(M.es[j]), M.e0 + j))
        for j, i in enumerate(M.ne_v):
            E[int(i)].append((t1, M.ne0 + j))
        Y = lambda i, l, b: M.y0 + (i * 2 + l) * NB + b
        Z = [(int(M.zv[j]), int(M.zk[j]), int(M.zs[j]), M.z0 + j) for j in range(M.zv.size)]
        Wc = {(int(M.wk[j]), int(M.ws[j])): M.w0 + j for j in range(M.wk.size)}
        U = {int(d): M.u0 + q for q, d in enumerate(M.days)}
        Tcol = lambda i: M.L0 + i
        S = lambda l, b: M.sg0 + l * NB + b
        tech = list(M.tech)
        CW = {}                                        # (k, t, g) -> column of the technician crew of trip (k, t)
        for a_, j_ in enumerate(M.cw_w):
            for gi_, g_ in enumerate(tech):
                CW[(int(M.wk[j_]), int(M.ws[j_]), g_)] = M.cw0 + a_ * len(tech) + gi_

        # ---- bounds / integrality (3.15) and the bounds of (3.10)
        lo = np.zeros(n); hi = np.ones(n); integ = np.ones(n, np.int8)
        for (k, t), c in Wc.items():
            hi[c] = max(1, len(V) // trips[k].q)       # trivial upper bound (not binding)
        for d, c in U.items():
            hi[c] = inst.rent_max                       # u^max_out ; u^0 = 0 in a static window
        for (k, t, g_), c in CW.items():
            hi[c] = (trips[k].q + 1) * hi[Wc[(k, t)]]
        for i in range(len(V)):
            hi[Tcol(i)] = np.inf; integ[Tcol(i)] = 0
        for l in (0, 1):
            for b in range(NB):
                hi[S(l, b)] = np.inf if l == 0 else 0; integ[S(l, b)] = 0   # E_b internal only
        late_v = [i for i, v in enumerate(V) if v['tasks'] and v['deadline'] < INF]
        for i in range(len(V)):
            if i not in late_v:
                hi[Tcol(i)] = 0

        # ---- objective (3.1)
        c = np.zeros(n)
        delta = SLOT_MIN
        for (i, j, t, col) in X:
            c[col] = V[i]['modes'][j].cost
        for (i, k), col in O.items():
            c[col] = suffix_cost[i][k] - suffix_cost[i][k + 1]               # c^rem_ik = V_ik - V_i,k+1
        for (k, t), col in Wc.items():
            c[col] = trips[k].cost
        for d, col in U.items():
            c[col] = inst.rent_cost_day
        for i in range(len(V)):
            c[Tcol(i)] = inst.lateness_cost_min
        for i, lst in E.items():
            rho = V[i]['pickup_req']
            for (t, col) in lst:
                c[col] = inst.pickup_wait_cost_min * delta * max(0, t - rho)
        for l in (0, 1):
            for b in range(NB):
                c[S(l, b)] = 0.0                        # E_b enters F, not Z
        # holding term c^hold * delta * (t^c_i - t0), t^c_i from eq. (3.14)
        ch = getattr(inst, 'hold_cost_day', 0.0) / (24 * 60)          # per minute
        if ch > 0:
            for i, v in enumerate(V):
                nk = len(v['tasks'])
                if nk == 0:
                    continue
                for (ii, j, t, col) in X:
                    md = V[ii]['modes'][j]
                    if ii == i and md.g == nk - 1:
                        c[col] += ch * delta * (pend(md, t) - v.get('hold_start', t0))
                    elif ii == i and pend(md, t) > t1:          # eq. [15], last term
                        c[col] += ch * delta * (pend(md, t) - t1)
                c[O[(i, nk - 1)]] += ch * delta * (t1 - v.get('hold_start', t0))
                for k in range(nk):
                    c[O[(i, k)]] += ch * delta * (suffix_duration[i][k] - suffix_duration[i][k + 1])
        self.c = c

        pmax = max([pend(V[i]['modes'][j], t) - t for (i, j, t, col) in X] + [md.p for v in V for md in v['modes']] + [1])
        # (3.2) cover
        for i, v in enumerate(V):
            for k in range(len(v['tasks'])):
                d = {O[(i, k)]: 1.0}
                for (ii, j, t, col) in X:
                    md = V[ii]['modes'][j]
                    if ii == i and md.f <= k <= md.g:
                        d[col] = d.get(col, 0) + 1
                add(d, 1, 1, '3.2')
        # (3.3) carry-over order, (3.4) precedence   (k < n_i)
        for i, v in enumerate(V):
            nk = len(v['tasks'])
            for k in range(nk - 1):
                add({O[(i, k)]: 1, O[(i, k + 1)]: -1}, -np.inf, 0, '3.3')
                d = defaultdict(float)
                for (ii, j, t, col) in X:
                    if ii != i:
                        continue
                    md = V[i]['modes'][j]
                    if md.g == k:
                        d[col] += pend(md, t)
                    if md.f == k + 1:
                        d[col] -= t
                d[O[(i, k + 1)]] -= (t1 + pmax)
                add(dict(d), -np.inf, 0, '3.4')
        # (3.5) delivery
        for i, lst in E.items():
            add({col: 1 for (t, col) in lst}, 1, 1, '3.5')

        # ---- v^arr, v^dep : lists of (col) per (i, l, t)
        arr = defaultdict(list); dep = defaultdict(list)
        for (i, j, t, col) in X:
            md = V[i]['modes'][j]
            dep[(i, 0, t)].append(col)                              # 1{l=in} x_ijt
            if pend(md, t) < t1:
                arr[(i, 0, pend(md, t))].append(col)                # 1{l=in} x_ijt', t' + p_ijt' = t
        for (i, lst) in E.items():
            for (t, col) in lst:
                if t < t1:
                    dep[(i, 0, t)].append(col)                      # 1{l=in} e_it
        for (i, k, t, col) in Z:
            tt = trips[k]
            dep[(i, LOCS.index(tt.src), t)].append(col)
            if t + tt.move < t1:
                arr[(i, LOCS.index(tt.dst), t + tt.move)].append(col)
        # eta and y^0
        eta = defaultdict(int); y0 = defaultdict(int)
        for i, v in enumerate(V):
            l = LOCS.index(v['entry_loc'])
            if v['entry_slot'] is None:
                y0[(i, l)] = 1
            elif t0 <= v['entry_slot'] < t1:
                eta[(i, l, v['entry_slot'])] = 1

        def yprev(d, i, l, b, coef):
            """add coef*y_{il,b-1} to row dict d; returns constant moved to the rhs"""
            if b >= 1:
                d[Y(i, l, b - 1)] = d.get(Y(i, l, b - 1), 0) + coef
                return 0.0
            return coef * y0[(i, l)]

        # (3.6) flow
        for i in range(len(V)):
            for l in (0, 1):
                for b in range(NB):
                    d = defaultdict(float)
                    d[Y(i, l, b)] += 1
                    const = -yprev(d, i, l, b, -1.0)       # y_ilb - y_il,b-1 - sum(v+ - v-) = sum eta
                    for t in Tb(b):
                        for col in arr[(i, l, t)]:
                            d[col] -= 1
                        for col in dep[(i, l, t)]:
                            d[col] += 1
                        const += eta[(i, l, t)]
                    add(dict(d), const, const, '3.6')
        # (3.7) order inside a block (m >= 2), t in T^dep_il
        if m >= 2:
            for (i, l, t) in sorted({key for key, v in dep.items() if v}):
                b = blk(t)
                d = defaultdict(float)
                rhs = -yprev(d, i, l, b, -1.0)
                for tp in Tb(b):
                    if tp > t:
                        break
                    for col in dep[(i, l, tp)]:
                        d[col] += 1
                    for col in arr[(i, l, tp)]:
                        d[col] -= 1
                    rhs += eta[(i, l, tp)]
                add(dict(d), -np.inf, rhs, '3.7')
        # (3.8)-(3.9) capacity with Y_ilb and G_lb
        Sres = {0: inst.park_cap['IN'] - np.asarray(W.fixed_park),
                1: np.full(t1 - t0, inst.park_cap['OUT']) - np.asarray(W.fixed_out_park)}
        for l in (0, 1):
            for b in range(NB):
                d = defaultdict(float)
                rhs = float(min(Sres[l][t - t0] for t in Tb(b)))
                for i in range(len(V)):
                    if m == 1:
                        d[Y(i, l, b)] += 1
                    else:
                        rhs -= yprev(d, i, l, b, 1.0)
                        for t in Tb(b):
                            for col in arr[(i, l, t)]:
                                d[col] += 1
                            rhs -= eta[(i, l, t)]
                if l == 0:                                          # G_in,b
                    tb = set(Tb(b))
                    for (i, j, t, col) in X:
                        md = V[i]['modes'][j]
                        if any(u in tb for u in occupancy(md, t, md.park, False)):
                            d[col] += 1
                    for (i, k, t, col) in Z:
                        tt = trips[k]
                        if tt.mode == 'tow':
                            hs = t if tt.src == 'IN' else t + tt.move - 1
                            if hs in tb:
                                d[col] += 1
                d[S(l, b)] -= 1
                if l == 1:
                    # Each date intersecting this half-open block needs enough
                    # rented spaces, including at midnight and the last block.
                    for day in sorted({dayof(t) for t in Tb(b)}):
                        dated = dict(d)
                        dated[U[day]] = dated.get(U[day], 0) - 1
                        add(dated, -np.inf, rhs, '3.9')
                else:
                    add(dict(d), -np.inf, rhs, '3.9')
        # (3.11) retrieval precondition
        for (i, k, t, col) in Z:
            tt = trips[k]
            if tt.src != 'OUT':
                continue
            d = defaultdict(float)
            for (i2, k2, t2, col2) in Z:
                if i2 == i and t2 == t and trips[k2].src == 'OUT' and trips[k2].pre == tt.pre:
                    d[col2] += 1
                    # 𝒜^dep_out(a)
            bb = (t - tt.pre - t0) // m                     # b(t - p^pre)
            if bb - 1 >= 0:
                d[Y(i, 1, bb - 1)] -= 1
                rhs = 0.0
            else:
                rhs = float(y0[(i, 1)])
            add(dict(d), -np.inf, rhs, '3.11')
        # (3.12) trip link
        for (k, t), colw in Wc.items():
            d = {colw: -trips[k].q}
            for (i, k2, t2, col) in Z:
                if k2 == k and t2 == t:
                    d[col] = d.get(col, 0) + 1
            add(d, 0, 0, '3.12')
        # (3.13) resources, beta profiles written out from the text of 3.4.9
        use = defaultdict(lambda: defaultdict(float))       # (r, t) -> col -> coef
        for (i, j, t, col) in X:
            md = V[i]['modes'][j]
            for r, prof in md.res.items():
                for u, val in occupancy(md, t, prof, r in STAFF_).items():
                    if u < t1:
                        use[(r, u)][col] += val
        for (k, t), col in Wc.items():
            tt = trips[k]
            if tt.mode == 'staff':
                for tp in range(t - tt.pre, t + tt.back + REST_SLOTS):
                    use[('gen', tp)][col] += tt.q + 1
                    for g_ in tech:                     # technicians replace general staff one for one
                        use[('gen', tp)][CW[(k, t, g_)]] -= 1
                        use[(g_, tp)][CW[(k, t, g_)]] += 1
                for tp in range(t - tt.pre, t + tt.back):
                    use[('supportcar', tp)][col] += 1
            else:
                for tp in range(t, t + tt.move):
                    use[('sup_tow', tp)][col] += 1
                hs = t if tt.src == 'IN' else t + tt.move - 1
                use[('gen', hs)][col] += 1
        for r in sorted(inst.caps):
            Q = inst.cap_series(r, t0, t1) - W.fixed_res.get(r, np.zeros(t1 - t0))
            for t in range(t0, t1):
                d = dict(use.get((r, t), {}))
                if d:
                    add(d, -np.inf, float(Q[t - t0]), '3.13')
                else:
                    assert Q[t - t0] >= 0
        # crew of a staff trip: technicians taking part cannot exceed the persons needed
        for (k, t), colw in Wc.items():
            if trips[k].mode == 'staff' and tech:
                d = {CW[(k, t, g_)]: 1.0 for g_ in tech}
                d[colw] = -(trips[k].q + 1)
                add(d, -np.inf, 0, 'crew')
        # (3.14) lateness
        for i in late_v:
            v = V[i]; nk = len(v['tasks'])
            d = defaultdict(float)
            d[Tcol(i)] += 1
            for (ii, j, t, col) in X:
                md = V[ii]['modes'][j]
                if ii == i and md.g == nk - 1:
                    d[col] -= delta * pend(md, t)
                elif ii == i and pend(md, t) > t1:
                    d[col] -= delta * (pend(md, t) - t1)
            d[O[(i, nk - 1)]] -= delta * t1
            for k in range(nk):
                pr = suffix_duration[i][k] - suffix_duration[i][k + 1]
                d[O[(i, k)]] -= delta * pr
            add(dict(d), -delta * v['deadline'], np.inf, '3.14')

        # F = delta * sum_b |T_b| E_b   (eq. [1]); row used as F <= F* in the second stage
        self.cF = np.zeros(n)
        for b in range(NB):
            self.cF[S(0, b)] = delta * len(Tb(b))
        add({S(0, b): delta * len(Tb(b)) for b in range(NB)}, -np.inf, np.inf, 'F')
        self.frow = len(rows) - 1
        self.rows = rows
        R, C, Vv = [], [], []
        for r, (d, _, _, _) in enumerate(rows):
            for col, val in d.items():
                if val != 0:
                    R.append(r); C.append(col); Vv.append(val)
        self.A = sp.csr_matrix((Vv, (R, C)), shape=(len(rows), n))
        self.rlo = np.array([r[1] for r in rows], float); self.rhi = np.array([r[2] for r in rows], float)
        self.tags = np.array([r[3] for r in rows])
        self.lo, self.hi, self.integ = lo, hi, integ

    def violation(self, x):
        ax = self.A @ x
        v = np.maximum(self.rlo - ax, 0) + np.maximum(ax - self.rhi, 0)
        vb = np.maximum(self.lo - x, 0) + np.maximum(x - self.hi, 0)
        worst = {}
        for tag in np.unique(self.tags):
            worst[str(tag)] = float(v[self.tags == tag].max())
        return float(max(v.max(), vb.max())), worst

    def solve(self, time_limit=60, stage=2, fcap=None):
        h = highspy.Highs(); h.setOptionValue('output_flag', False); h.setOptionValue('threads', 1)
        A = self.A.tocsc()
        lp = highspy.HighsLp()
        lp.num_col_, lp.num_row_ = A.shape[1], A.shape[0]
        lp.col_cost_ = self.cF if stage == 1 else self.c; lp.col_lower_ = self.lo; lp.col_upper_ = np.where(np.isinf(self.hi), highspy.kHighsInf, self.hi)
        lp.row_lower_ = np.where(np.isinf(self.rlo), -highspy.kHighsInf, self.rlo)
        rhi_ = self.rhi.copy()
        if fcap is not None:
            rhi_[self.frow] = fcap
        lp.row_upper_ = np.where(np.isinf(rhi_), highspy.kHighsInf, rhi_)
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_ = A.indptr; lp.a_matrix_.index_ = A.indices; lp.a_matrix_.value_ = A.data
        lp.integrality_ = [highspy.HighsVarType.kInteger if q else highspy.HighsVarType.kContinuous for q in self.integ]
        h.passModel(lp)
        h.setOptionValue('time_limit', float(time_limit)); h.setOptionValue('mip_rel_gap', 0.0)
        t = time.perf_counter(); h.run(); el = time.perf_counter() - t
        info = h.getInfo()
        has = info.primal_solution_status == 2
        return dict(status=h.modelStatusToString(h.getModelStatus()), time=el,
                    x=np.array(h.getSolution().col_value) if has else None,
                    obj=info.objective_function_value if has else None)


def code_violation(M, x):
    ax = M.A @ x
    v = np.maximum(M.rlo - ax, 0) + np.maximum(ax - M.rhi, 0)
    vb = np.maximum(M.lo - x, 0) + np.maximum(x - M.hi, 0)
    return float(max(v.max(), vb.max()))


def compare(W, m, tl=120, elastic=False, label=''):
    M = WindowModel(W, m=m, elastic=elastic)
    P = PaperModel(M)
    assert np.allclose(P.c, M.cost), 'objective coefficients differ'
    rc = M.solve(time_limit=tl)
    rp = P.solve(time_limit=tl)
    out = dict(label=label, m=m, ncol=M.ncol, rows_code=M.nrow, rows_paper=len(P.rows),
               code=dict(status=rc['status'], obj=rc['obj'], time=round(rc['time'], 1)),
               paper=dict(status=rp['status'], obj=rp['obj'], time=round(rp['time'], 1)),
               objective_coefficients_equal=True)
    if rc['x'] is not None:
        xr = np.round(rc['x'])                               # integral vector
        xr[M.L0:M.L0 + len(W.vehs)] = rc['x'][M.L0:M.L0 + len(W.vehs)]
        xr[M.sg0:M.sg0 + 2 * M.NB] = rc['x'][M.sg0:M.sg0 + 2 * M.NB]
        mv, worst = P.violation(xr)
        out['code_solution_in_paper_model'] = dict(max_violation=mv, worst_by_eq={k: v for k, v in worst.items() if v > 1e-6},
                                                   objective=float(P.c @ xr))
    if rp['x'] is not None:
        xp = np.round(rp['x'])
        xp[M.L0:M.L0 + len(W.vehs)] = rp['x'][M.L0:M.L0 + len(W.vehs)]
        xp[M.sg0:M.sg0 + 2 * M.NB] = rp['x'][M.sg0:M.sg0 + 2 * M.NB]
        out['paper_solution_in_code_model'] = dict(max_violation=code_violation(M, xp), objective=float(M.cost @ xp))
    both_opt = rc['status'] == 'Optimal' and rp['status'] == 'Optimal'
    out['both_optimal'] = both_opt
    if both_opt:
        out['optimal_values_equal'] = abs(rc['obj'] - rp['obj']) <= 1e-6 * max(1, abs(rc['obj']))
    return out


if __name__ == '__main__':
    sys.argv = sys.argv[:1]
    from instance import (draw_tasks, add_business_days, generate, Instance, Vehicle, build_modes,
                          make_trip_types, DEFAULT_CAPS, STAFF, pickup_slot)
    from model import Window

    def hm(day, h, mi=0):
        return day * SLOTS_PER_DAY + (h * 60 + mi) // SLOT_MIN

    def make(vehicles, days=7, caps=None, park_in=12, park_out=18, rent_max=0):
        c = dict(DEFAULT_CAPS); c.update(caps or {})
        return Instance(days * SLOTS_PER_DAY, vehicles, c, set(STAFF), dict(IN=park_in, OUT=park_out),
                        rent_max, 5000.0, make_trip_types(), name='case')

    def veh(vid, arrival, tasks, deadline, **kw):
        return Vehicle(vid, arrival, kw.pop('init_loc', None), tasks, build_modes(tasks) if tasks else [],
                       deadline, **kw)

    def static_window(inst, t0, t1):          # same as e1_cases.static_window
        vehs = []
        for v in inst.vehicles:
            es = None if v.arrival is None else v.arrival
            vehs.append(dict(vid=v.vid, entry_slot=es if es is None or es > t0 else None,
                             entry_loc=v.init_loc or 'IN', tasks=v.tasks, modes=v.modes,
                             hold=v.approval_at or t0, deadline=v.deadline,
                             pickup_req=pickup_slot(v, 0) if not v.tasks else None, offset=0))
        return Window(inst, t0, t1, vehs, {}, np.zeros(t1 - t0, int))
    from simulate import Rolling
    res = []
    # 1) boundary case C6a (A leaves at 09:00, B arrives 09:00..09:15, IN capacity 1)
    for arr_min in (0, 5, 10):
        a = veh(0, None, [], 0, init_loc='IN', response_delay=0, done=1)
        b = veh(1, hm(1, 9, arr_min), ['wash', 'insp'], hm(3, 18), response_delay=12)
        inst = make([a, b], park_in=1, park_out=0, rent_max=0, days=3, caps=dict(supportcar=0, sup_tow=0))
        W = static_window(inst, hm(1, 8), hm(3, 0)); W.vehs[0]['pickup_req'] = hm(1, 9)
        for m in (1, 2, 3):
            res.append(compare(W, m, tl=30, label=f'C6a_B_09:{arr_min:02d}'))
            print(res[-1], flush=True)
    # 2) C7: 6 vehicles, 3-day window, trips, rental
    rng = np.random.default_rng(7)
    vs = []
    for i in range(6):
        tasks = draw_tasks(rng)
        arr = hm(0, 9 + int(rng.integers(0, 6)))
        vs.append(veh(i, arr, tasks, add_business_days(arr, 2), response_delay=12))
    inst = make(vs, park_in=3, park_out=4, rent_max=2, days=3, caps=dict(gen=3, mech=1, body=1, insp=1))
    W = static_window(inst, hm(0, 8), hm(3, 0))
    for m in (1, 2, 3):
        res.append(compare(W, m, tl=180, label='C7_6veh_3day'))
        print(res[-1], flush=True)
    # 3) a rolling window with carried-over state (OUT inventory, in-progress work, fixed usage)
    inst = generate(10, 1)
    R = Rolling(inst, policy='construct', m=1, budget=20)
    times = R.replan_times()
    for q in range(3):
        R.replan(times[q], times[q + 1])
    t0 = times[3]
    W = R.window_state(t0, t0 + R.window)
    W.inst = inst
    for m in (1, 2, 3):
        res.append(compare(W, m, tl=60, elastic=True, label=f'rolling_daily10_t0={t0}'))
        print(res[-1], flush=True)
    json.dump(res, open(os.path.join(os.path.dirname(__file__), 'results', 'paper_model_check.json'), 'w'),
              ensure_ascii=False, indent=1, default=str)
