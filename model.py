"""MILP builder for one planning window of the inventory-carryover reconditioning model.

Decisions (absolute 5-minute slots, window [t0, t1)):
  x[i,b,s]  start mode b of vehicle i at slot s                     (binary)
  o[i,k]    remaining task k of vehicle i is deferred beyond t1     (binary)
  e[i,s]    vehicle i is physically released at s (pickup request known)   (binary)
  ne[i]     released vehicle not handed over inside the window      (binary)
  y[i,l,b]  vehicle i is waiting at parking l at the END of parking block b (binary)
  z[i,k,s]  vehicle i joins trip type k departing at s              (binary)
  w[k,s]    number of trips of type k departing at s                (integer)
  u[d]      extra OUT spaces rented on calendar day d               (integer)
  L[i]      shop-responsible lateness of completion (minutes)       (continuous)
  sg[l,b]   parking overflow slack (only in the elastic phase)      (continuous)

Parking representation: block length m slots (m=1 is the exact 5-minute model).  Work, trips,
handovers, resources and lateness always stay on the 5-minute grid.  For m>1 the capacity row of
a block uses the conservative bound  y[i,l,b-1] + (arrivals of i at l inside block b), which is an
upper bound on the real occupancy at every slot of the block; per-vehicle guard rows keep the
order of in/out events inside a block.  Every solution of the m>1 model is therefore feasible
for the 5-minute model (soundness), but some 5-minute schedules are excluded.
"""
from __future__ import annotations
import time
import numpy as np
import scipy.sparse as sp
import highspy

from instance import (SLOT_MIN, SLOTS_PER_DAY, REST_SLOTS, INF, business_mask,
                      remaining_relaxation, mode_span, mode_profile)

LOC = {'IN': 0, 'OUT': 1}


class ColBlock:
    def __init__(self, start, n):
        self.start, self.n = start, n


class Window:
    """Everything the builder needs: the frozen state at t0 and the instance data."""

    def __init__(self, inst, t0, t1, vehs, fixed_res, fixed_park, fixed_out_park=None):
        self.inst, self.t0, self.t1 = inst, t0, t1
        self.vehs = vehs              # list of dict (see simulate.window_state)
        self.fixed_res = fixed_res    # r -> np.array(t1-t0) usage already committed
        self.fixed_park = fixed_park  # np.array(t1-t0) IN handover occupancy already committed
        self.fixed_out_park = np.zeros(t1 - t0, int) if fixed_out_park is None else fixed_out_park


class WindowModel:
    def __init__(self, W: Window, m: int = 1, elastic: bool = False, prune_preentry: bool = False,
                 sigma_cost: float = 0.0):
        t_build = time.perf_counter()
        self.W, self.m, self.elastic = W, m, elastic
        inst, t0, t1 = W.inst, W.t0, W.t1
        self.t0, self.t1 = t0, t1
        T = t1 - t0
        NB = (T + m - 1) // m
        self.NB = NB
        bm = business_mask(t1 + 4 * SLOTS_PER_DAY)
        self.bm = bm
        V = W.vehs
        nV = len(V)
        # Eq. (3.14): P[k] is the minimum duration of the entire task suffix.
        # Its adjacent differences telescope over a deferred suffix. A bundled
        # alternative is therefore counted once, never once per covered task.
        _rel = [remaining_relaxation(v['tasks'], v['modes']) for v in V]
        self.remaining_cost = [r[0] for r in _rel]
        self.remaining_duration = [r[1] for r in _rel]
        self.p_rem = [p[:-1] - p[1:] for p in self.remaining_duration]
        cost, lo, hi, integ = [], [], [], []
        ncol = 0

        def addcols(n, c, l, u, it):
            nonlocal ncol
            s = ncol
            cost.append(np.broadcast_to(np.asarray(c, float), (n,)).copy())
            lo.append(np.broadcast_to(np.asarray(l, float), (n,)).copy())
            hi.append(np.broadcast_to(np.asarray(u, float), (n,)).copy())
            integ.append(np.full(n, it, dtype=np.int8))
            ncol += n
            return s

        blk = lambda t: (t - t0) // m           # block index of absolute slot
        trips = inst.trip_types
        Pmax = max([md.p for v in V for md in v['modes']] + [1])

        # ------------------------------------------------------------ x : mode starts
        xv, xb, xs = [], [], []
        for i, v in enumerate(V):
            ready = max(t0, v['entry_slot'] if v['entry_slot'] is not None else t0, v['hold'])
            for bi, md in enumerate(v['modes']):
                if ready >= t1:
                    continue
                g = md.grid
                first = ready + ((-ready) % g)
                cand = np.arange(first, t1, g)
                if cand.size == 0:
                    continue
                # staff-requiring relative slots must lie in business hours
                ss = md.staff_slots
                ok = np.ones(cand.size, bool)
                if md.kind != 'O':
                    ok &= bm[cand]          # internal work: start in business hours, may resume next day
                elif ss is not None and ss.size:
                    ok &= bm[cand[:, None] + ss[None, :]].all(axis=1)
                cand = cand[ok]
                xv.append(np.full(cand.size, i)); xb.append(np.full(cand.size, bi)); xs.append(cand)
        xv = np.concatenate(xv) if xv else np.zeros(0, int)
        xb = np.concatenate(xb) if xb else np.zeros(0, int)
        xs = np.concatenate(xs) if xs else np.zeros(0, int)
        xc = np.array([V[i]['modes'][b].cost for i, b in zip(xv, xb)], float)
        self.x0 = addcols(xv.size, xc, 0, 1, 1)
        self.xv, self.xb, self.xs = xv, xb, xs
        # p_ijt: start-dependent duration (interruptions outside business hours included)
        xp = np.array([mode_span(V[i]['modes'][b], s) for i, b, s in zip(xv, xb, xs)], int)
        self.xp = xp
        Pmax = int(max(Pmax, xp.max(initial=1)))

        # ------------------------------------------------------------ o : deferral of remaining tasks
        ov, ok_ = [], []
        ocost = []
        for i, v in enumerate(V):
            for k in range(len(v['tasks'])):
                ov.append(i); ok_.append(k)
                # C2: increment of the minimum direct cost of the remaining suffix (V_ik - V_i,k+1)
                ocost.append(float(self.remaining_cost[i][k] - self.remaining_cost[i][k + 1]))
        self.ov, self.ok = np.array(ov, int), np.array(ok_, int)
        self.o0 = addcols(len(ov), ocost, 0, 1, 1)
        self.ocost = np.array(ocost, float)
        oidx = {(i, k): self.o0 + j for j, (i, k) in enumerate(zip(ov, ok_))}
        self.oidx = oidx

        # ------------------------------------------------------------ e / ne : physical release
        ev, es, ec = [], [], []
        ne_v = []
        cq = inst.pickup_wait_cost_min * SLOT_MIN
        for i, v in enumerate(V):
            if v['pickup_req'] is None or v['tasks']:
                continue
            req = max(v['pickup_req'], t0, v['entry_slot'] if v['entry_slot'] is not None else t0)
            if req >= t1:
                continue
            cand = np.arange(req + ((-req) % 6), t1, 6)
            cand = cand[bm[cand]]
            if req not in cand and bm[req]:
                cand = np.r_[req, cand]
            ev.append(np.full(cand.size, i)); es.append(cand)
            ec.append(cq * (cand - v['pickup_req']).clip(0))
            ne_v.append(i)
        ev = np.concatenate(ev) if ev else np.zeros(0, int)
        es = np.concatenate(es) if es else np.zeros(0, int)
        ec = np.concatenate(ec) if ec else np.zeros(0)
        self.e0 = addcols(ev.size, ec, 0, 1, 1)
        self.ev, self.es = ev, es
        ne_cost = [cq * (t1 - V[i]['pickup_req']) for i in ne_v]
        self.ne0 = addcols(len(ne_v), ne_cost, 0, 1, 1)
        self.ne_v = np.array(ne_v, int)

        # ------------------------------------------------------------ y : waiting state per block
        ny = nV * 2 * NB
        self.y0 = addcols(ny, 0, 0, 1, 1)
        yid = lambda i, l, b: self.y0 + (i * 2 + l) * NB + b

        # ------------------------------------------------------------ z / w : trips
        tk, ts = [], []
        for tt in trips:
            g = 6
            first = t0 + ((-t0) % g)
            cand = np.arange(first, t1, g)
            if tt.mode == 'staff':
                a, b_ = cand - tt.pre, cand + tt.back + REST_SLOTS
            else:
                a, b_ = cand, cand + tt.move
            okc = (a >= t0) & (b_ <= t1)
            cand, a, b_ = cand[okc], a[okc], b_[okc]
            okb = np.array([bm[x:y].all() for x, y in zip(a, b_)], bool) if a.size else np.zeros(0, bool)
            cand = cand[okb]
            tk.append(np.full(cand.size, tt.kid)); ts.append(cand)
        tk = np.concatenate(tk); ts = np.concatenate(ts)
        self.wk, self.ws = tk, ts
        wcost = np.array([trips[k].cost for k in tk], float)
        wmax = np.array([max(1, nV // trips[k].q) for k in tk], float)
        self.w0 = addcols(tk.size, wcost, 0, wmax, 1)
        widx = {(k, s): self.w0 + j for j, (k, s) in enumerate(zip(tk, ts))}
        # crew of staff trips: cw[w, g] technicians of group g taking part in the trips w (rest: general staff)
        self.tech = [g for g in ('mech', 'body', 'insp') if g in inst.caps]
        self.cw_w = np.array([j for j, k in enumerate(tk) if trips[k].mode == 'staff'], int)
        nt_ = len(self.tech)
        cwhi = np.repeat(np.array([(trips[tk[j]].q + 1) * wmax[j] for j in self.cw_w], float), nt_) \
            if self.cw_w.size and nt_ else np.zeros(0)
        self.cw0 = addcols(self.cw_w.size * nt_, 0, 0, cwhi if cwhi.size else 0, 1)
        self.cw_pos = {int(j): a for a, j in enumerate(self.cw_w)}
        zv, zk, zs = [], [], []
        for i, v in enumerate(V):
            ent = v['entry_slot'] if v['entry_slot'] is not None else t0
            sel = ts >= ent
            zv.append(np.full(sel.sum(), i)); zk.append(tk[sel]); zs.append(ts[sel])
        zv = np.concatenate(zv) if zv else np.zeros(0, int)
        zk = np.concatenate(zk) if zk else np.zeros(0, int)
        zs = np.concatenate(zs) if zs else np.zeros(0, int)
        self.z0 = addcols(zv.size, 0, 0, 1, 1)
        self.zv, self.zk, self.zs = zv, zk, zs

        # ------------------------------------------------------------ u : rented OUT spaces per day
        days = np.arange(t0 // SLOTS_PER_DAY, (t1 - 1) // SLOTS_PER_DAY + 1)
        self.days = days
        self.block_days = [tuple(range((t0 + b * m) // SLOTS_PER_DAY,
                                       (min(t1, t0 + (b + 1) * m) - 1) // SLOTS_PER_DAY + 1))
                           for b in range(NB)]
        self.u0 = addcols(days.size, inst.rent_cost_day, 0, inst.rent_max, 1)
        # ------------------------------------------------------------ L : lateness (minutes)
        self.L0 = addcols(nV, inst.lateness_cost_min, 0, np.inf, 0)
        # ------------------------------------------------------------ sigma : overflow slack
        # sg[IN,b] = E_b (internal excess waiting, always allowed); sg[OUT,b] is fixed to 0
        self.sg0 = addcols(2 * NB, sigma_cost, 0, np.r_[np.full(NB, np.inf), np.zeros(NB)], 0)
        self.Fw = SLOT_MIN * np.array([min(t1, t0 + (b + 1) * m) - (t0 + b * m) for b in range(NB)], float)
        self.ncol = ncol

        # ================================================================== rows
        R, C, Vv, rlo, rhi = [], [], [], [], []
        nrow = 0

        def addrows(n, l, u):
            nonlocal nrow
            s = nrow
            rlo.append(np.broadcast_to(np.asarray(l, float), (n,)).copy())
            rhi.append(np.broadcast_to(np.asarray(u, float), (n,)).copy())
            nrow += n
            return s

        def ent(r, c, v):
            R.append(np.asarray(r, np.int64).ravel()); C.append(np.asarray(c, np.int64).ravel())
            Vv.append(np.broadcast_to(np.asarray(v, float), np.asarray(r).shape).ravel().copy())

        xcols = self.x0 + np.arange(xv.size)
        self.row_groups = {}
        # (1) task cover:  sum x(b covers k) + o[i,k] = 1
        r0 = addrows(len(ov), 1, 1)
        self.row_groups['cover'] = (r0, len(ov))
        cover_row = {(i, k): r0 + j for j, (i, k) in enumerate(zip(ov, ok_))}
        ent(np.arange(r0, r0 + len(ov)), self.o0 + np.arange(len(ov)), 1)
        fx = np.array([V[i]['modes'][b].f for i, b in zip(xv, xb)], int)
        gx = np.array([V[i]['modes'][b].g for i, b in zip(xv, xb)], int)
        for k in range(int(gx.max()) + 1 if gx.size else 0):
            sel = (fx <= k) & (gx >= k)
            rr = [cover_row[(i, k)] for i in xv[sel]]
            ent(rr, xcols[sel], 1)
        # (2) deferral monotone  o[i,k] - o[i,k+1] <= 0
        pairs = [(oidx[(i, k)], oidx[(i, k + 1)]) for (i, k) in oidx if (i, k + 1) in oidx]
        r0 = addrows(len(pairs), -np.inf, 0)
        self.row_groups['defer'] = (r0, len(pairs))
        if pairs:
            pa = np.array(pairs)
            ent(np.arange(r0, r0 + len(pairs)), pa[:, 0], 1)
            ent(np.arange(r0, r0 + len(pairs)), pa[:, 1], -1)
        # (3) precedence  sum_{g=k}(s+p)x - sum_{f=k+1} s x - (t1+Pmax) o[i,k+1] <= 0
        bnd = [(i, k) for (i, k) in oidx if (i, k + 1) in oidx]
        r0 = addrows(len(bnd), -np.inf, 0)
        self.row_groups['prec'] = (r0, len(bnd))
        prow = {ik: r0 + j for j, ik in enumerate(bnd)}
        if bnd:
            sel = np.array([(i, g) in prow for i, g in zip(xv, gx)], bool)
            ent([prow[(i, g)] for i, g in zip(xv[sel], gx[sel])], xcols[sel], (xs + xp)[sel])
            sel = np.array([(i, f - 1) in prow for i, f in zip(xv, fx)], bool)
            ent([prow[(i, f - 1)] for i, f in zip(xv[sel], fx[sel])], xcols[sel], -xs[sel])
            ent([prow[ik] for ik in bnd], [oidx[(ik[0], ik[1] + 1)] for ik in bnd], -(t1 + Pmax))
        # (4) release: sum e + ne = 1
        r0 = addrows(len(self.ne_v), 1, 1)
        self.row_groups['release'] = (r0, len(self.ne_v))
        rel_row = {i: r0 + j for j, i in enumerate(self.ne_v)}
        if len(self.ne_v):
            ent([rel_row[i] for i in ev], self.e0 + np.arange(ev.size), 1)
            ent(np.arange(r0, r0 + len(self.ne_v)), self.ne0 + np.arange(len(self.ne_v)), 1)

        # ---- in / out event lists per (vehicle, loc): (col, slot, sign)  sign +1 = in, -1 = out
        # mode start: out of IN at s ; mode end: into IN at s+p (if inside window)
        ev_col, ev_veh, ev_loc, ev_t, ev_in = [], [], [], [], []

        def add_ev(cols, vehs, loc, ts_, isin):
            cols = np.asarray(cols); keep = (ts_ >= t0) & (ts_ < t1)
            ev_col.append(cols[keep]); ev_veh.append(np.asarray(vehs)[keep])
            ev_loc.append(np.full(keep.sum(), loc)); ev_t.append(np.asarray(ts_)[keep])
            ev_in.append(np.full(keep.sum(), isin))
        add_ev(xcols, xv, 0, xs, False)
        add_ev(xcols, xv, 0, xs + xp, True)
        add_ev(self.e0 + np.arange(ev.size), ev, 0, es, False)
        zcols = self.z0 + np.arange(zv.size)
        zsrc = np.array([LOC[trips[k].src] for k in zk], int)
        zdst = np.array([LOC[trips[k].dst] for k in zk], int)
        zmove = np.array([trips[k].move for k in zk], int)
        zpre = np.array([trips[k].pre for k in zk], int)
        for l in (0, 1):
            s1 = zsrc == l
            add_ev(zcols[s1], zv[s1], l, zs[s1], False)
            s2 = zdst == l
            add_ev(zcols[s2], zv[s2], l, (zs + zmove)[s2], True)
        ev_col = np.concatenate(ev_col); ev_veh = np.concatenate(ev_veh)
        ev_loc = np.concatenate(ev_loc); ev_t = np.concatenate(ev_t); ev_in = np.concatenate(ev_in)
        ev_blk = (ev_t - t0) // m
        self.events = (ev_col, ev_veh, ev_loc, ev_t, ev_in)

        # constant entries (arrivals / returns from in-progress modes or trips) and initial state
        const_in = np.zeros((nV, 2, NB))
        y_init = np.zeros((nV, 2))
        for i, v in enumerate(V):
            if v['entry_slot'] is None:          # waiting at t0
                y_init[i, LOC[v['entry_loc']]] = 1
            elif t0 <= v['entry_slot'] < t1:
                const_in[i, LOC[v['entry_loc']], blk(v['entry_slot'])] += 1
        self.y_init, self.const_in = y_init, const_in

        # (5) flow rows: y[b] - y[b-1] + sum out - sum in(var) = const_in[b] + [b==0] y_init
        r0 = addrows(ny, 0, 0)
        self.row_groups['flow'] = (r0, ny)
        rhs = const_in.copy(); rhs[:, :, 0] += y_init
        rlo[-1][:] = rhs.ravel(); rhi[-1][:] = rhs.ravel()
        fr = lambda i, l, b: r0 + (i * 2 + l) * NB + b
        ent(np.arange(r0, r0 + ny), self.y0 + np.arange(ny), 1)
        ii, ll, bb = np.meshgrid(np.arange(nV), [0, 1], np.arange(1, NB), indexing='ij')
        ent(fr(ii, ll, bb), yid(ii, ll, bb - 1), -1)
        ent(fr(ev_veh, ev_loc, ev_blk), ev_col, np.where(ev_in, -1.0, 1.0))

        # (6) guard rows (m>1): cumulative out up to t within block <= y[b-1] + cumulative in up to t
        if m > 1:
            outmask = ~ev_in
            keys = np.unique(np.stack([ev_veh[outmask], ev_loc[outmask], ev_t[outmask]], 1), axis=0)
            gr0 = addrows(len(keys), -np.inf, 0)
            self.row_groups['guard'] = (gr0, len(keys))
            # rhs: const in within [block start, t] + init (b==0)
            rhsg = np.zeros(len(keys))
            order = np.lexsort((ev_t, ev_loc, ev_veh))
            evs = (ev_veh[order], ev_loc[order], ev_t[order], ev_col[order], ev_in[order], ev_blk[order])
            # map (veh,loc) -> slice of events
            from collections import defaultdict
            byvl = defaultdict(list)
            for j in range(order.size):
                byvl[(evs[0][j], evs[1][j])].append(j)
            rows_g, cols_g, vals_g = [], [], []
            for gi, (i, l, t) in enumerate(keys):
                b = (t - t0) // m
                bs = t0 + b * m
                for j in byvl[(i, l)]:
                    tj = evs[2][j]
                    if bs <= tj <= t:
                        rows_g.append(gr0 + gi); cols_g.append(evs[3][j])
                        vals_g.append(-1.0 if evs[4][j] else 1.0)
                if b >= 1:
                    rows_g.append(gr0 + gi); cols_g.append(yid(i, l, b - 1)); vals_g.append(-1.0)
                    rhsg[gi] = 0
                else:
                    rhsg[gi] = y_init[i, l]
                # constant entries of i at l within [bs, t]
                es_ = V[i]['entry_slot']
                if es_ is not None and LOC[V[i]['entry_loc']] == l and bs <= es_ <= t:
                    rhsg[gi] += 1
            rhi[-1][:] = rhsg
            ent(rows_g, cols_g, vals_g)

        # (7) parking capacity per (loc, block)
        cr0 = addrows(2 * NB, -np.inf, 0)
        self.row_groups['cap'] = (cr0, 2 * NB)
        cap_entries_start = len(R)
        crow = lambda l, b: cr0 + l * NB + b
        caprhs = np.zeros((2, NB))
        caprhs[0, :] = inst.park_cap['IN']
        caprhs[1, :] = inst.park_cap['OUT']
        fp = np.zeros(NB * m); fp[:T] = W.fixed_park
        fpb = fp.reshape(NB, m).max(axis=1)
        caprhs[0] -= fpb
        fpo = np.zeros(NB * m); fpo[:T] = W.fixed_out_park
        caprhs[1] -= fpo.reshape(NB, m).max(axis=1)
        if m == 1:
            ii, ll, bb = np.meshgrid(np.arange(nV), [0, 1], np.arange(NB), indexing='ij')
            ent(crow(ll, bb), yid(ii, ll, bb), 1)
        else:
            ii, ll, bb = np.meshgrid(np.arange(nV), [0, 1], np.arange(1, NB), indexing='ij')
            ent(crow(ll, bb), yid(ii, ll, bb - 1), 1)
            inm = ev_in
            ent(crow(ev_loc[inm], ev_blk[inm]), ev_col[inm], 1)
            caprhs -= const_in.sum(axis=0)
            caprhs[:, 0] -= y_init.sum(axis=0)
        # mode handover parking (IN): any-overlap of the profile with the block
        for j in range(xv.size):
            pk = mode_profile(V[xv[j]]['modes'][xb[j]], xs[j])[2]
            nz = np.nonzero(pk)[0]
            if nz.size == 0:
                continue
            tt_ = xs[j] + nz
            tt_ = tt_[(tt_ >= t0) & (tt_ < t1)]
            bl = np.unique((tt_ - t0) // m)
            ent(crow(0, bl), np.full(bl.size, xcols[j]), 1)
        # towing handover parking at IN: departure slot s (src IN) or arrival slot s+move-1 (dst IN)
        for j in range(zv.size):
            tt = trips[zk[j]]
            if tt.mode != 'tow':
                continue
            hs = zs[j] if tt.src == 'IN' else zs[j] + tt.move - 1
            if t0 <= hs < t1:
                ent([crow(0, (hs - t0) // m)], [zcols[j]], 1)
        # rented spaces (OUT) and slack
        bday = (t0 + np.arange(NB) * m) // SLOTS_PER_DAY
        ent(crow(1, np.arange(NB)), self.u0 + np.searchsorted(days, bday), -1)
        ent(crow(np.repeat([0, 1], NB), np.tile(np.arange(NB), 2)), self.sg0 + np.arange(2 * NB), -1)
        rhi[-1][:] = caprhs.ravel()

        # A coarse block may intersect several dates. Its occupancy bound must
        # fit the rented spaces on EACH of those dates (including a partial last
        # block); checking just the block's start date misses the next day's rent.
        cap_parts = tuple(zip(R[cap_entries_start:], C[cap_entries_start:], Vv[cap_entries_start:]))
        cross = [(b, d) for b, dates in enumerate(self.block_days) for d in dates[1:]]
        self.cross_day_blocks = np.array([b for b, _ in cross], dtype=int)
        calendar0 = addrows(len(cross), -np.inf, 0)
        self.row_groups['cap_calendar'] = (calendar0, len(cross))
        for q, (b, day) in enumerate(cross):
            row = calendar0 + q
            for rr, cc, vv in cap_parts:
                sel = rr == crow(1, b)
                if sel.any():
                    ent(np.full(sel.sum(), row), cc[sel], vv[sel])
            first_rent = self.u0 + np.searchsorted(days, self.block_days[b][0])
            this_rent = self.u0 + np.searchsorted(days, day)
            ent([row, row], [first_rent, this_rent], [1, -1])
            rhi[-1][q] = caprhs[1, b]

        # (8) recovery precondition: y[i,OUT, blk(s-pre)-1] >= sum z[i,k,s] (k from OUT)
        sel = zsrc == 1
        keys = {}
        for j in np.nonzero(sel)[0]:
            keys.setdefault((zv[j], zs[j], zpre[j]), []).append(zcols[j])
        pr0 = addrows(len(keys), -np.inf, 0)
        self.row_groups['precond'] = (pr0, len(keys))
        rh = np.zeros(len(keys))
        for gi, ((i, s, pre), cols) in enumerate(keys.items()):
            ent([pr0 + gi] * len(cols), cols, 1)
            b = (s - pre - t0) // m - 1
            if b >= 0:
                ent([pr0 + gi], [yid(i, 1, b)], -1)
            else:
                rh[gi] = y_init[i, 1]
        rhi[-1][:] = rh

        # (9) resources per (r, t)
        rnames = sorted(inst.caps)
        self.rnames = rnames
        # extended range T+: resource use of alternatives started before t1 is constrained until it ends
        tX = int(max(t1, (xs + xp).max(initial=t1)))
        self.tX = tX
        TX = tX - t0

        def fixed_x(r):
            out = np.zeros(TX)
            a_ = W.fixed_res.get(r)
            if a_ is not None:
                n_ = min(TX, len(a_)); out[:n_] = a_[:n_]
            return out
        self.Qx = {r: inst.cap_series(r, t0, tX) - fixed_x(r) for r in rnames}
        rr0 = addrows(len(rnames) * TX, -np.inf, 0)
        self.row_groups['res'] = (rr0, len(rnames) * TX)
        rrow = {r: rr0 + q * TX for q, r in enumerate(rnames)}
        rcap = np.concatenate([self.Qx[r] for r in rnames])
        rhi[-1][:] = rcap
        for j in range(xv.size):
            md = V[xv[j]]['modes'][xb[j]]
            for r, prof in mode_profile(md, xs[j])[1].items():
                nz = np.nonzero(prof)[0]
                tt_ = xs[j] + nz
                keep = tt_ < tX
                ent(rrow[r] + tt_[keep] - t0, np.full(keep.sum(), xcols[j]), prof[nz][keep])
        for j, (k, s) in enumerate(zip(tk, ts)):
            tt = trips[k]
            col = self.w0 + j
            if tt.mode == 'staff':
                span = np.arange(s - tt.pre, s + tt.back + REST_SLOTS)
                ent(rrow['gen'] + span - t0, np.full(span.size, col), tt.q + 1)
                for gi, g_ in enumerate(self.tech):
                    cc = self.cw0 + self.cw_pos[j] * len(self.tech) + gi
                    ent(rrow['gen'] + span - t0, np.full(span.size, cc), -1)
                    ent(rrow[g_] + span - t0, np.full(span.size, cc), 1)
                span2 = np.arange(s - tt.pre, s + tt.back)
                ent(rrow['supportcar'] + span2 - t0, np.full(span2.size, col), 1)
            else:
                span = np.arange(s, s + tt.move)
                ent(rrow['sup_tow'] + span - t0, np.full(span.size, col), 1)
                hs = s if tt.src == 'IN' else s + tt.move - 1
                ent([rrow['gen'] + hs - t0], [col], 1)

        # (10) trip link: sum_i z[i,k,s] - q w[k,s] = 0
        lr0 = addrows(tk.size, 0, 0)
        self.row_groups['trip'] = (lr0, tk.size)
        ent(lr0 + np.arange(tk.size), self.w0 + np.arange(tk.size), -np.array([trips[k].q for k in tk], float))
        zw = np.array([widx[(k, s)] - self.w0 for k, s in zip(zk, zs)], int)
        ent(lr0 + zw, zcols, 1)
        # crew rows: sum_g cw[w,g] <= (q+1) w
        nt_ = len(self.tech)
        if self.cw_w.size and nt_:
            cr_ = addrows(self.cw_w.size, -np.inf, 0)
            self.row_groups['crew'] = (cr_, self.cw_w.size)
            for a, j in enumerate(self.cw_w):
                ent([cr_ + a] * nt_, self.cw0 + a * nt_ + np.arange(nt_), 1)
                ent([cr_ + a], [self.w0 + j], -(trips[tk[j]].q + 1))

        # (11) lateness, with suffix-duration differences p^rem from eq. (3.14).
        late_v = [i for i, v in enumerate(V) if v['tasks'] and v['deadline'] < INF]
        lt0 = addrows(len(late_v), 0, np.inf)
        self.row_groups['late'] = (lt0, len(late_v))
        lrow = {i: lt0 + j for j, i in enumerate(late_v)}
        lrhs = np.array([-SLOT_MIN * V[i]['deadline'] for i in late_v], float)
        rlo[-1][:] = lrhs
        if late_v:
            ent(lt0 + np.arange(len(late_v)), self.L0 + np.array(late_v), 1)
            last = np.array([len(V[i]['tasks']) - 1 for i in xv], int)
            sel = np.array([(gx[j] == last[j]) and (xv[j] in lrow) for j in range(xv.size)], bool)
            ent([lrow[i] for i in xv[sel]], xcols[sel], -SLOT_MIN * (xs + xp)[sel])
            # eq. [15], last term: a newly started non-final alternative that ends after t1
            selo = np.array([(gx[j] < last[j]) and (xv[j] in lrow) and (xs[j] + xp[j] > t1)
                             for j in range(xv.size)], bool)
            if selo.any():
                ent([lrow[i] for i in xv[selo]], xcols[selo], -SLOT_MIN * (xs + xp - t1)[selo])
            for i in late_v:
                nk = len(V[i]['tasks'])
                ent([lrow[i]], [oidx[(i, nk - 1)]], -SLOT_MIN * t1)
                for k in range(nk):
                    ent([lrow[i]], [oidx[(i, k)]], -SLOT_MIN * self.p_rem[i][k])
        # vehicles without lateness row: L fixed to 0
        nolate = np.setdiff1d(np.arange(nV), np.array(late_v, int))

        # F-cap row (stage 2 of the lexicographic search): sum_b Fw_b E_b <= F*
        self.fcap_row = addrows(1, -np.inf, np.inf)
        ent(np.full(NB, self.fcap_row), self.sg0 + np.arange(NB), self.Fw)
        self.nrow = nrow
        Rr = np.concatenate(R); Cc = np.concatenate(C); Vals = np.concatenate(Vv)
        A = sp.csc_matrix((Vals, (Rr, Cc)), shape=(nrow, ncol))
        A.sum_duplicates()
        self.A = A
        self.cost = np.concatenate(cost); self.lo = np.concatenate(lo); self.hi = np.concatenate(hi)
        self.cost_base = self.cost.copy()
        # holding cost of unfinished vehicles (eq. 3.1, last term before the penalty):
        #   c_hold * delta * (t^c_i - max(t0, arrival_i)); the reference is fixed input
        ch = getattr(inst, 'hold_cost_day', 0.0) / SLOTS_PER_DAY          # per slot
        if ch > 0:
            lastx = np.array([len(V[i]['tasks']) - 1 for i in xv], int)
            selx = gx == lastx
            hold_start = np.array([V[i].get('hold_start', t0) for i in xv], int)
            self.cost[xcols[selx]] += ch * (xs + xp - hold_start)[selx]
            selo = (gx < lastx) & (xs + xp > t1)
            self.cost[xcols[selo]] += ch * (xs + xp - t1)[selo]
            for i, v in enumerate(V):
                nk = len(v['tasks'])
                if nk == 0:
                    continue
                self.cost[oidx[(i, nk - 1)]] += ch * (t1 - v.get('hold_start', t0))
                for k in range(nk):
                    self.cost[oidx[(i, k)]] += ch * self.p_rem[i][k]
        self.integ = np.concatenate(integ)
        self.hi[self.L0 + nolate] = 0
        self.rlo = np.concatenate(rlo); self.rhi = np.concatenate(rhi)
        self.build_time = time.perf_counter() - t_build
        self.h = None

    # ------------------------------------------------------------------ statistics
    def stats(self):
        A = self.A
        nb = self.integ.astype(bool)
        groups = {k: v[1] for k, v in self.row_groups.items()}
        y_cols = len(self.W.vehs) * 2 * self.NB
        return dict(m=self.m, ncol=self.ncol, nrow=self.nrow, nnz=int(A.nnz),
                    n_int=int(nb.sum()), n_x=int(self.xv.size), n_y=int(y_cols), n_z=int(self.zv.size),
                    n_w=int(self.wk.size), n_e=int(self.ev.size), n_o=int(self.ov.size),
                    rows=groups, build_s=round(self.build_time, 3),
                    nnz_y=int(A[:, self.y0:self.y0 + y_cols].nnz))

    # ------------------------------------------------------------------ HiGHS
    def to_highs(self, threads=1, verbose=False):
        h = highspy.Highs()
        h.setOptionValue('output_flag', verbose)
        h.setOptionValue('threads', threads)
        lp = highspy.HighsLp()
        lp.num_col_ = self.ncol
        lp.num_row_ = self.nrow
        lp.col_cost_ = self.cost
        lp.col_lower_ = self.lo
        lp.col_upper_ = self.hi
        lp.row_lower_ = self.rlo
        lp.row_upper_ = self.rhi
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_ = self.A.indptr.astype(np.int32)
        lp.a_matrix_.index_ = self.A.indices.astype(np.int32)
        lp.a_matrix_.value_ = self.A.data
        lp.integrality_ = [highspy.HighsVarType.kInteger if v else highspy.HighsVarType.kContinuous
                           for v in self.integ]
        h.passModel(lp)
        self.h = h
        return h

    def solve(self, time_limit=60.0, start=None, mip_gap=0.0, lo=None, hi=None, threads=1):
        if self.h is None:
            self.to_highs(threads=threads)
        h = self.h
        if lo is not None:
            idx = np.arange(self.ncol, dtype=np.int32)
            h.changeColsBounds(self.ncol, idx, lo, hi)
        h.setOptionValue('time_limit', float(max(0.05, time_limit)))
        h.setOptionValue('mip_rel_gap', float(mip_gap))
        if start is not None:
            sol = highspy.HighsSolution()
            sol.col_value = list(map(float, start))
            sol.value_valid = True
            h.setSolution(sol)
        t = time.perf_counter()
        h.run()
        el = time.perf_counter() - t
        st = h.getModelStatus()
        info = h.getInfo()
        has = info.primal_solution_status == 2   # kSolutionStatusFeasible
        xsol = np.array(h.getSolution().col_value) if has else None
        return dict(status=h.modelStatusToString(st), time=el, x=xsol,
                    obj=info.objective_function_value if has else None,
                    bound=info.mip_dual_bound, gap=info.mip_gap)

    # ------------------------------------------------------------------ decode
    def decode(self, sol):
        V = self.W.vehs
        r = lambda a, n: np.nonzero(sol[a:a + n] > 0.5)[0]
        modes = [(V[self.xv[j]]['vid'], int(self.xb[j]), int(self.xs[j])) for j in r(self.x0, self.xv.size)]
        trips = [(V[self.zv[j]]['vid'], int(self.zk[j]), int(self.zs[j])) for j in r(self.z0, self.zv.size)]
        exits = [(V[self.ev[j]]['vid'], int(self.es[j])) for j in r(self.e0, self.ev.size)]
        rent = {int(d): int(round(sol[self.u0 + q])) for q, d in enumerate(self.days)}
        ex = self.excess(sol)
        sigma = float(ex.sum())
        defer = [(V[self.ov[j]]['vid'], int(self.ok[j])) for j in r(self.o0, self.ov.size)]
        crew = {}
        nt_ = len(self.tech)
        for a, j in enumerate(self.cw_w):
            if sol[self.w0 + j] > 0.5:
                c = {g: int(round(sol[self.cw0 + a * nt_ + gi])) for gi, g in enumerate(self.tech)
                     if sol[self.cw0 + a * nt_ + gi] > 0.5}
                if c:
                    crew[(int(self.wk[j]), int(self.ws[j]))] = c
        return dict(modes=modes, trips=trips, exits=exits, rent=rent, sigma=sigma, defer=defer,
                    crew=crew, F=float(self.Fw @ ex))

    # ------------------------------------------------------------------ lexicographic objective (F, Z)
    def excess(self, sol):
        """E_b: smallest internal excess waiting that satisfies the IN capacity rows (eq. [9])."""
        r0, _ = self.row_groups['cap']
        NB = self.NB
        act = self.A[r0:r0 + NB] @ sol + sol[self.sg0:self.sg0 + NB]
        return np.maximum(0.0, np.round(act - self.rhi[r0:r0 + NB], 6))

    def F(self, sol):
        return float(self.Fw @ self.excess(sol))

    def Z(self, sol):
        return float(self.cost @ sol)

    def normalize(self, sol):
        s = np.array(sol, float)
        s[self.sg0:self.sg0 + self.NB] = self.excess(sol)
        return s

    def set_stage(self, stage, Fstar=None):
        """stage 1: minimise F;  stage 2: minimise Z subject to F <= F*."""
        h = self.h
        idx = np.arange(self.ncol, dtype=np.int32)
        if stage == 1:
            c = np.zeros(self.ncol); c[self.sg0:self.sg0 + self.NB] = self.Fw
            h.changeRowBounds(int(self.fcap_row), -highspy.kHighsInf, highspy.kHighsInf)
        else:
            c = self.cost
            h.changeRowBounds(int(self.fcap_row), -highspy.kHighsInf, float(Fstar) + 1e-6)
        h.changeColsCost(self.ncol, idx, np.asarray(c, float))

    def cost_parts(self, sol):
        c = self.cost_base * sol
        part = dict(hold=float((self.cost - self.cost_base) @ sol), work=c[self.x0:self.x0 + self.xv.size].sum(),
                    residual=c[self.o0:self.o0 + self.ov.size].sum(),
                    pickup_wait=c[self.e0:self.e0 + self.ev.size].sum() + c[self.ne0:self.ne0 + len(self.ne_v)].sum(),
                    trips=c[self.w0:self.w0 + self.wk.size].sum(),
                    rent=c[self.u0:self.u0 + self.days.size].sum(),
                    late=c[self.L0:self.L0 + len(self.W.vehs)].sum(),
                    sigma=c[self.sg0:self.sg0 + 2 * self.NB].sum())
        return part

    def encode(self, plan):
        """Column vector from a plan dict (modes/trips/exits/rent), y/w/L/o derived by propagation.
        Returns vector consistent with flow rows (assuming the plan is physically consistent)."""
        V = self.W.vehs
        vid2i = {v['vid']: i for i, v in enumerate(V)}
        sol = np.zeros(self.ncol)
        xkey = {(int(self.xv[j]), int(self.xb[j]), int(self.xs[j])): j for j in range(self.xv.size)}
        for (vid, b, s) in plan['modes']:
            sol[self.x0 + xkey[(vid2i[vid], b, s)]] = 1
        zkey = {(int(self.zv[j]), int(self.zk[j]), int(self.zs[j])): j for j in range(self.zv.size)}
        for (vid, k, s) in plan['trips']:
            sol[self.z0 + zkey[(vid2i[vid], k, s)]] = 1
        wkey = {(int(k), int(s)): j for j, (k, s) in enumerate(zip(self.wk, self.ws))}
        q = {tt.kid: tt.q for tt in self.W.inst.trip_types}
        from collections import Counter
        cnt = Counter((k, s) for (_, k, s) in plan['trips'])
        for (k, s), c in cnt.items():
            sol[self.w0 + wkey[(k, s)]] = c / q[k]
        nt_ = len(self.tech)
        for (k, s), cr in (plan.get('crew') or {}).items():
            a = self.cw_pos[wkey[(int(k), int(s))]]
            for gi, g in enumerate(self.tech):
                sol[self.cw0 + a * nt_ + gi] = cr.get(g, 0)
        ekey = {(int(self.ev[j]), int(self.es[j])): j for j in range(self.ev.size)}
        exited = set()
        for (vid, s) in plan['exits']:
            sol[self.e0 + ekey[(vid2i[vid], s)]] = 1; exited.add(vid2i[vid])
        for j, i in enumerate(self.ne_v):
            if i not in exited:
                sol[self.ne0 + j] = 1
        # deferral: tasks not covered by a chosen mode
        covered = set()
        for (vid, b, s) in plan['modes']:
            md = V[vid2i[vid]]['modes'][b]
            for k in range(md.f, md.g + 1):
                covered.add((vid2i[vid], k))
        for j, (i, k) in enumerate(zip(self.ov, self.ok)):
            if (i, k) not in covered:
                sol[self.o0 + j] = 1
        for q_, d in enumerate(self.days):
            sol[self.u0 + q_] = plan.get('rent', {}).get(int(d), 0)
        # y by propagation of flow rows
        ev_col, ev_veh, ev_loc, ev_t, ev_in = self.events
        act = sol[ev_col] > 0.5
        nV, NB, m = len(V), self.NB, self.m
        delta = self.const_in.copy()
        blkk = (ev_t[act] - self.t0) // m
        np.add.at(delta, (ev_veh[act], ev_loc[act], blkk), np.where(ev_in[act], 1, -1))
        y = self.y_init[:, :, None] + np.cumsum(delta, axis=2)
        sol[self.y0:self.y0 + nV * 2 * NB] = y.ravel()
        # lateness
        for i, v in enumerate(V):
            if not v['tasks'] or v['deadline'] >= INF:
                continue
            nk = len(v['tasks'])
            comp = None
            for (vid, b, s) in plan['modes']:
                if vid2i[vid] == i and v['modes'][b].g == nk - 1:
                    comp = s + mode_span(v['modes'][b], s)
            if comp is None:
                comp = self.t1 + sum(self.p_rem[i][k]
                                     for k in range(nk) if (i, k) not in covered)
                comp += max([0] + [s + mode_span(v['modes'][b], s) - self.t1 for (vid, b, s) in plan['modes']
                                   if vid2i[vid] == i])
            sol[self.L0 + i] = max(0.0, SLOT_MIN * (comp - v['deadline']))
        # slack: overflow implied
        for group, slack_cols in [
            ('cap', self.sg0 + np.arange(2 * NB)),
            ('cap_calendar', self.sg0 + NB + self.cross_day_blocks),
        ]:
            r0, n = self.row_groups[group]
            if n:
                act_ = self.A[r0:r0 + n] @ sol - self.rhi[r0:r0 + n]
                need = np.maximum(0, act_ + sol[slack_cols])
                np.maximum.at(sol, slack_cols, need)
        return sol

    def max_violation(self, sol):
        a = self.A @ sol
        v = np.maximum(self.rlo - a, 0).max(initial=0)
        v = max(v, np.maximum(a - self.rhi, 0).max(initial=0))
        v = max(v, np.maximum(self.lo - sol, 0).max(initial=0), np.maximum(sol - self.hi, 0).max(initial=0))
        return float(v)
