"""Instance data and synthetic generator for the reconditioning scheduling prototype.

All times are absolute 5-minute slot indices counted from day-0 00:00 (day 0 = Monday).
Every numeric value produced by the generator is an EXPERIMENT ASSUMPTION ([실험 가정]),
not field data. See README for the list.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

SLOT_MIN = 5
SLOTS_PER_DAY = 24 * 60 // SLOT_MIN          # 288
OPEN_SLOT = 9 * 60 // SLOT_MIN               # 09:00 -> 108
CLOSE_SLOT = 18 * 60 // SLOT_MIN             # 18:00 -> 216
REPLAN_SLOT = 9 * 60 // SLOT_MIN             # 09:00 -> 108
REST_SLOTS = 2                               # 10-minute rest after an internal trip (v5 rule)
INF = 10 ** 9


def is_business_day(day: int) -> bool:
    return (day % 7) < 5                     # Mon-Fri  [실험 가정]


def business_mask(n_slots: int) -> np.ndarray:
    """1 if slot t is inside business hours of a business day."""
    m = np.zeros(n_slots, dtype=bool)
    for d in range(n_slots // SLOTS_PER_DAY + 1):
        if not is_business_day(d):
            continue
        a, b = d * SLOTS_PER_DAY + OPEN_SLOT, d * SLOTS_PER_DAY + CLOSE_SLOT
        m[max(0, a):max(0, min(b, n_slots))] = True
    return m


@dataclass
class Mode:
    """An execution alternative covering the contiguous tasks f..g of one vehicle."""
    f: int
    g: int
    kind: str                 # 'I' internal, 'O' outsourced, 'Q' internal final inspection
    p: int                    # duration in slots (start -> return & handover done)
    res: dict                 # resource name -> np.ndarray(len p) of usage
    park: np.ndarray          # IN-parking occupancy per relative slot (handover on a parking space)
    cost: float
    label: str
    grid: int = 12            # start-candidate spacing in slots (12 = hourly, 6 = 30 min)
    staff_slots: np.ndarray = None   # relative slots that need staff in business hours


@dataclass
class Vehicle:
    vid: int
    arrival: int | None       # arrival slot (None = initial inventory present at t=0)
    init_loc: str | None      # 'IN'/'OUT' for initial inventory
    tasks: list               # task type names, last one is 'insp' unless already inspected
    modes: list               # list[Mode]
    deadline: int             # promised completion slot (shop-responsible service deadline)
    approval_at: int | None = None     # exogenous customer approval time (None: not needed, INF: never)
    response_delay: int | None = None  # slots between completion notice and pickup request (None: no reply)
    fault: dict | None = None          # {'notice': slot, 'decision': slot, 'outcome': 'extend'|'return',
                                       #  'extend_slots': int, 'collect_delay': slots}
    done: int = 0             # number of tasks already completed before t=0 (initial inventory)
    cohort: str = 'new'       # 'init' or 'new'


@dataclass
class TripType:
    kid: int
    src: str
    dst: str
    q: int
    pre: int
    move: int
    back: int
    mode: str                 # 'staff' internal staff trip, 'tow' external towing
    cost: float


@dataclass
class Instance:
    horizon: int                              # observation horizon in slots (14 days = 4032)
    vehicles: list
    caps: dict                                # resource -> int (base count during availability)
    staff_resources: set                      # resources that follow business-hour availability
    park_cap: dict                            # 'IN'/'OUT' -> spaces
    rent_max: int                             # additional OUT spaces per day
    rent_cost_day: float
    trip_types: list
    lateness_cost_min: float = 20.0           # v5 synthetic coefficient (원/분)
    pickup_wait_cost_min: float = 20.0        # customer waiting for retrieval after request (원/분) [가정]
    hold_cost_day: float = 0.0                # holding cost of an unfinished vehicle per day (원/대·일) [실험 가정]
    storage_fee_day: float = 10000.0          # example storage fee, customer-side waiting [예시]
    storage_grace_days: int = 3               # [예시]
    name: str = ''
    meta: dict = field(default_factory=dict)

    def cap_series(self, r: str, t0: int, t1: int) -> np.ndarray:
        base = self.caps[r]
        if r in self.staff_resources:
            return base * business_mask(t1)[t0:t1].astype(int)
        return np.full(t1 - t0, base, dtype=int)


# ---------------------------------------------------------------- task catalogue  [실험 가정]
# internal: (cost, minutes, staff_group, equipment)   outsourced: (cost incl. round-trip tow, calendar minutes, supply)
# Synthetic unit prices, not quotations. Edit the FIRST number in an O tuple or
# BUNDLES tuple to reflect another external-price scenario; no model change is
# needed. Small/medium work starts at cost parity. Heavy work stays cheaper IN.
CAT = {
    'wash':  dict(I=(45000, 60, 'gen', 'washbay'),   O=(45000, 6 * 60, 'sup_wash')),
    'maint': dict(I=(45000, 90, 'mech', 'lift'),     O=(45000, 24 * 60, 'sup_maint')),
    'elec':  dict(I=(65000, 120, 'mech', 'lift'),    O=(65000, 24 * 60, 'sup_maint')),
    'engine': dict(I=(240000, 240, 'mech', 'lift'),  O=(480000, 72 * 60, 'sup_engine')),
    'body_l': dict(I=(65000, 120, 'body', 'bodybay')),
    'paint_l': dict(I=(85000, 120, 'body', 'paintline')),
    'body_h': dict(I=(160000, 240, 'body', 'bodybay')),
    'paint_h': dict(I=(140000, 240, 'body', 'paintline')),
    'insp':  dict(I=(15000, 30, 'insp', 'inspbay')),
}
BUNDLES = {('body_l', 'paint_l'): (150000, 48 * 60, 'sup_body'),
           ('body_h', 'paint_h'): (360000, 72 * 60, 'sup_body')}
HANDOVER = 1   # slots of handover at departure and at return (staff 'gen' + one IN parking space)


STAFF_RES = ('gen', 'mech', 'body', 'insp')
_CAL = {'n': 0}
_SPAN = {}


def _calendar(n):
    if _CAL['n'] < n:
        bm = business_mask(n + 20 * SLOTS_PER_DAY)
        _CAL.update(n=bm.size, bm=bm, slots=np.nonzero(bm)[0], cum=np.cumsum(bm))
    return _CAL


def work_offsets(p, s):
    """Offsets (relative to s) of the p working slots of an internal task started at s.
    Work stops at the end of business and resumes at the next business opening (no overtime)."""
    r = _SPAN.get((p, s))
    if r is None:
        c = _calendar(s + p + 8 * SLOTS_PER_DAY)
        k0 = int(c['cum'][s]) - (1 if c['bm'][s] else 0)
        r = c['slots'][k0:k0 + p] - s
        _SPAN[(p, s)] = r
    return r


def mode_span(md, s):
    """p_ijt: slots from the start s to the end of the alternative, interruptions included."""
    if md.kind == 'O':
        return md.p
    return int(work_offsets(md.p, s)[-1]) + 1


def mode_profile(md, s):
    """(span, resource profiles, handover-parking profile) of an alternative started at s.
    During an interruption staff usage is 0; the work bay and equipment stay occupied."""
    if md.kind == 'O':
        return md.p, md.res, md.park
    rel = work_offsets(md.p, s)
    span = int(rel[-1]) + 1
    if span == md.p:
        return md.p, md.res, md.park
    nxt = np.r_[rel[1:], span]
    res = {}
    for r, prof in md.res.items():
        arr = np.zeros(span, prof.dtype)
        if r in STAFF_RES:
            arr[rel] = prof
        else:
            for q in np.nonzero(prof)[0]:
                arr[rel[q]:nxt[q]] = prof[q]
        res[r] = arr
    park = np.zeros(span, md.park.dtype)
    for q in np.nonzero(md.park)[0]:
        park[rel[q]:nxt[q]] = md.park[q]
    return span, res, park


def _internal_mode(k, ttype):
    cost, mins, staff, eq = CAT[ttype]['I']
    p = mins // SLOT_MIN
    res = {staff: np.ones(p, int), eq: np.ones(p, int)}
    kind = 'Q' if ttype == 'insp' else 'I'
    grid = 6 if mins <= 60 else 12
    return Mode(k, k, kind, p, res, np.zeros(p, int), cost, f'{ttype}:I', grid, np.arange(p))


def _outsourced_mode(f, g, cost, mins, sup, label):
    p = mins // SLOT_MIN
    gen = np.zeros(p, int); gen[:HANDOVER] = 1; gen[p - HANDOVER:] = 1
    park = np.zeros(p, int); park[:HANDOVER] = 1; park[p - HANDOVER:] = 1
    supply = np.zeros(p, int); supply[HANDOVER:p - HANDOVER] = 1
    staff_slots = np.r_[np.arange(HANDOVER), np.arange(p - HANDOVER, p)]
    return Mode(f, g, 'O', p, {'gen': gen, sup: supply}, park, cost, label, 12, staff_slots)


def build_modes(tasks):
    modes = []
    for k, tt in enumerate(tasks):
        if 'I' in CAT[tt]:
            modes.append(_internal_mode(k, tt))
        if 'O' in CAT[tt]:
            c, m, s = CAT[tt]['O']
            modes.append(_outsourced_mode(k, k, c, m, s, f'{tt}:O'))
    for k in range(len(tasks) - 1):
        key = (tasks[k], tasks[k + 1])
        if key in BUNDLES:
            c, m, s = BUNDLES[key]
            modes.append(_outsourced_mode(k, k + 1, c, m, s, f'{key[0]}+{key[1]}:O'))
    # Only heavy work has the specified strict premium. Small/medium external
    # quotes may be equal or lower. Compare a bundle to the SUM of internal work.
    for md in modes:
        if md.kind != 'O':
            continue
        scope = tasks[md.f:md.g + 1]
        heavy = any(tt in ('engine', 'body_h', 'paint_h') for tt in scope)
        if heavy and all('I' in CAT[tt] for tt in scope):
            internal_cost = sum(CAT[tt]['I'][0] for tt in scope)
            if md.cost <= internal_cost:
                raise ValueError(
                    f'Heavy outsourcing must cost more than the same internal work: '
                    f'{md.label}: external={md.cost}, internal={internal_cost}'
                )
    return modes


def remaining_relaxation(tasks, modes=None):
    """Minimum remaining cost and duration ignoring calendars and shared resources.

    Modes cover consecutive tasks, including valid outsourcing bundles. Dynamic
    programming over the task path gives lower bounds on every feasible schedule.
    Duration is returned in 5-minute slots, cost in KRW. Optional modes must use
    indices local to tasks. Passing them preserves the actual alternatives of a
    planning window instead of reconstructing a catalogue for a test/custom input.
    """
    n = len(tasks)
    cost = np.full(n + 1, np.inf)
    duration = np.full(n + 1, np.inf)
    cost[n] = duration[n] = 0.0
    modes = build_modes(tasks) if modes is None else modes
    for k in range(n - 1, -1, -1):
        options = [md for md in modes if md.f == k]
        if not options:
            raise ValueError(f'No processing alternative for task {tasks[k]}')
        cost[k] = min(md.cost + cost[md.g + 1] for md in options)
        duration[k] = min(md.p + duration[md.g + 1] for md in options)
    return cost, duration


def make_trip_types():
    """Internal staff trips IN<->OUT with 1..3 target cars (+1 support car driver) and
    external towing (q=1).  Times in slots: pre(staff leaves before car), move, back."""
    tt, kid = [], 0
    for q in (1, 2, 3):
        # IN -> OUT : staff drive cars, walk/ride back with support car   total absence 20-30 min
        tt.append(TripType(kid, 'IN', 'OUT', q, 0, 2, 4 + (q - 1), 'staff', 2000)); kid += 1
        # OUT -> IN : staff go first (pre), then drive cars back
        tt.append(TripType(kid, 'OUT', 'IN', q, 2, 2, 2 + (q - 1), 'staff', 2000)); kid += 1
    tt.append(TripType(kid, 'IN', 'OUT', 1, 0, 3, 3, 'tow', 10000)); kid += 1
    tt.append(TripType(kid, 'OUT', 'IN', 1, 0, 3, 3, 'tow', 10000)); kid += 1
    return tt


DEFAULT_CAPS = dict(gen=6, mech=4, body=3, insp=2,          # v5 staff split of 15 (synthetic)
                    lift=8, bodybay=3, paintline=1, washbay=2, inspbay=2, supportcar=1,
                    sup_wash=4, sup_maint=4, sup_engine=2, sup_body=2, sup_tow=2)
STAFF = {'gen', 'mech', 'body', 'insp', 'lift', 'bodybay', 'paintline', 'washbay', 'inspbay',
         'supportcar', 'sup_tow'}       # usable only in business hours


# ---------------------------------------------------------------- deadlines by scope [사용자 제공 구조, 수치는 가정]
def scope_deadline_days(tasks, rng):
    heavy = any(t in ('engine', 'body_h') for t in tasks)
    body = any(t.startswith('body') for t in tasks)
    if heavy:
        return int(rng.choice([5, 7, 10, 14, 21, 30], p=[.3, .3, .2, .1, .05, .05]))
    if body:
        return int(rng.choice([3, 4, 5], p=[.4, .4, .2]))
    return int(rng.choice([1, 2, 3, 4], p=[.3, .3, .2, .2]))


def add_business_days(slot, ndays):
    """End of the ndays-th business day counted from the day of `slot` (day of arrival = day 1)."""
    d = slot // SLOTS_PER_DAY
    cnt = 0
    while True:
        if is_business_day(d):
            cnt += 1
            if cnt >= ndays:
                return d * SLOTS_PER_DAY + CLOSE_SLOT
        d += 1


def next_business_open(slot):
    d = slot // SLOTS_PER_DAY
    while True:
        if is_business_day(d):
            a = d * SLOTS_PER_DAY + OPEN_SLOT
            b = d * SLOTS_PER_DAY + CLOSE_SLOT
            if slot < a:
                return a
            if slot < b:
                return slot
        d += 1
        slot = d * SLOTS_PER_DAY


def pickup_slot(v, completion: int):
    """Exogenous customer behaviour: reply `response_delay` after the completion notice and book a
    pickup on the next business day after the reply day, 10:00 + (0..6 h) offset fixed per vehicle.
    Returns None if the customer never replies.  [실험 가정]"""
    if v.response_delay is None:
        return None
    reply = completion + v.response_delay
    d = reply // SLOTS_PER_DAY + 1
    while not is_business_day(d):
        d += 1
    return d * SLOTS_PER_DAY + 10 * 60 // SLOT_MIN + (v.vid * 37 % 7) * 12


def reply_slot(v, completion: int):
    if v.response_delay is None:
        return None
    return completion + v.response_delay


def collect_slot(v):
    """Physical release of a vehicle returned after a shop-responsible event (settled at decision)."""
    f = v.fault
    return next_business_open(f['decision'] + f['collect_delay'])


def draw_tasks(rng):
    r = rng.random()
    if r < 0.25:
        t = ['wash']
    elif r < 0.50:
        t = ['maint'] + (['wash'] if rng.random() < .4 else [])
    elif r < 0.62:
        t = ['elec'] + (['wash'] if rng.random() < .4 else [])
    elif r < 0.82:
        t = ['body_l', 'paint_l', 'wash']
    elif r < 0.92:
        t = ['body_h', 'paint_h', 'wash']
    else:
        t = ['engine'] + (['wash'] if rng.random() < .4 else [])
    return t + ['insp']


def generate(daily: int, seed: int, days: int = 14, init_factor: float = 1.0,
             p_approval=0.20, p_fault=0.03, park_in=12, park_out=18, rent_max=60,
             caps=None, name=None) -> Instance:
    """Daily-inflow instance: `daily` new orders on every business day of the 14 calendar days,
    plus an initial unfinished inventory of round(init_factor*daily) vehicles."""
    rng = np.random.default_rng(seed)
    H = days * SLOTS_PER_DAY
    vehicles = []
    vid = 0

    def customer_draws(v):
        # approval wait (customer side) [실험 가정]
        if rng.random() < p_approval:
            v.approval_at = INF if rng.random() < 0.05 else None  # placeholder, set below
            if v.approval_at is None:
                base = v.arrival if v.arrival is not None else 0
                v.approval_at = next_business_open(base + int(rng.integers(24, 2 * SLOTS_PER_DAY)))
        # pickup response after completion notice [실험 가정]
        r = rng.random()
        if r < 0.70:
            v.response_delay = int(rng.integers(12, SLOTS_PER_DAY))          # 1h .. 1 day
        elif r < 0.95:
            v.response_delay = int(rng.integers(2 * SLOTS_PER_DAY, 5 * SLOTS_PER_DAY))
        else:
            v.response_delay = None                                           # no reply
        # shop-responsible fault event [실험 가정]
        if rng.random() < p_fault:
            base = v.arrival if v.arrival is not None else 0
            notice = next_business_open(base + int(rng.integers(36, 2 * SLOTS_PER_DAY)))
            decision = next_business_open(notice + SLOTS_PER_DAY)
            v.fault = dict(notice=notice, decision=decision,
                           outcome='return' if rng.random() < 0.5 else 'extend',
                           extend_slots=3 * SLOTS_PER_DAY,
                           collect_delay=int(rng.integers(SLOTS_PER_DAY, 3 * SLOTS_PER_DAY)))

    # initial inventory: arrived during the previous week, partly processed
    n_init = int(round(init_factor * daily))
    for _ in range(n_init):
        tasks = draw_tasks(rng)
        done = int(rng.integers(0, len(tasks)))       # tasks already done (<= all but maybe insp)
        if rng.random() < 0.15:
            done = len(tasks)                         # completed, waiting for customer pickup
        rem = tasks[done:]
        modes = build_modes(rem) if rem else []
        dl_days = scope_deadline_days(tasks, rng)
        # remaining promised time measured from t=0 (arrived during the previous week)
        deadline = add_business_days(0, max(1, dl_days - int(rng.integers(0, 3))))
        v = Vehicle(vid, None, None, rem, modes, deadline, done=done, cohort='init')
        customer_draws(v)
        v.fault = None
        if not rem:
            v.approval_at = None
            # completed before t=0: response time counted from slot 0
        vehicles.append(v); vid += 1
    # place initial inventory: IN up to 80 % of capacity, rest OUT
    k_in = int(0.8 * park_in)
    for j, v in enumerate([v for v in vehicles if v.cohort == 'init']):
        v.init_loc = 'IN' if j < k_in else 'OUT'

    # new orders: arrivals 09:00-16:00 on business days, 5-minute grid [실험 가정]
    for d in range(days):
        if not is_business_day(d):
            continue
        times = np.sort(rng.integers(OPEN_SLOT, CLOSE_SLOT - 24, size=daily))
        for tm in times:
            arr = d * SLOTS_PER_DAY + int(tm)
            tasks = draw_tasks(rng)
            dl = add_business_days(arr, scope_deadline_days(tasks, rng))
            v = Vehicle(vid, arr, None, tasks, build_modes(tasks), dl, cohort='new')
            customer_draws(v)
            vehicles.append(v); vid += 1

    c = dict(DEFAULT_CAPS)
    if caps:
        c.update(caps)
    return Instance(H, vehicles, c, set(STAFF), dict(IN=park_in, OUT=park_out), rent_max,
                    5000.0, make_trip_types(),
                    name=name or f'daily{daily}_s{seed}',
                    meta=dict(daily=daily, seed=seed, days=days, n_init=n_init,
                              p_approval=p_approval, p_fault=p_fault))


def generate_snapshot(n: int, seed: int, days: int = 14, park_in=12, park_out=18, rent_max=12,
                      caps=None, name=None) -> Instance:
    """v5-like snapshot input (for the parking-representation comparison only): n vehicles in total,
    arriving over the first 6 business days at 09:00/13:00, no customer events.  This is a
    re-generated analogue of the v5 setting, NOT the original v5 input files."""
    rng = np.random.default_rng(seed)
    H = days * SLOTS_PER_DAY
    bdays = [d for d in range(days) if is_business_day(d)][:6]
    vehicles = []
    for i in range(n):
        d = bdays[i % len(bdays)]
        tm = OPEN_SLOT if (i // len(bdays)) % 2 == 0 else 13 * 60 // SLOT_MIN
        arr = d * SLOTS_PER_DAY + tm
        tasks = draw_tasks(rng)
        dl = add_business_days(arr, 3)
        vehicles.append(Vehicle(i, arr, None, tasks, build_modes(tasks), dl, cohort='new',
                                response_delay=12))
    c = dict(DEFAULT_CAPS)
    if caps:
        c.update(caps)
    return Instance(H, vehicles, c, set(STAFF), dict(IN=park_in, OUT=park_out), rent_max,
                    5000.0, make_trip_types(), name=name or f'snap{n}_s{seed}',
                    meta=dict(n=n, seed=seed, days=days))
