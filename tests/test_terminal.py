"""Focused regression checks for bundle pricing and ongoing work accounting."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from instance import remaining_relaxation, build_modes, generate, Vehicle, DEFAULT_CAPS, STAFF, make_trip_types, Instance
from checker import evaluate

def run():
    c, p = remaining_relaxation(['body_h', 'paint_h', 'insp'])
    # Internal heavy body+paint (300,000) is cheaper than outsourcing (360,000).
    # The final internal inspection costs 15,000 on either route.
    assert c[0] == 315000, c
    assert p[0] == 102, p  # 240 + 240 + 30 minutes on unconstrained internal route
    c, p = remaining_relaxation(['paint_h', 'insp'])
    assert c[0] == 155000 and p[0] == 54
    assert remaining_relaxation([])[0][0] == 0
    v = Vehicle(vid=1, arrival=None, init_loc='IN', tasks=['maint', 'insp'], modes=build_modes(['maint','insp']), deadline=128)
    inst = Instance(288, [v], DEFAULT_CAPS, STAFF, {'IN':12,'OUT':18}, 15, 5000, make_trip_types())
    inst.hold_cost_day = 5000
    md = next(md for md in v.modes if md.label == 'maint:I')
    trajectory = dict(modes=[(1,0,md,108)], trips=[], exits=[], rent={})
    chk = dict(vrec={1:dict(completion=None, exit=None)}, occ_in=np.zeros(110,int), occ_out=np.zeros(110,int))
    e=evaluate(inst,trajectory,chk,110)
    assert e['work'] == 45000
    assert e['residual_work_lb'] == 15000
    assert e['residual_late_lb'] == 400  # internal work starts at 09:00 (slot 108), ready 126, inspection 6 slots; deadline 128
    assert abs(e['hold_cost'] - 110/288*5000) < 1e-8
    parked = Vehicle(vid=2, arrival=None, init_loc='OUT', tasks=[], modes=[], deadline=120)
    inst.vehicles=[parked]
    tr=dict(modes=[],trips=[(2,1,120)],exits=[],rent={})
    ck=dict(vrec={2:dict(completion=0,exit=None)},occ_in=np.zeros(119,int),occ_out=np.zeros(119,int))
    e=evaluate(inst,tr,ck,119)
    assert e['trips']==2000 and e['n_staff_trips']==1
    print('terminal checks PASS: bundle price, suffix cover, empty tasks, paid ongoing mode, outstanding duration, holding units')

if __name__ == '__main__': run()
