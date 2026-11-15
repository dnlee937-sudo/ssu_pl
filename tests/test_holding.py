"""Holding reference is the later of current time and arrival, never task availability."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from instance import *
from model import Window, WindowModel

def run():
    assert REPLAN_SLOT == OPEN_SLOT == 108
    t0,t1=108,396
    inst=Instance(4032, [], DEFAULT_CAPS, STAFF, {'IN':12,'OUT':18}, 15, 5000, make_trip_types())
    inst.hold_cost_day=5000
    vehicle=dict(vid=0,entry_slot=120,entry_loc='IN',tasks=['insp'],modes=build_modes(['insp']),hold=t0,deadline=300,pickup_req=None,offset=0,hold_start=120)
    M=WindowModel(Window(inst,t0,t1,[vehicle],{},np.zeros(t1-t0,int)),m=1)
    ch=5000/288
    assert np.allclose((M.cost-M.cost_base)[M.x0:M.x0+M.xv.size],ch*(M.xs+M.xp-120))
    assert abs((M.cost-M.cost_base)[M.oidx[(0,0)]]-ch*(t1-120+6))<1e-8
    print('holding checks PASS: 09 regular time and future-arrival reference in scheduled/deferred objective')
if __name__ == '__main__': run()
