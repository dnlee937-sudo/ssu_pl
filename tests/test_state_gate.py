"""Boundary and execution-gate regression tests; no optimisation solves."""
import sys, pathlib, unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from instance import Instance, Vehicle, DEFAULT_CAPS, STAFF, make_trip_types, build_modes
from model import WindowModel
from simulate import Rolling, ExecutionGateError


def sample(loc='OUT'):
    tasks=['insp'];v=Vehicle(0,None,loc,tasks,build_modes(tasks),200,response_delay=None)
    return Instance(288,[v],DEFAULT_CAPS.copy(),STAFF.copy(),{'IN':2,'OUT':2},3,5000,make_trip_types())


class BoundaryTests(unittest.TestCase):
    def test_commit_at_staff_departure(self):
        R=Rolling(sample(), budget=0)
        W=R.window_state(108,180);M=WindowModel(W)
        plan={'modes':[],'trips':[(0,1,120)],'exits':[],'rent':{}}
        self.assertEqual(len(R.compose(M,plan,upto=119)['trips']),1)
        self.assertEqual(len(R.compose(M,plan,upto=118)['trips']),0)

    def test_pending_source_wait_and_arrival(self):
        R=Rolling(sample(), budget=0)
        R.traj['trips']=[(0,1,120)]
        R.by_v[0]=[('trip',120,R.trip_by_id[1])]
        for t0 in (119,120,121,122):
            W=R.window_state(t0,180)
            self.assertEqual(int(W.fixed_out_park.sum()),max(0,120-t0))
            v=W.vehs[0]
            self.assertEqual(v['entry_loc'],'IN')
            self.assertEqual(v['entry_slot'],122 if t0<122 else None)
            for m in (1,2,3):
                M=WindowModel(W,m=m,elastic=True)
                plan=R.construct(W,M);sol=M.encode(plan)
                self.assertLessEqual(M.max_violation(sol),1e-6)
                self.assertTrue(all(s>=122 for _,_,s in plan['modes']))

    def test_invalid_plan_stops_without_commit(self):
        R=Rolling(sample('IN'),policy='init',budget=0)
        with patch.object(R,'plan_check',return_value=({'ok':False,'violations':['injected']},{})):
            with self.assertRaises(ExecutionGateError):R.replan(108,216)
        self.assertEqual(R.traj['modes'],[])
        self.assertEqual(R.log[-1]['stop_reason'],'no_validated_plan')

    def test_invalid_prefix_stops_without_commit(self):
        R=Rolling(sample('IN'),policy='init',budget=0)
        with patch.object(R,'plan_check',return_value=({'ok':True,'violations':[]},{})):
            with patch('simulate.checker.check',return_value={'ok':False,'violations':['injected prefix']}):
                with self.assertRaises(ExecutionGateError):R.replan(108,216)
        self.assertEqual(R.traj['modes'],[])
        self.assertEqual(R.log[-1]['stop_reason'],'invalid_execution_prefix')

    def test_valid_prefix_commits(self):
        R=Rolling(sample('IN'),policy='init',budget=0)
        rec=R.replan(108,216)
        self.assertTrue(rec['prefix_check']);self.assertEqual(len(R.traj['modes']),1)

if __name__=='__main__':unittest.main()
