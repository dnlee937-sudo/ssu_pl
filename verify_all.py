"""Reproduce the Chapter 3 fixes, compare equations, and check execution.

All inputs here are synthetic validation examples. This does not rerun the
dissertation's large experiments or estimate changes in their improvement rates.
"""
from pathlib import Path
import copy
import json
import time

import numpy as np
import verify_outsourcing_costs as pricing  # also connects the available real HiGHS engine
import verification_cases as q
from instance import remaining_relaxation, SLOT_MIN
from paper_model_check import PaperModel
from construct import construct
from simulate import Rolling
import checker

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'validation'
OUT.mkdir(exist_ok=True)
EPS = 1e-6


def compare(label, W, m):
    M = q.WindowModel(W, m=m)
    P = PaperModel(M)
    assert np.allclose(M.cost, P.c, rtol=0, atol=EPS), label
    code, paper = M.solve(time_limit=5), P.solve(time_limit=5)
    assert code['status'] == paper['status'], (label, code['status'], paper['status'])
    assert code['status'] in ['Optimal', 'Infeasible'], (label, code['status'])
    row = dict(label=label, m=m, status=code['status'], code_objective=code['obj'],
               paper_objective=paper['obj'], objective_coefficients_equal=True,
               code_rows=M.nrow, reference_rows=len(P.rows))
    if code['x'] is not None:
        assert abs(code['obj'] - paper['obj']) < EPS, row
        for name, result in [('code', code), ('reference', paper)]:
            x = result['x']
            code_error, paper_error = q.max_error(M, x), q.max_error(P, x)
            literal = q.literal_objective(M, x)
            assert max(code_error.values()) < EPS, (label, name, code_error)
            assert max(paper_error.values()) < EPS, (label, name, paper_error)
            assert abs(float(M.cost @ x) - literal['total']) < EPS, (label, literal)
            row[name] = dict(code_residual=code_error, reference_residual=paper_error,
                             literal_objective=literal, plan=M.decode(x))
    return row


def duration_regression():
    W = next(W for label, W, _ in q.fixtures() if label == 'fast_bundle_counterexample')
    M = q.WindowModel(W)
    empty_plan = dict(modes=[], trips=[], exits=[], rent={})
    x = M.encode(empty_plan)
    assert M.max_violation(x) < EPS
    assert x[M.L0] == 35, x[M.L0]
    assert abs(M.remaining_duration[0][0] * SLOT_MIN - 35) < EPS
    # Verify telescoping even when a very fast bundle creates a negative
    # adjacent difference; clipping it would silently overestimate the suffix.
    fast = copy.deepcopy(W.vehs[0])
    fast['modes'][-1] = q.md(0, 2, 100, g=1, kind='O')
    _, suffix = remaining_relaxation(fast['tasks'], fast['modes'])
    assert suffix.tolist() == [3., 5., 1., 0.], suffix
    differences = suffix[:-1] - suffix[1:]
    assert differences.tolist() == [-2., 4., 1.]
    for start in range(3):
        assert differences[start:].sum() == suffix[start]
    Wfast = copy.deepcopy(W); Wfast.vehs[0] = fast
    negative_case = compare('negative_duration_difference', Wfast, 1)
    # Infinite deadlines still need the same completion expression for holding.
    Wfree = copy.deepcopy(W); Wfree.vehs[0]['deadline'] = q.INF
    no_deadline = compare('holding_without_finite_deadline', Wfree, 1)
    return dict(previous_minutes=45, corrected_minutes=35,
                encoded_lateness_minutes=float(x[M.L0]),
                encoded_plan_max_violation=M.max_violation(x),
                negative_marginal_case=negative_case, no_finite_deadline_case=no_deadline)


def midnight_case(t0, t1, m, fixed_cars=0, elastic=False):
    W = q.window([q.vd(tasks=[], loc='OUT')], t0=t0, t1=t1,
                 park_in=0, park_out=0, rent=1 + fixed_cars, movement=False)
    W.inst.horizon = t1
    W.inst.vehicles = [q.Vehicle(i, None, 'OUT', [], [], q.INF, response_delay=None)
                       for i in range(1 + fixed_cars)]
    W.fixed_out_park[:] = fixed_cars
    M = q.WindowModel(W, m=m, elastic=elastic)
    P = PaperModel(M)
    result = M.solve(time_limit=5)
    ref = P.solve(time_limit=5)
    assert result['status'] == ref['status'] == 'Optimal'
    assert abs(result['obj'] - ref['obj']) < EPS
    assert max(q.max_error(P, result['x']).values()) < EPS
    assert max(q.max_error(M, ref['x']).values()) < EPS
    plan = M.decode(result['x'])
    n_days = (t1 - 1) // 288 - t0 // 288 + 1
    assert result['obj'] == 25 * (1 + fixed_cars) * n_days
    assert all(plan['rent'][int(d)] == 1 + fixed_cars for d in M.days)
    earlier_rent = {d: 1 + fixed_cars for d in range(t0 // 288)}
    trajectory = dict(modes=[], trips=[], exits=[], rent={**earlier_rent, **plan['rent']})
    physical = checker.check(W.inst, trajectory, t_end=t1)
    assert physical['ok'], physical['violations']
    initial = construct(W, M)
    initial_x = M.encode(initial)
    assert M.decode(initial_x)['sigma'] == 0
    assert M.max_violation(initial_x) < EPS
    assert all(initial['rent'][int(d)] >= 1 + fixed_cars for d in M.days)
    return dict(start=t0, end=t1, m=m, fixed_cars=fixed_cars,
                objective=result['obj'], rent=plan['rent'], checker_ok=physical['ok'],
                initial_rent=initial['rent'], initial_max_violation=M.max_violation(initial_x),
                extra_calendar_rows=M.row_groups['cap_calendar'][1])


def slack_and_rejection():
    W = q.window([q.vd(tasks=[], loc='OUT')], t0=287, t1=289,
                 park_in=0, park_out=0, rent=1, movement=False)
    bad = dict(modes=[], trips=[], exits=[], rent={0: 1, 1: 0})
    hard = q.WindowModel(W, m=2)
    encoded = hard.encode(bad)
    assert hard.max_violation(encoded) >= 1
    elastic = q.WindowModel(W, m=2, elastic=True)
    encoded_elastic = elastic.encode(bad)
    assert elastic.max_violation(encoded_elastic) < EPS
    assert elastic.decode(encoded_elastic)['sigma'] == 1
    # No rentable space on either date: a hard model must report infeasible.
    W.inst.rent_max = 0
    impossible = q.WindowModel(W, m=2).solve(time_limit=5)
    assert impossible['status'] == 'Infeasible'
    return dict(missing_next_day_rent_rejected=True, elastic_overflow_detected=1,
                no_space_status=impossible['status'])


def rolling_case(m):
    vs = [q.Vehicle(0, None, 'IN', ['insp'], q.build_modes(['insp']), 200, response_delay=0),
          q.Vehicle(1, None, 'OUT', ['insp'], q.build_modes(['insp']), 200, response_delay=0),
          q.Vehicle(2, 120, None, ['insp'], q.build_modes(['insp']), 200, response_delay=0)]
    inst = q.Instance(576, vs, dict(q.DEFAULT_CAPS), q.STAFF.copy(), {'IN': 3, 'OUT': 2},
                      2, 5000., q.make_trip_types(), hold_cost_day=5000.)
    R = Rolling(inst, policy='lns', m=m, budget=8., window_days=1, t_obs=576,
                max_iter=3, repair_limit=1., seed=11)
    checked, evaluation = R.run()
    assert checked['ok'], checked['violations']
    assert len(R.traj['exits']) == 3
    return dict(m=m, checker_ok=True, replans=len(R.log),
                work_starts=len(R.traj['modes']), transfers=len(R.traj['trips']),
                deliveries=len(R.traj['exits']), lns_iterations=sum(r['lns_iter'] for r in R.log),
                evaluation=evaluation, log=R.log)


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(type(value).__name__)


def main():
    started = time.time()
    result = dict(date='2026-11-10', synthetic_inputs=True, shared_candidate_sets=True,
                  independent_reference_rows=True, full_experiments_rerun=False)
    pairs = []
    for label, W, _ in q.fixtures():
        for m in [1, 2, 3]:
            pairs.append(compare(label, copy.deepcopy(W), m))
    result['equation_pairs'] = pairs
    print('Equation comparisons:', len(pairs), 'PASS', flush=True)
    result['mutations'] = q.mutations()
    result['duration_regression'] = duration_regression()
    print('Duration and mutation regressions: PASS', flush=True)
    result['midnight'] = [midnight_case(a, b, m)
                         for a, b in [(287, 289), (287, 288), (286, 289), (288, 290)]
                         for m in [1, 2, 3]]
    result['fixed_outside_occupancy'] = [midnight_case(287, 289, m, fixed_cars=1)
                                        for m in [1, 2, 3]]
    result['slack_and_rejection'] = slack_and_rejection()
    print('Calendar and initial schedule regressions: PASS', flush=True)
    result['rolling'] = [rolling_case(m) for m in [1, 2, 3]]
    print('Rolling execution: 5/10/15 minutes PASS', flush=True)
    pricing.main()
    result['pricing_validation'] = json.loads((ROOT / 'validation_cost/cost_validation.json').read_text())
    result['summary'] = dict(equation_pairs=len(pairs),
                              optimal_pairs=sum(p['status'] == 'Optimal' for p in pairs),
                              infeasible_pairs=sum(p['status'] == 'Infeasible' for p in pairs),
                              injected_violations=len(result['mutations']),
                              calendar_cases=len(result['midnight']) + len(result['fixed_outside_occupancy']),
                              rolling_runs=len(result['rolling']),
                              source_regression_scripts=len(result['pricing_validation']['regressions']),
                              seconds=round(time.time() - started, 3))
    (OUT / 'verification_results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, default=serial))
    print(json.dumps(result['summary'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
