"""Verify the revised synthetic cost hierarchy and two complete MILP schedules.

The completion requirement below is local to the two test fixtures. Production
rolling planning still permits unfinished inventory to carry to the next window.
"""
from pathlib import Path
import contextlib
import csv
import io
import json
import runpy
import sys

import numpy as np
import scipy

try:
    import highspy
    SOLVER_BINDING = 'highspy'
except ModuleNotFoundError as exc:
    if exc.name != 'highspy':
        raise
    # The validation host has a real HiGHS engine bundled with SciPy.
    import scipy.optimize._highspy._core as highspy
    highspy.Highs = highspy._Highs
    sys.modules['highspy'] = highspy
    SOLVER_BINDING = 'SciPy bundled _highspy._core'

from instance import (CAT, BUNDLES, DEFAULT_CAPS, STAFF, Instance, Vehicle,
                      build_modes, generate, make_trip_types, remaining_relaxation)
from model import Window, WindowModel
import checker

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'validation_cost'
OUT.mkdir(exist_ok=True)


def table():
    rows = []
    for task, options in CAT.items():
        if 'I' in options and 'O' in options:
            rows.append(dict(scope=task, internal=options['I'][0], external=options['O'][0]))
    for scope, (cost, _, _) in BUNDLES.items():
        rows.append(dict(scope='+'.join(scope), internal=sum(CAT[t]['I'][0] for t in scope), external=cost))
    for row in rows:
        row['premium'] = row['external'] / row['internal'] - 1
        if row['scope'] in ('engine', 'body_h+paint_h'):
            assert row['external'] > row['internal'], row
        else:
            assert row['external'] > 0, row  # Equal, cheaper, and dearer scenarios are editable.
    return rows


def solve_case(body_workers):
    tasks = ['body_h', 'paint_h', 'insp']
    modes = build_modes(tasks)
    t0, t1 = 108, 3 * 288 + 216   # Monday 09:00 through Thursday 18:00
    caps = dict(DEFAULT_CAPS, body=body_workers)
    vehicle = Vehicle(0, t0, None, tasks, modes, t1, response_delay=None)
    inst = Instance(4 * 288, [vehicle], caps, STAFF.copy(), {'IN': 3, 'OUT': 2},
                    0, 5000., make_trip_types(), lateness_cost_min=0.,
                    pickup_wait_cost_min=0., hold_cost_day=0.)
    v = dict(vid=0, tasks=tasks, modes=modes, entry_loc='IN', entry_slot=None,
             hold=t0, deadline=t1, pickup_req=None, offset=0, hold_start=t0)
    W = Window(inst, t0, t1, [v], {}, np.zeros(t1 - t0, int))
    M = WindowModel(W, m=1)
    for col in M.oidx.values():
        M.hi[col] = 0.          # Test only: compare complete alternatives.
    result = M.solve(time_limit=10.)
    assert result['status'] == 'Optimal', result['status']
    plan = M.decode(result['x'])
    trajectory = dict(modes=[(vid, 0, modes[b], start) for vid, b, start in plan['modes']],
                      trips=plan['trips'], exits=plan['exits'], rent=plan['rent'])
    check = checker.check(inst, trajectory)
    assert check['ok'], check['violations']
    evaluation = checker.evaluate(inst, trajectory, check, inst.horizon)
    selected = [modes[b].label for _, b, _ in plan['modes']]
    expected = (sum(CAT[t]['I'][0] for t in tasks) if body_workers
                else BUNDLES[('body_h', 'paint_h')][0] + CAT['insp']['I'][0])
    assert abs(result['obj'] - expected) < 1e-6, result['obj']
    assert evaluation['work'] == expected, evaluation
    assert any(label.endswith(':O') for label in selected) == (body_workers == 0), selected
    return dict(body_workers=body_workers, status=result['status'],
                work_cost=evaluation['work'], objective=result['obj'],
                selected_modes=selected, checker_ok=check['ok'],
                max_row_violation=float(M.max_violation(result['x'])),
                final_inspection_internal='insp:I' in selected)


def main():
    rows = table()
    cost, duration = remaining_relaxation(['body_h', 'paint_h', 'insp'])
    assert cost[0] == 315000 and duration[0] == 102
    # Check every emitted outsourcing alternative in several input sizes/seeds.
    vehicles = alternatives = 0
    for daily in [10, 20, 30, 40]:
        for seed in [1, 2]:
            inst = generate(daily, seed, days=2)
            vehicles += len(inst.vehicles)
            for v in inst.vehicles:
                for md in v.modes:
                    if md.kind == 'O':
                        internal = sum(CAT[t]['I'][0] for t in v.tasks[md.f:md.g + 1])
                        scope = v.tasks[md.f:md.g + 1]
                        heavy = any(t in ('engine', 'body_h', 'paint_h') for t in scope)
                        assert (md.cost > internal if heavy else md.cost > 0), (v.vid, md.label)
                        alternatives += 1
    # An accidentally cheaper heavy bundle must fail before entering the solver.
    saved = BUNDLES[('body_h', 'paint_h')]
    try:
        BUNDLES[('body_h', 'paint_h')] = (230000, saved[1], saved[2])
        try:
            build_modes(['body_h', 'paint_h', 'insp'])
        except ValueError:
            guard_passed = True
        else:
            raise AssertionError('Invalid price hierarchy was accepted')
    finally:
        BUNDLES[('body_h', 'paint_h')] = saved
    # Existing price cells accept a slightly cheaper small/medium scenario.
    saved_wash, saved_bundle = CAT['wash']['O'], BUNDLES[('body_l', 'paint_l')]
    try:
        CAT['wash']['O'] = (42750, *saved_wash[1:])
        BUNDLES[('body_l', 'paint_l')] = (142500, *saved_bundle[1:])
        assert next(md.cost for md in build_modes(['wash', 'insp']) if md.kind == 'O') == 42750
        assert next(md.cost for md in build_modes(['body_l', 'paint_l', 'insp']) if md.kind == 'O') == 142500
        discount_scenario_passed = True
    finally:
        CAT['wash']['O'], BUNDLES[('body_l', 'paint_l')] = saved_wash, saved_bundle
    cases = [solve_case(3), solve_case(0)]
    regressions = []
    for name in ['test_holding.py', 'test_terminal.py', 'test_state_gate.py']:
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            old_argv = sys.argv
            sys.argv = [str(ROOT / 'tests' / name)]
            try:
                runpy.run_path(str(ROOT / 'tests' / name), run_name='__main__')
            except SystemExit as exc:
                if exc.code not in (None, 0):
                    raise
            finally:
                sys.argv = old_argv
        regressions.append(dict(script=name, passed=True, output=captured.getvalue()))
    report = dict(date='2026-11-10', prices_are_synthetic=True, heavy_premium_provisional=True,
                  premium_basis='Provisional heavy-work scenario; edit existing catalogue prices',
                  small_medium_prices='editable_existing_catalogue', rework_modelled=False,
                  python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                  solver=highspy.Highs().version(), binding=SOLVER_BINDING, catalogue=rows,
                  generated_vehicles=vehicles, checked_outsource_alternatives=alternatives,
                  cheap_bundle_guard_passed=guard_passed, discount_scenario_passed=discount_scenario_passed,
                  cases=cases, regressions=regressions,
                  full_experiments_rerun=False)
    (OUT / 'cost_validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    with (OUT / 'cost_table.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({k: report[k] for k in ['generated_vehicles', 'checked_outsource_alternatives',
                     'cheap_bundle_guard_passed', 'cases']}, ensure_ascii=False, indent=2))
    print('PASS: 3 regression scripts; previous full experiment results were not reused.')


if __name__ == '__main__':
    main()
