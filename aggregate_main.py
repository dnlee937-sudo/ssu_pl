"""본실험 결과 집계: results/main/*.json -> results/main/summary/{runs.csv, groups.csv, paired.csv, tables.md}

  python3 aggregate_main.py               # 완료된 실행만으로 집계(중간 점검에도 사용 가능)
평균만 보고하지 않는다. 중앙값·범위와 함께 실패·미완료를 포함한 분모를 적는다.
"""
import csv, glob, json, os, sys
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
IN = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'results', 'main')
OUT = os.path.join(IN, 'summary')
os.makedirs(OUT, exist_ok=True)

DAILY = (10, 20, 30, 40); CONDS = ('base', 'staff', 'park'); SEEDS = (1, 2, 3, 4, 5)
POLS = ('init_m1', 'lns_m1', 'lns_m2', 'lns_m3')
COND_KO = dict(base='기준', staff='인력 부족', park='주차 부족')
POL_KO = dict(init_m1='규칙 기반(5분)', lns_m1='LNS 5분', lns_m2='LNS 10분', lns_m3='LNS 15분')


def row(d):
    e = d['eval']; st = e.get('status', {})
    resid = e['residual_work_lb'] + e['residual_late_lb']
    open_n = sum(st.get(k, 0) for k in ('open_in_time', 'open_overdue', 'wait_approval', 'wait_fault_decision'))
    comp = e['completed']
    return dict(name=d['name'], daily=d['daily'], cond=d['cond'], seed=d['seed'],
                pol=f"{d['policy'] if d['policy'] == 'lns' else 'init'}_m{d['m']}",
                n_vehicles=d['n_vehicles'], checker_ok=d['checker_ok'], n_viol=d['n_viol'],
                replans=d['replans'], replans_valid=d['replans_valid'], over_budget=d['replans_over_budget'],
                econ=e['econ'], hold=e.get('hold_cost', 0.0), residual=resid,
                econ_hold=e['econ'] + e.get('hold_cost', 0.0), total=e['econ'] + e.get('hold_cost', 0.0) + resid,
                work=e['work'], outsourced=e['work_outsourced'], trips=e['trips'], rent=e['rent'],
                late_cost=e['late_cost'], pickup_wait=e['pickup_wait_cost'],
                completed=comp, released=st.get('released', 0), open_end=open_n,
                open_overdue=st.get('open_overdue', 0), done_waiting=st.get('done_waiting_customer', 0),
                returned=st.get('returned_settled', 0) + st.get('returned_released', 0),
                ontime_rate=(e['ontime'] / comp) if comp else float('nan'), late_min=e['late_min'],
                unfinished_days=e.get('unfinished_vehicle_days', float('nan')),
                storage_billed=e['storage_billed'], max_in=e['max_in'], max_out=e['max_out'],
                cum_plan_s=d['cum_plan_s'], wall_s=d['wall_s'], maxrss_mb=d['maxrss_mb'])


rows = []
for f in sorted(glob.glob(os.path.join(IN, '*.json'))):
    try:
        rows.append(row(json.load(open(f))))
    except Exception as ex:
        print('건너뜀', f, ex)
if not rows:
    sys.exit('결과 파일이 없습니다: ' + IN)
keys = list(rows[0])
with open(os.path.join(OUT, 'runs.csv'), 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)
R = {(r['daily'], r['cond'], r['seed'], r['pol']): r for r in rows}


def mr(v, fmt='{:,.0f}'):
    v = [x for x in v if x == x]
    if not v:
        return '-'
    return f"{fmt.format(np.median(v))} [{fmt.format(min(v))}, {fmt.format(max(v))}]"


# ------------------------------------------------------------------ group table
groups = []
for d in DAILY:
    for c in CONDS:
        for p in POLS:
            rs = [R[(d, c, s, p)] for s in SEEDS if (d, c, s, p) in R]
            if not rs:
                continue
            ok = [r for r in rs if r['checker_ok']]
            groups.append(dict(daily=d, cond=c, pol=p, done=len(rs), planned=len(SEEDS), checker_ok=len(ok),
                               replans_valid=sum(r['replans_valid'] for r in rs), replans=sum(r['replans'] for r in rs),
                               econ=mr([r['econ'] for r in ok]), econ_hold=mr([r['econ_hold'] for r in ok]),
                               total=mr([r['total'] for r in ok]), completed=mr([r['completed'] for r in ok], '{:.0f}'),
                               open_end=mr([r['open_end'] for r in ok], '{:.0f}'),
                               ontime=mr([100 * r['ontime_rate'] for r in ok], '{:.1f}'),
                               plan_h=mr([r['cum_plan_s'] / 3600 for r in rs], '{:.2f}'),
                               rss=mr([r['maxrss_mb'] for r in rs], '{:.0f}')))
with open(os.path.join(OUT, 'groups.csv'), 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, list(groups[0])); w.writeheader(); w.writerows(groups)

# ------------------------------------------------------------------ paired comparisons (same scenario)
PAIRS = [('lns_m1', 'init_m1'), ('lns_m2', 'lns_m1'), ('lns_m3', 'lns_m1')]
paired = []
for (a, b) in PAIRS:
    for d in DAILY:
        for c in CONDS:
            dif = defaultdict(list); n_both = 0; n_missing = 0; n_fail = 0
            for s in SEEDS:
                ra, rb = R.get((d, c, s, a)), R.get((d, c, s, b))
                if ra is None or rb is None:
                    n_missing += 1; continue
                if not (ra['checker_ok'] and rb['checker_ok']):
                    n_fail += 1; continue
                n_both += 1
                for k in ('econ', 'econ_hold', 'total'):
                    dif[k].append(100 * (ra[k] - rb[k]) / rb[k])
                dif['completed'].append(ra['completed'] - rb['completed'])
                dif['open_end'].append(ra['open_end'] - rb['open_end'])
                dif['valid_rate'].append(ra['replans_valid'] / ra['replans'] - rb['replans_valid'] / rb['replans'])
            if n_both == 0 and n_fail == 0:
                continue
            paired.append(dict(compare=f'{a} vs {b}', daily=d, cond=c, pairs=n_both, check_fail=n_fail, missing=n_missing,
                               d_econ_pct=mr(dif['econ'], '{:+.2f}'), d_econ_hold_pct=mr(dif['econ_hold'], '{:+.2f}'),
                               d_total_pct=mr(dif['total'], '{:+.2f}'),
                               improved_total=sum(x < 0 for x in dif['total']),
                               d_completed=mr(dif['completed'], '{:+.0f}'), d_open_end=mr(dif['open_end'], '{:+.0f}'),
                               d_valid_rate=mr(dif['valid_rate'], '{:+.2f}')))
if paired:
    with open(os.path.join(OUT, 'paired.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, list(paired[0])); w.writeheader(); w.writerows(paired)

# ------------------------------------------------------------------ markdown tables
L = ['# 본실험 집계', '', f'완료 실행 {len(rows)} / 240. 값은 중앙값 [최소, 최대]이며 검산 통과 실행만 비용에 포함한다.',
     '총평가 = 경제비용 C + 보유비용 + 잔여 의무(하한).', '',
     '## 조건·정책별 결과', '',
     '| 일일 | 자원 | 정책 | 완료/계획 | 검산 통과 | 유효 재계획 | C(원) | C+보유 | 총평가 | 완료 대수 | 기말 미완료 | 납기 준수(%) | 누적 계획(시간) |',
     '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
for g in groups:
    L.append(f"| {g['daily']} | {COND_KO[g['cond']]} | {POL_KO[g['pol']]} | {g['done']}/{g['planned']} | {g['checker_ok']} | "
             f"{g['replans_valid']}/{g['replans']} | {g['econ']} | {g['econ_hold']} | {g['total']} | {g['completed']} | "
             f"{g['open_end']} | {g['ontime']} | {g['plan_h']} |")
L += ['', '## 같은 시나리오의 짝지은 차이', '', '차이(%) = (앞 정책 − 뒤 정책) / 뒤 정책. 음수가 개선.', '',
      '| 비교 | 일일 | 자원 | 짝 수 | 검산 실패 | ΔC(%) | Δ(C+보유)(%) | Δ총평가(%) | 총평가 개선 짝 | Δ완료 대수 | Δ기말 미완료 | Δ유효 재계획 비율 |',
      '|---|---|---|---|---|---|---|---|---|---|---|---|']
for p in paired:
    L.append(f"| {p['compare']} | {p['daily']} | {COND_KO[p['cond']]} | {p['pairs']} | {p['check_fail']} | {p['d_econ_pct']} | "
             f"{p['d_econ_hold_pct']} | {p['d_total_pct']} | {p['improved_total']}/{p['pairs']} | {p['d_completed']} | "
             f"{p['d_open_end']} | {p['d_valid_rate']} |")
open(os.path.join(OUT, 'tables.md'), 'w').write('\n'.join(L) + '\n')
print('\n'.join(L[:12]))
print(f'... 저장: {OUT}/runs.csv, groups.csv, paired.csv, tables.md')
