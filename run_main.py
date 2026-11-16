"""본실험(표 5-11) 실행기 — 여러 실행을 동시에 돌리는 로컬 실행용.

실험 격자
  일일 신규 오더  10·20·30·40대 (영업일당)
  자원조건        base(기준), staff(인력 부족), park(주차 부족)
  시나리오        1∼5 (입고·작업·사건·응답의 공통 난수 = 생성 시드)
  정책            init_m1(규칙 기반 초기 구성, 5분), lns_m1·lns_m2·lns_m3(LNS, 5·10·15분)
  → 4 × 3 × 5 × 4 = 240회.  LNS 탐색시드 11 고정, 재계획 1회 120초, 1스레드.

단계(stage)
  A : init_m1 + lns_m1          (120회, 핵심 비교: 규칙 기반 대 LNS)
  B : lns_m2 + lns_m3           (120회, 주차 표현 비교)
  all : A + B

사용 예
  python3 run_main.py list --stage A                # 실행 목록만 출력
  python3 run_main.py run --stage A --jobs 6        # 6개 동시 실행
  python3 run_main.py run --stage all --jobs 6
  python3 run_main.py smoke --jobs 2                # 설치 확인(짧은 예산, 2일)
  python3 run_main.py status                        # 진행 현황
  python3 run_main.py one d20_base_s3_lns_m1        # 한 실행만(내부용)

결과: results/main/<이름>.json (완료 시 원자적 저장), 로그: results/main/logs/<이름>.log
이미 결과 파일이 있는 실행은 건너뛰므로 중단 후 같은 명령으로 이어서 실행한다.
"""
import argparse, json, os, platform, resource, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results', 'main')
LOGS = os.path.join(OUT, 'logs')

# ------------------------------------------------------------------ 고정 설정 (결과를 보기 전에 고정)
DAILY = (10, 20, 30, 40)
CONDS = ('base', 'staff', 'park')
SEEDS = (1, 2, 3, 4, 5)
POLICIES = {'init_m1': ('init', 1), 'lns_m1': ('lns', 1), 'lns_m2': ('lns', 2), 'lns_m3': ('lns', 3),
            'lnsc_m1': ('lns_carry', 1)}         # lnsc: optional variant that carries the previous plan
STAGES = {'A': ('init_m1', 'lns_m1'), 'B': ('lns_m2', 'lns_m3'), 'C': ('lnsc_m1',),
          'all': ('init_m1', 'lns_m1', 'lns_m2', 'lns_m3')}
BUDGET = 120.0            # 재계획 1회 전체 예산(초)
LNS_SEED = 11
# 보유비용: 미완료 차량 1대·1일. 외부 주차 1면·1일 임차비와 같은 값으로 둔다 [실험 가정].
HOLD_COST_DAY = 5000.0
# 자원: 일일 10대 기준의 합성 자원(직원 15명, 내부 12면, 외부 18면, 추가 임차 상한 15면)을
#        일일 오더 수에 비례해 늘린다(scale = daily/10) [실험 가정].
STAFF_GROUPS = ('gen', 'mech', 'body', 'insp')
SHORT_FACTOR = 0.5        # 부족 조건: 해당 자원을 기준의 50%로(올림, 최소 1) [실험 가정]


def resources(daily, cond):
    from instance import DEFAULT_CAPS
    sc = daily / 10
    caps = {k: max(1, int(round(v * sc))) for k, v in DEFAULT_CAPS.items()}
    park_in, park_out, rent_max = int(round(12 * sc)), int(round(18 * sc)), int(round(15 * sc))
    if cond == 'staff':
        for g in STAFF_GROUPS:
            caps[g] = max(1, int(np.ceil(caps[g] * SHORT_FACTOR)))
    elif cond == 'park':
        park_in = max(1, int(np.ceil(park_in * SHORT_FACTOR)))
    elif cond[:2] in ('pk', 'st') and cond[2:].isdigit():
        # resource-level sweep: pkNN = internal parking at NN %, stNN = every staff group at NN %
        f = int(cond[2:]) / 100.0
        if cond[:2] == 'pk':
            park_in = max(1, int(np.ceil(park_in * f)))
        else:
            for g in STAFF_GROUPS:
                caps[g] = max(1, int(np.ceil(caps[g] * f)))
    return caps, park_in, park_out, rent_max


def run_name(daily, cond, seed, pol):
    return f'd{daily}_{cond}_s{seed}_{pol}'


def parse_name(nm):
    d, cond, s, p1, p2 = nm.split('_')
    return int(d[1:]), cond, int(s[1:]), f'{p1}_{p2}'


def plan(stage, daily=None, conds=None, seeds=None):
    runs = []
    for d in (daily or DAILY):
        for c in (conds or CONDS):
            for s in (seeds or SEEDS):
                for p in STAGES[stage]:
                    runs.append(run_name(d, c, s, p))
    # 오래 걸리는 실행(큰 물량, LNS)을 먼저 배치해 끝부분의 대기를 줄인다
    runs.sort(key=lambda n: (-parse_name(n)[0], parse_name(n)[3].startswith('init'), n))
    return runs


def conv(o):
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.bool_): return bool(o)
    if isinstance(o, np.ndarray): return o.tolist()
    return str(o)


def one(nm, budget=BUDGET, obs_days=None, outdir=OUT):
    from instance import generate, SLOTS_PER_DAY
    from simulate import Rolling
    daily, cond, seed, pol = parse_name(nm)
    policy, m = POLICIES[pol]
    caps, park_in, park_out, rent_max = resources(daily, cond)
    inst = generate(daily, seed, park_in=park_in, park_out=park_out, rent_max=rent_max, caps=caps)
    inst.hold_cost_day = HOLD_COST_DAY
    t_obs = obs_days * SLOTS_PER_DAY if obs_days else None
    t = time.time()
    R = Rolling(inst, policy=policy, m=m, budget=budget, seed=LNS_SEED, verbose=True, t_obs=t_obs)
    chk, ev = R.run()
    log = []
    for r in R.log:
        r = dict(r); r.pop('history', None)
        log.append(r)
    res = dict(name=nm, daily=daily, cond=cond, seed=seed, policy=policy, m=m, budget=budget,
               hold_cost_day=HOLD_COST_DAY, caps=caps, park_in=park_in, park_out=park_out, rent_max=rent_max,
               n_vehicles=len(inst.vehicles), meta=inst.meta,
               n_new=sum(v.cohort == 'new' for v in inst.vehicles),
               n_init=sum(v.cohort == 'init' for v in inst.vehicles),
               checker_ok=chk['ok'], n_viol=chk['n_viol'], violations=chk['violations'][:10],
               excess_in_space_min=chk['excess_in_space_min'], excess_in_slots=chk['excess_in_slots'],
               tech_trip_persons=sum(sum(c.values()) for c in R.traj['crew'].values()),
               staff_persons=chk['staff_persons'], eval=ev, replans=len(R.log),
               replans_valid=sum(r['valid'] for r in R.log),
               replans_over_budget=sum(r['total_s'] > budget + 1 for r in R.log),
               cum_plan_s=sum(r['total_s'] for r in R.log), wall_s=time.time() - t,
               maxrss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * (1024 if sys.platform == 'darwin' else 1)),
               log=log,
               trajectory=dict(modes=[(vid, off, md.label, md.f, md.g, s) for (vid, off, md, s) in R.traj['modes']],
                               trips=R.traj['trips'], exits=R.traj['exits'],
                               rent={str(k): v for k, v in R.traj['rent'].items()},
                               crew={f'{k}@{s}': c for (k, s), c in R.traj['crew'].items()}))
    os.makedirs(outdir, exist_ok=True)
    fn = os.path.join(outdir, nm + '.json')
    tmp = fn + '.tmp'
    json.dump(res, open(tmp, 'w'), default=conv, indent=1)
    os.replace(tmp, fn)                     # 완료된 실행만 결과 파일로 남긴다
    print('DONE', nm, 'check', res['checker_ok'], 'valid', res['replans_valid'], '/', res['replans'],
          'C', round(ev['econ']), flush=True)
    return res


def launch(runs, jobs, outdir, extra=()):
    os.makedirs(os.path.join(outdir, 'logs'), exist_ok=True)
    todo = [n for n in runs if not os.path.exists(os.path.join(outdir, n + '.json'))]
    print(f'전체 {len(runs)}회, 완료 {len(runs) - len(todo)}회, 남은 실행 {len(todo)}회, 동시 실행 {jobs}', flush=True)
    meta = dict(started=time.strftime('%Y-%m-%d %H:%M:%S'), jobs=jobs, n_runs=len(runs), python=sys.version,
                platform=platform.platform(), machine=platform.machine(), processor=platform.processor(),
                cpu_count=os.cpu_count(), budget=BUDGET, hold_cost_day=HOLD_COST_DAY, lns_seed=LNS_SEED)
    try:
        import highspy, scipy
        meta.update(numpy=np.__version__, scipy=scipy.__version__,
                    highspy=getattr(highspy, '__version__', 'unknown'))
    except Exception:
        pass
    with open(os.path.join(outdir, 'run_meta.jsonl'), 'a') as f:
        f.write(json.dumps(meta, ensure_ascii=False) + '\n')
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    t0 = time.time()

    def work(nm):
        lf = os.path.join(outdir, 'logs', nm + '.log')
        t = time.time()
        with open(lf, 'w') as f:
            rc = subprocess.run([sys.executable, os.path.abspath(__file__), 'one', nm, '--outdir', outdir, *extra],
                                stdout=f, stderr=subprocess.STDOUT, cwd=HERE, env=env).returncode
        return nm, rc, time.time() - t

    done = 0
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = [ex.submit(work, n) for n in todo]
        for fu in as_completed(futs):
            nm, rc, el = fu.result()
            done += 1
            ok = os.path.exists(os.path.join(outdir, nm + '.json'))
            print(f'[{done}/{len(todo)}] {nm}  {"완료" if ok else "실패(rc=%d, 로그 확인)" % rc}  {el / 60:.1f}분  '
                  f'경과 {(time.time() - t0) / 3600:.2f}시간', flush=True)


def status(outdir=OUT):
    runs = plan('all')
    have = [n for n in runs if os.path.exists(os.path.join(outdir, n + '.json'))]
    print(f'완료 {len(have)} / {len(runs)}')
    for st in ('A', 'B'):
        r = plan(st)
        print(f'  단계 {st}: {sum(n in have for n in r)} / {len(r)}')
    running = [f[:-4] for f in os.listdir(os.path.join(outdir, 'logs'))] if os.path.isdir(os.path.join(outdir, 'logs')) else []
    running = [n for n in running if n not in have]
    if running:
        print('진행 중 또는 실패(결과 없음, 로그 있음):')
        for n in sorted(running):
            lf = os.path.join(outdir, 'logs', n + '.log')
            last = open(lf).read().strip().splitlines()[-1:] if os.path.getsize(lf) else ['(로그 없음)']
            print(f'  {n}: {last[0][:120]}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='본실험 실행기')
    ap.add_argument('cmd', choices=['list', 'run', 'smoke', 'status', 'one'])
    ap.add_argument('name', nargs='?')
    ap.add_argument('--stage', default='A', choices=list(STAGES))
    ap.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument('--daily', type=str, default=None, help='예: 10,20')
    ap.add_argument('--cond', type=str, default=None, help='예: base,park')
    ap.add_argument('--seeds', type=str, default=None, help='예: 1,2')
    ap.add_argument('--outdir', default=OUT)
    ap.add_argument('--budget', type=float, default=BUDGET)
    ap.add_argument('--obs-days', type=int, default=None)
    a = ap.parse_args()
    sel = dict(daily=[int(x) for x in a.daily.split(',')] if a.daily else None,
               conds=a.cond.split(',') if a.cond else None,
               seeds=[int(x) for x in a.seeds.split(',')] if a.seeds else None)
    if a.cmd == 'list':
        rr = plan(a.stage, **sel)
        print('\n'.join(rr)); print(f'# {len(rr)}회')
    elif a.cmd == 'run':
        launch(plan(a.stage, **sel), a.jobs, a.outdir)
    elif a.cmd == 'smoke':
        # 설치·병렬 동작 확인: 2관측일, 재계획 예산 15초, 결과는 results/smoke 에 따로 저장
        rr = [run_name(10, 'base', 1, 'init_m1'), run_name(10, 'base', 1, 'lns_m1'),
              run_name(20, 'staff', 1, 'lns_m3'), run_name(40, 'park', 1, 'lns_m2')]
        launch(rr, a.jobs, os.path.join(HERE, 'results', 'smoke'), extra=('--budget', '15', '--obs-days', '2'))
    elif a.cmd == 'status':
        status(a.outdir)
    elif a.cmd == 'one':
        one(a.name, budget=a.budget, obs_days=a.obs_days, outdir=a.outdir)
