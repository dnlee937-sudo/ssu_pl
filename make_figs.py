"""Tables (markdown) and figures (PNG) for chapter 5 from results/."""
import json, os, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
FIG = os.path.join(HERE, 'results', 'fig')
os.makedirs(FIG, exist_ok=True)
for f in fm.findSystemFonts():
    if 'NotoSansCJK-Regular' in f:
        fm.fontManager.addfont(f)
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False
INK, MUTED, GRID = '#222222', '#666666', '#DDDDDD'
COL = {1: '#1F4E79', 2: '#2E86AB', 3: '#E07A1F', 6: '#8C8C8C'}
LBL = {1: '5분', 2: '10분', 3: '15분', 6: '30분'}
out = []


def style(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.spines['left'].set_color(MUTED); ax.spines['bottom'].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(axis='y', color=GRID, lw=0.6)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- E2
rows = [json.loads(l) for l in open(os.path.join(RES, 'e2_snapshot.jsonl'))]
stats = [r for r in rows if r.get('kind') == 'stats' and 'stats' in r]
runs = [r for r in rows if r.get('kind') in ('run', 'runS')]
out.append('## E2 모형 규모(스냅숏, 14일 창, 구성만)\n')
out.append('| N | 표현 | 열 | 정수 열 | y 열 | y 비중 | 행 | flow+guard+cap 행 비중 | 비영 원소 | 구성 초 | 최대 RSS MB |')
out.append('|---|---|---|---|---|---|---|---|---|---|---|')
for r in sorted(stats, key=lambda r: (r['N'], r['m'])):
    s = r['stats']
    park_rows = s['rows'].get('flow', 0) + s['rows'].get('guard', 0) + s['rows'].get('cap', 0)
    out.append(f"| {r['N']} | {LBL[r['m']]} | {s['ncol']:,} | {s['n_int']:,} | {s['n_y']:,} | {s['n_y']/s['ncol']:.1%} | "
               f"{s['nrow']:,} | {park_rows/s['nrow']:.1%} | {s['nnz']:,} | {s['build_s']:.2f} | {r['maxrss_mb']:.0f} |")
# figure: columns by N
fig, ax = plt.subplots(figsize=(6.2, 3.2), dpi=200)
for m in (1, 2, 3, 6):
    xs = [r['N'] for r in sorted(stats, key=lambda r: r['N']) if r['m'] == m]
    ys = [r['stats']['ncol'] / 1000 for r in sorted(stats, key=lambda r: r['N']) if r['m'] == m]
    ax.plot(xs, ys, marker='o', ms=4, lw=1.8, color=COL[m], label=f'주차 {LBL[m]}')
    ax.text(xs[-1] + 2, ys[-1], LBL[m], color=COL[m], va='center', fontsize=8.5)
ax.set_xlabel('차량 수(14일 스냅숏)', fontsize=9, color=INK)
ax.set_ylabel('열 수(천)', fontsize=9, color=INK)
ax.set_xticks([20, 40, 60, 80, 100]); ax.set_xlim(15, 112)
style(ax)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig5_1_size.png')); plt.close(fig)

out.append('\n## E2 실행(120초 전체 예산, 구성+충돌복구+LNS)\n')
out.append('| N | 자원 | 시드 | 표현 | 유효 | 5분 검산 | 초기 목적값 | 최종 목적값 | 5분 대비 | LNS 반복/수용 | 초기화 초 | 전체 초 | RSS MB |')
out.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
base = {(r['N'], r['seed'], r['kind']): r for r in runs if r['m'] == 1 and 'final_obj' in r}
for r in sorted(runs, key=lambda r: (r['kind'], r['N'], r['seed'], r['m'])):
    if 'final_obj' not in r:
        out.append(f"| {r['N']} | {r['seed']} | {LBL[r['m']]} | 오류 | - | - | - | - | - | - | - | - | - |")
        continue
    b = base.get((r['N'], r['seed'], r['kind']))
    res_lbl = '기준' if r['kind'] == 'run' else f"×{r['N']/40:g}"
    rel = f"{(r['final_obj'] / b['final_obj'] - 1) * 100:+.2f}%" if b and b['valid'] and r['valid'] else '-'
    out.append(f"| {r['N']} | {res_lbl} | {r['seed']} | {LBL[r['m']]} | {'예' if r['valid'] else '아니오'} | "
               f"{'통과' if r['checker_5min_ok'] else '실패'} | {r['init_obj']:,.0f} | {r['final_obj']:,.0f} | {rel} | "
               f"{r['lns_iter']}/{r['lns_acc']} | {r['init_s']:.1f} | {r['total_s']:.1f} | {r['maxrss_mb']:.0f} |")

# ---------------------------------------------------------------- E3
e3 = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RES, 'e3', '*.json')))]
if e3:
    out.append('\n## E3 일일 유입 14일 롤링\n')
    out.append('| 실행 | 차량(신규/초기) | 재계획 유효/전체 | 예산 초과 | 궤적 검산 | 경제비용 C | 내부작업 | 외주 | 운송 | 임차 | 지연비 | 지연 대수 | 완료 | 반출 | 완료 후 대기 | 미완료(납기 전/경과) | 승인 대기 | 반환 | 잔여 작업 부담 | 잔여 지연 부담 | 보관료 청구 | 내부 최대 | 외부 최대 | 누적 계획 초 |')
    out.append('|' + '---|' * 25)
    for r in e3:
        ev = r['eval']; st = ev['status']
        ret = st.get('returned_settled', 0) + st.get('returned_released', 0)
        out.append(f"| {r['name']} | {r['n_vehicles']}({r['n_new']}/{r['n_init']}) | {r['replans_valid']}/{r['replans']} | {r['replans_over_budget']} | "
                   f"{'통과' if r['checker_ok'] else '실패(' + str(r['n_viol']) + ')'} | {ev['econ']:,.0f} | {ev['work_internal']:,.0f} | {ev['work_outsourced']:,.0f} | "
                   f"{ev['trips']:,.0f} | {ev['rent']:,.0f} | {ev['late_cost']:,.0f} | {ev['late_cnt']} | {ev['completed']} | {st.get('released', 0)} | "
                   f"{st.get('done_waiting_customer', 0)} | {st.get('open_in_time', 0)}/{st.get('open_overdue', 0)} | {st.get('wait_approval', 0)} | {ret} | "
                   f"{ev['residual_work_lb']:,.0f} | {ev['residual_late_lb']:,.0f} | {ev['storage_billed']:,.0f} | {ev['max_in']} | {ev['max_out']} | {r['cum_plan_s']:.0f} |")
    for d in sorted({r['daily'] for r in e3}):
        sub = [r for r in e3 if r['daily'] == d]
        fig, ax = plt.subplots(figsize=(6.2, 3.0), dpi=200)
        pal = ['#8C8C8C', '#1F4E79', '#2E86AB', '#E07A1F']
        for k, r in enumerate(sub):
            lab = ('초기 구성' if r['policy'] == 'init' else f"초기 구성+LNS") + f" ({LBL[r['m']]})"
            inv = r['eval']['daily_inventory_17h']
            ax.plot(range(1, len(inv) + 1), inv, marker='o', ms=3, lw=1.6, color=pal[k % 4], label=lab)
        ax.set_xlabel('관측일(1일 = 월요일)', fontsize=9, color=INK)
        ax.set_ylabel('17시 주차 대기 차량(내부+외부, 대)', fontsize=9, color=INK)
        ax.set_xticks(range(1, 15))
        ax.legend(fontsize=8, frameon=False, loc='upper left')
        style(ax)
        fig.tight_layout(); fig.savefig(os.path.join(FIG, f'fig5_inv_d{d}.png')); plt.close(fig)
    # per-replan stats
    out.append('\n## E3 재계획별 통계\n')
    out.append('| 실행 | 재계획 | 평균 차량 | 평균 열 | 최대 열 | y 비중 | 평균 구성+초기화 초 | 평균 전체 초 | 최대 전체 초 | 충돌복구 발생 | LNS 반복 합 | 수용 합 | 초기→최종 목적값 개선 합 | 최대 RSS MB |')
    out.append('|' + '---|' * 14)
    for r in e3:
        L = r['log']
        cols = [x['stats']['ncol'] for x in L]
        ysh = np.mean([x['stats']['n_y'] / x['stats']['ncol'] for x in L])
        imp = sum(x['init_obj'] - x['final_obj'] for x in L if x['valid'])
        out.append(f"| {r['name']} | {len(L)} | {np.mean([x['n_veh'] for x in L]):.0f} | {np.mean(cols):,.0f} | {max(cols):,} | {ysh:.1%} | "
                   f"{np.mean([x['init_s'] for x in L]):.1f} | {np.mean([x['total_s'] for x in L]):.1f} | {max(x['total_s'] for x in L):.1f} | "
                   f"{sum(x['conflict_rounds'] > 0 for x in L)} | {sum(x['lns_iter'] for x in L)} | {sum(x['lns_acc'] for x in L)} | {imp:,.0f} | {r['maxrss_mb']:.0f} |")
open(os.path.join(RES, 'tables.md'), 'w').write('\n'.join(out))
print('\n'.join(out))
