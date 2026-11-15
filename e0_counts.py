"""E0: conditional arithmetic of dense per-vehicle parking-state counts (not model statistics).
N x |P| x (K+1) state points, 14 days = 20,160 min, |P| = 2.  Compared with the only reported
v5 model size (Appendix B.2: 80 vehicles, 755,182 variables, 1,068,359 constraints)."""
import json, os
rows = []
for d in (5, 10, 15, 30):
    K = 20160 // d
    rows.append(dict(delta_min=d, K=K, y_100_state=100 * 2 * (K + 1), y_80_state=80 * 2 * (K + 1),
                     y_100_interval=100 * 2 * K))
v5_total_vars_80 = 755182
v5_rows_80 = 1068359
y80 = 80 * 2 * 4033
out = dict(table=rows, v5_80=dict(total_vars=v5_total_vars_80, total_rows=v5_rows_80, dense_y=y80,
                                  dense_y_share=y80 / v5_total_vars_80,
                                  dense_flow_rows=y80, dense_flow_row_share=y80 / v5_rows_80,
                                  dense_y_plus_C2d_rows=(y80 + 80 * 4032) / v5_rows_80),
           note='y defined for every vehicle, location and state point as in v5 C2 (t in T^state); '
                'C2c gives one flow row per y; C2d one recovery row per vehicle and interval. '
                'Presolve and the real code may differ.')
os.makedirs('results', exist_ok=True)
json.dump(out, open('results/e0_counts.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
