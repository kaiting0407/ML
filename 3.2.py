import numpy as np
import matplotlib.pyplot as plt
import math

# 設定 matplotlib 支援繁體中文字型，避免中文亂碼
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial Unicode MS', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False


# 設定隨機種子以確保可重現性
rng = np.random.default_rng(12345)

# 固定的參數（承襲自 Problem 3.1）
rad = 10.0
thk = 5.0
N_total = 2000
N_per_class = N_total // 2
R_inner = rad
R_outer = rad + thk

# 工具函式：在半圓環帶內進行面積均勻採樣
def sample_semi_ring(n, angle_low, angle_high, center=(0.0, 0.0), rng=None):
    if rng is None:
        rng = np.random.default_rng()
    u = rng.random(n)
    r = np.sqrt(u * (R_outer**2 - R_inner**2) + R_inner**2)
    theta = rng.uniform(angle_low, angle_high, size=n)
    x = r * np.cos(theta) + center[0]
    y = r * np.sin(theta) + center[1]
    return np.column_stack([x, y])

# 感知器學習法（PLA）
def pla(X_aug, y, max_updates=1_000_000):
    w = np.zeros(X_aug.shape[1])
    updates = 0
    n = X_aug.shape[0]
    while updates < max_updates:
        error_found = False
        for i in range(n):
            if y[i] * (w @ X_aug[i]) <= 0:
                w = w + y[i] * X_aug[i]
                updates += 1
                error_found = True
                break
        if not error_found:
            break
    converged = not error_found
    return w, updates, converged

# 掃描 sep 於 {0.2, 0.4, ..., 5.0}
sep_values = np.round(np.arange(0.2, 5.0 + 1e-9, 0.2), 1)
iterations = []
converged_flags = []

for sep in sep_values:
    # 針對每個 sep 產生資料
    center_top = (0.0, 0.0)
    center_bottom = (0.0, -sep)
    X_pos = sample_semi_ring(N_per_class, 0.0, math.pi, center_top, rng)
    X_neg = sample_semi_ring(N_per_class, math.pi, 2*math.pi, center_bottom, rng)
    X = np.vstack([X_pos, X_neg])
    y = np.concatenate([np.ones(N_per_class), -np.ones(N_per_class)])

    # 打亂資料順序
    perm = rng.permutation(N_total)
    X = X[perm]
    y = y[perm]

    # 增加 bias 項（常數 1）
    X_aug = np.hstack([np.ones((N_total, 1)), X])

    # 執行 PLA
    w, updates, converged = pla(X_aug, y)
    iterations.append(updates)
    converged_flags.append(converged)

# --- Plot: linear y ---
plt.figure(figsize=(8,5))
plt.plot(sep_values, iterations, marker='o', lw=1.5, color='#1f77b4')
plt.xlabel('sep')
plt.ylabel('PLA 收斂所需更新次數')
plt.title('Problem 3.2: sep vs PLA 收斂更新次數')
plt.grid(True, alpha=0.3)
fig_path1 = 'problem3_2_sep_vs_iterations.png'
plt.tight_layout(); plt.savefig(fig_path1, dpi=160); plt.close()

# --- Plot: log y (更清楚看趨勢) ---
plt.figure(figsize=(8,5))
plt.semilogy(sep_values, iterations, marker='o', lw=1.5, color='#d62728')
plt.xlabel('sep')
plt.ylabel('PLA 收斂更新次數（對數刻度）')
plt.title('Problem 3.2: sep vs PLA 收斂更新次數（log-y）')
plt.grid(True, which='both', alpha=0.3)
fig_path2 = 'problem3_2_sep_vs_iterations_logy.png'
plt.tight_layout(); plt.savefig(fig_path2, dpi=160); plt.close()

#（可選）列印部分數據摘要
import pandas as pd
summary = pd.DataFrame({'sep': sep_values, 'pla_updates': iterations, 'converged': converged_flags})
print(summary.head(10).to_string(index=False))
print('...')
print(summary.tail(5).to_string(index=False))
print('All converged?', all(converged_flags))
print('Saved plots:', fig_path1, fig_path2)