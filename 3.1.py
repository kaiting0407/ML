import numpy as np
import matplotlib.pyplot as plt
import math

# 設定 matplotlib 支援繁體中文字型，避免中文亂碼
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial Unicode MS', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

#設定隨機種子以確保可重現性
rng = np.random.default_rng(42)

#題目3.1的參數設定
rad = 10.0
thk = 5.0
sep = 5.0
N_total = 2000
N_per_class = N_total // 2

R_inner = rad
R_outer = rad + thk

def sample_semi_ring(n, angle_low, angle_high, center=(0.0, 0.0)):
    #在圓環帶內進行面積均勻採樣：
    #r = sqrt(u * (R_outer^2 - R_inner^2) + R_inner^2)
    u = rng.random(n)
    r = np.sqrt(u * (R_outer**2 - R_inner**2) + R_inner**2)
    theta = rng.uniform(angle_low, angle_high, size=n)
    x = r * np.cos(theta) + center[0]
    y = r * np.sin(theta) + center[1]
    return np.column_stack([x, y])

#幾何結構：上半圓中心在(0,0)，θ在[0,π]；下半圓中心在(0,-sep)，θ在[π, 2π]
X_pos = sample_semi_ring(N_per_class, 0.0, math.pi, (0.0, 0.0))        # y = +1
X_neg = sample_semi_ring(N_per_class, math.pi, 2*math.pi, (0.0, -sep)) # y = -1

X = np.vstack([X_pos, X_neg])
y = np.concatenate([np.ones(N_per_class), -np.ones(N_per_class)])

#打亂資料順序
perm = rng.permutation(N_total)
X = X[perm]; y = y[perm]

#增加bias項(常數 1)
X_aug = np.hstack([np.ones((N_total, 1)), X])

#感知器學習法(PLA)
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

w_pla, updates_pla, converged_pla = pla(X_aug, y)
pred_pla = np.sign(X_aug @ w_pla); pred_pla[pred_pla == 0] = -1
acc_pla = (pred_pla == y).mean()

#線性回歸分類
w_lin, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
pred_lin = np.sign(X_aug @ w_lin); pred_lin[pred_lin == 0] = -1
acc_lin = (pred_lin == y).mean()

print('PLA converged:', converged_pla)
print('PLA updates:', updates_pla)
print('PLA training accuracy:', acc_pla)
print('Linear Regression training accuracy:', acc_lin)

#繪圖
x_min, x_max = X[:,0].min() - 2, X[:,0].max() + 2
y_min, y_max = X[:,1].min() - 2, X[:,1].max() + 2

def scatter_data(ax):
    #繪製資料點
    ax.scatter(X[y==1,0],  X[y==1,1],  s=10, c='#1f77b4', label='+1 (上半圓)', alpha=0.8)
    ax.scatter(X[y==-1,0], X[y==-1,1], s=10, c='#d62728', label='-1 (下半圓)', alpha=0.8)
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x1'); ax.set_ylabel('x2')

def plot_boundary(ax, w, color, label):
    #繪製決策邊界
    if abs(w[2]) < 1e-12:
        xv = -w[0]/w[1]
        ax.plot([xv, xv], [y_min, y_max], color=color, linestyle='--', label=label)
    else:
        xs = np.linspace(x_min, x_max, 400)
        ys = -(w[0] + w[1]*xs)/w[2]
        ax.plot(xs, ys, color=color, linestyle='--', label=label)

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
scatter_data(axes[0]); plot_boundary(axes[0], w_pla, color='black', label=f'PLA boundary (updates={updates_pla})')
axes[0].set_title(f'PLA (準確率={acc_pla*100:.2f}%)'); axes[0].legend(loc='upper right', fontsize=8)

scatter_data(axes[1]); plot_boundary(axes[1], w_lin, color='green', label='Linear Regression boundary')
axes[1].set_title(f'線性回歸 (準確率={acc_lin*100:.2f}%)'); axes[1].legend(loc='upper right', fontsize=8)

#自動調整子圖間距並顯示圖形
plt.tight_layout()
plt.show()
