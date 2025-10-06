import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import warnings

# 設定支援繁體中文的字型
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'PingFang TC', 'Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False  # 正確顯示負號

# Suppress overflow warnings (handled by weight normalization)
warnings.filterwarnings('ignore', category=RuntimeWarning)

np.random.seed(42)

# Suppress overflow warnings (handled by weight normalization)
warnings.filterwarnings('ignore', category=RuntimeWarning)

np.random.seed(42)

@dataclass
class Dataset:
    X: np.ndarray  # 形狀 (N, 3) - 特徵矩陣
    y: np.ndarray  # 形狀 (N,) - 標籤
    w_target: np.ndarray  # 形狀 (3,) - 目標權重

def make_linearly_separable_target():
    p1, p2 = np.random.uniform(-1, 1, size=(2, 2))
    # Ensure p1 and p2 are not too close
    while np.linalg.norm(p2 - p1) < 0.1:
        p2 = np.random.uniform(-1, 1, size=2)
    dx, dy = p2 - p1
    a, b = dy, -dx
    c = -(a * p1[0] + b * p1[1])
    w = np.array([c, a, b], dtype=float)
    n = np.linalg.norm(w)
    if n > 1e-10:
        w = w / n
    else:
        # Fallback to a simple target function
        w = np.array([0.0, 1.0, 1.0])
        w = w / np.linalg.norm(w)
    return w

def generate_dataset_from_target(w_target, N=100, noise_rate=0.1):
    X2 = np.random.uniform(-1, 1, size=(N, 2))
    X = np.column_stack([np.ones(N), X2])
    raw = X @ w_target
    y = np.where(raw >= 0, 1, -1)
    n_flip = max(1, int(round(noise_rate * N)))
    flip_idx = np.random.choice(N, size=n_flip, replace=False)
    y_noisy = y.copy()
    y_noisy[flip_idx] *= -1
    return Dataset(X=X, y=y_noisy, w_target=w_target)

def sign01(z):
    return np.where(z >= 0, 1, -1)

def pocket_perceptron(X, y, T=1000):
    N, d = X.shape
    w = np.zeros(d)
    preds = sign01(X @ w)
    best_err = (preds != y).mean()
    w_best = w.copy()

    w_hist = np.zeros((T, d))
    wbest_hist = np.zeros((T, d))
    ein_hist = np.zeros(T)
    ein_pocket_hist = np.zeros(T)

    for t in range(T):
        mis_idx = np.where(sign01(X @ w) != y)[0]
        if mis_idx.size > 0:
            i = np.random.choice(mis_idx)
            w = w + y[i] * X[i]
            # Prevent overflow: normalize if weights get too large
            w_norm = np.linalg.norm(w)
            if w_norm > 1e3:
                w = w / w_norm * 1e3
        ein = (sign01(X @ w) != y).mean()
        if ein < best_err:
            best_err = ein
            w_best = w.copy()
        w_hist[t] = w
        wbest_hist[t] = w_best
        ein_hist[t] = ein
        ein_pocket_hist[t] = best_err
    return w_hist, wbest_hist, ein_hist, ein_pocket_hist

# 實驗設定
N_train = 100      # 訓練集大小
N_test = 1000      # 測試集大小
T = 1000           # 每次實驗的迭代次數
runs = 20          # 重複實驗次數
noise_rate = 0.1   # 雜訊率（標籤翻轉比例）

Ein_curves = np.zeros((runs, T))
Ein_pocket_curves = np.zeros((runs, T))
Eout_curves = np.zeros((runs, T))
Eout_pocket_curves = np.zeros((runs, T))
illustrative = None

for r in range(runs):
    w_target = make_linearly_separable_target()
    ds_train = generate_dataset_from_target(w_target, N=N_train, noise_rate=noise_rate)
    ds_test = generate_dataset_from_target(w_target, N=N_test, noise_rate=noise_rate)

    w_hist, wbest_hist, ein_hist, ein_pocket_hist = pocket_perceptron(ds_train.X, ds_train.y, T=T)

    logits_cur = ds_test.X @ w_hist.T  # (N_test, T)
    logits_poc = ds_test.X @ wbest_hist.T
    ytest = ds_test.y[:, None]
    eout_hist = (np.where(logits_cur >= 0, 1, -1) != ytest).mean(axis=0)
    eout_pocket_hist = (np.where(logits_poc >= 0, 1, -1) != ytest).mean(axis=0)

    Ein_curves[r] = ein_hist
    Ein_pocket_curves[r] = ein_pocket_hist
    Eout_curves[r] = eout_hist
    Eout_pocket_curves[r] = eout_pocket_hist

    if illustrative is None:
        illustrative = {
            'X2': ds_train.X[:, 1:3],
            'y': ds_train.y,
            'w_pocket': wbest_hist[-1].copy(),
            'w_last': w_hist[-1].copy(),
            'w_target': w_target.copy()
        }

# 計算統計數據
mean_Ein = Ein_curves.mean(axis=0)
mean_Ein_poc = Ein_pocket_curves.mean(axis=0)
std_Ein = Ein_curves.std(axis=0)
std_Ein_poc = Ein_pocket_curves.std(axis=0)

mean_Eout = Eout_curves.mean(axis=0)
mean_Eout_poc = Eout_pocket_curves.mean(axis=0)
std_Eout = Eout_curves.std(axis=0)
std_Eout_poc = Eout_pocket_curves.std(axis=0)

summary = {
    'Ein_current_T_mean': float(mean_Ein[-1]),
    'Ein_current_T_std': float(std_Ein[-1]),
    'Ein_pocket_T_mean': float(mean_Ein_poc[-1]),
    'Ein_pocket_T_std': float(std_Ein_poc[-1]),
    'Eout_current_T_mean': float(mean_Eout[-1]),
    'Eout_current_T_std': float(std_Eout[-1]),
    'Eout_pocket_T_mean': float(mean_Eout_poc[-1]),
    'Eout_pocket_T_std': float(std_Eout_poc[-1]),
}

print("\n=== Exercise 3.2 實驗結果 ===")
print(f"訓練集大小：{N_train}，測試集大小：{N_test}")
print(f"實驗次數：{runs} 次，每次迭代數：{T}")
print(f"雜訊率：{noise_rate * 100:.1f}%\n")
print("最終結果（在 T={} 時）".format(T))
print("-" * 60)
print(f"Ein（當前 w(T)）：  {summary['Ein_current_T_mean']:.4f} ± {summary['Ein_current_T_std']:.4f}")
print(f"Ein（Pocket ŵ）：   {summary['Ein_pocket_T_mean']:.4f} ± {summary['Ein_pocket_T_std']:.4f}")
print(f"Eout（當前 w(T)）： {summary['Eout_current_T_mean']:.4f} ± {summary['Eout_current_T_std']:.4f}")
print(f"Eout（Pocket ŵ）：  {summary['Eout_pocket_T_mean']:.4f} ± {summary['Eout_pocket_T_std']:.4f}")
print("-" * 60)

# --- 視覺化 ---
fig = plt.figure(figsize=(16, 5))

# 圖表 1：訓練資料分佈與三條分隔線
ax1 = fig.add_subplot(1, 3, 1)
X2 = illustrative['X2']
y_vis = illustrative['y']
w_pocket = illustrative['w_pocket']
w_last = illustrative['w_last']
w_target = illustrative['w_target']

pos_idx = y_vis == 1
neg_idx = y_vis == -1
ax1.scatter(X2[pos_idx, 0], X2[pos_idx, 1], c='blue', marker='o', s=30, alpha=0.6, label='正類（y=+1）')
ax1.scatter(X2[neg_idx, 0], X2[neg_idx, 1], c='red', marker='x', s=30, alpha=0.6, label='負類（y=-1）')

# 繪製決策邊界
x_line = np.array([-1, 1])

def plot_line(w, color, label, linestyle='-'):
    if abs(w[2]) > 1e-10:
        y_line = -(w[0] + w[1] * x_line) / w[2]
        ax1.plot(x_line, y_line, color=color, linestyle=linestyle, linewidth=2, label=label)
    elif abs(w[1]) > 1e-10:
        x_val = -w[0] / w[1]
        ax1.axvline(x=x_val, color=color, linestyle=linestyle, linewidth=2, label=label)

plot_line(w_target, 'green', '目標函數', linestyle='--')
plot_line(w_last, 'orange', 'PLA 當前 w(T)', linestyle='-.')
plot_line(w_pocket, 'purple', 'Pocket 最佳 ŵ', linestyle='-')

ax1.set_xlim(-1, 1)
ax1.set_ylim(-1, 1)
ax1.set_xlabel('x₁')
ax1.set_ylabel('x₂')
ax1.set_title('訓練資料分佈\n（單次示例）')
ax1.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=8, framealpha=0.9)
ax1.grid(True, alpha=0.3)

# 圖表 2：Ein 曲線
ax2 = fig.add_subplot(1, 3, 2)
iterations = np.arange(1, T + 1)
ax2.plot(iterations, mean_Ein, color='blue', label='Ein(w(t)) - 當前權重', linewidth=2)
ax2.fill_between(iterations, mean_Ein - std_Ein, mean_Ein + std_Ein, color='blue', alpha=0.2)
ax2.plot(iterations, mean_Ein_poc, color='red', label='Ein(ŵ) - Pocket 權重', linewidth=2)
ax2.fill_between(iterations, mean_Ein_poc - std_Ein_poc, mean_Ein_poc + std_Ein_poc, color='red', alpha=0.2)
ax2.set_xlabel('迭代次數 (t)')
ax2.set_ylabel('訓練誤差 (Ein)')
ax2.set_title(f'{runs} 次實驗的平均 Ein\n（陰影區域 = ±1 標準差）')
ax2.legend(loc='best', fontsize=9)
ax2.grid(True, alpha=0.3)

# 圖表 3：Eout 曲線
ax3 = fig.add_subplot(1, 3, 3)
ax3.plot(iterations, mean_Eout, color='blue', label='Eout(w(t)) - 當前權重', linewidth=2)
ax3.fill_between(iterations, mean_Eout - std_Eout, mean_Eout + std_Eout, color='blue', alpha=0.2)
ax3.plot(iterations, mean_Eout_poc, color='red', label='Eout(ŵ) - Pocket 權重', linewidth=2)
ax3.fill_between(iterations, mean_Eout_poc - std_Eout_poc, mean_Eout_poc + std_Eout_poc, color='red', alpha=0.2)
ax3.set_xlabel('迭代次數 (t)')
ax3.set_ylabel('測試誤差 (Eout)')
ax3.set_title(f'{runs} 次實驗的平均 Eout\n（陰影區域 = ±1 標準差）')
ax3.legend(loc='best', fontsize=9)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/kai/Desktop/ML/EX3.2_results.png', dpi=150, bbox_inches='tight')
print(f"\n圖表已儲存至：/Users/kai/Desktop/ML/EX3.2_results.png")
plt.show()

print("\n=== 結果分析 ===")
print("1. Ein 比較：")
print(f"   - Pocket 演算法維持較低的 Ein（{summary['Ein_pocket_T_mean']:.4f}），優於當前 w(T)（{summary['Ein_current_T_mean']:.4f}）")
print("2. Eout 比較：")
print(f"   - Pocket ŵ 達到更好的泛化能力：Eout = {summary['Eout_pocket_T_mean']:.4f}")
print(f"   - 當前 w(T) 有較高的 Eout：{summary['Eout_current_T_mean']:.4f}")
print("3. Pocket 演算法優勢：")
print("   - 透過保留歷史最佳假設，Pocket 避免被最近的錯誤分類點影響")
print(f"   - Pocket 的 Ein 與 Eout 差距較小（{abs(summary['Eout_pocket_T_mean'] - summary['Ein_pocket_T_mean']):.4f}），代表更好的泛化能力")
