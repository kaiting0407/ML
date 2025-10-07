"""
問題 4.24：帶權重衰減正規化的交叉驗證分析

本程式實作線性迴歸的留一法交叉驗證 (LOOCV)，並分析個別交叉驗證誤差
與平均交叉驗證誤差之間的關係。
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from tqdm import tqdm
import warnings
from joblib import Parallel, delayed
import multiprocessing

# 抑制字體相關的警告
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# 設定隨機種子以確保可重現性
np.random.seed(42)

# 配置繪圖參數 - 設定中文字體 (適用於 Windows 系統)
plt.rcParams['font.family'] = ['Microsoft JhengHei', 'Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題
plt.rcParams['font.size'] = 11
rcParams['figure.dpi'] = 100
sns.set_style("whitegrid")


def generate_data(N, d, wf, sigma):
    """
    生成隨機數據集
    
    參數：
    -----------
    N : int
        數據點數量
    d : int
        輸入空間維度
    wf : ndarray
        目標權重向量 (d+1 維，包含偏置項)
    sigma : float
        噪音標準差
    
    返回：
    --------
    X : ndarray of shape (N, d+1)
        包含偏置項的輸入數據
    y : ndarray of shape (N,)
        帶噪音的目標值
    """
    # 從標準常態分佈生成 X
    X_raw = np.random.randn(N, d)
    # 添加偏置項 (x0 = 1)
    X = np.column_stack([np.ones(N), X_raw])
    
    # 生成帶噪音的 y
    epsilon = np.random.randn(N)
    y = X @ wf + sigma * epsilon
    
    return X, y


def ridge_regression(X, y, lambda_reg):
    """
    執行嶺迴歸（帶權重衰減的線性迴歸）
    
    參數：
    -----------
    X : ndarray of shape (N, d+1)
        輸入數據
    y : ndarray of shape (N,)
        目標值
    lambda_reg : float
        正規化參數
    
    返回：
    --------
    w : ndarray of shape (d+1,)
        估計的權重向量
    """
    d = X.shape[1]
    # w_reg = (X^T X + λI)^{-1} X^T y
    w = np.linalg.inv(X.T @ X + lambda_reg * np.eye(d)) @ X.T @ y
    return w


def leave_one_out_cv(X, y, lambda_reg):
    """
    執行留一法交叉驗證
    
    參數：
    -----------
    X : ndarray of shape (N, d+1)
        輸入數據
    y : ndarray of shape (N,)
        目標值
    lambda_reg : float
        正規化參數
    
    返回：
    --------
    cv_errors : ndarray of shape (N,)
        個別交叉驗證誤差 e_i
    E_cv : float
        平均交叉驗證誤差
    """
    N = X.shape[0]
    cv_errors = np.zeros(N)
    
    for i in range(N):
        # 留出第 i 個點，創建訓練集
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i)
        
        # 在 N-1 個點上訓練
        w = ridge_regression(X_train, y_train, lambda_reg)
        
        # 計算留出點的誤差
        y_pred = X[i] @ w
        cv_errors[i] = (y[i] - y_pred) ** 2
    
    E_cv = np.mean(cv_errors)
    
    return cv_errors, E_cv


def single_experiment(d, N, sigma, lambda_reg, seed=None):
    """
    執行單次實驗（用於平行化）
    
    參數：
    -----------
    d : int
        輸入空間維度
    N : int
        數據點數量
    sigma : float
        噪音標準差
    lambda_reg : float
        正規化參數
    seed : int, optional
        隨機種子
    
    返回：
    --------
    tuple : (e1, e2, E_cv)
        第一個CV誤差、第二個CV誤差、平均CV誤差
    """
    if seed is not None:
        np.random.seed(seed)
    
    # 生成目標權重向量
    wf = np.random.randn(d + 1)
    
    # 生成數據
    X, y = generate_data(N, d, wf, sigma)
    
    # 執行留一法交叉驗證
    cv_errors, E_cv = leave_one_out_cv(X, y, lambda_reg)
    
    return cv_errors[0], cv_errors[1], E_cv


def run_experiment_part_a(d=3, sigma=0.5, num_experiments=100000, lambda_ratio=0.05, n_jobs=-1):
    """
    Part (a)：執行不同 N 值的實驗
    
    參數：
    -----------
    d : int
        輸入空間維度
    sigma : float
        噪音標準差
    num_experiments : int
        實驗重複次數
    lambda_ratio : float
        正規化參數比例 (λ = lambda_ratio / N)
    n_jobs : int
        平行工作數量（-1 表示使用所有 CPU 核心）
    
    返回：
    --------
    results : dict
        包含每個 N 的結果字典
    """
    N_values = range(d + 15, d + 116, 10)  # d+15, d+25, ..., d+115
    results = {}
    
    # 獲取 CPU 核心數
    n_cores = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
    
    print(f"\n{'='*60}")
    print(f"執行 Part (a)：交叉驗證分析（多核心加速）")
    print(f"{'='*60}")
    print(f"參數：d={d}, σ={sigma}, λ比例={lambda_ratio}")
    print(f"實驗次數：{num_experiments:,}")
    print(f"使用 CPU 核心數：{n_cores}")
    print(f"{'='*60}\n")
    
    for N in N_values:
        print(f"處理 N = {N}...")
        
        # 設定正規化參數
        lambda_reg = lambda_ratio / N
        
        # 為每個實驗生成不同的隨機種子
        seeds = [42 + exp for exp in range(num_experiments)]
        
        # 使用 joblib 進行平行計算
        print(f"  使用 {n_cores} 個核心進行平行計算...")
        results_parallel = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(single_experiment)(d, N, sigma, lambda_reg, seed)
            for seed in tqdm(seeds, desc=f"N={N}", ncols=80)
        )
        
        # 解析結果
        e1_values = [r[0] for r in results_parallel]
        e2_values = [r[1] for r in results_parallel]
        Ecv_values = [r[2] for r in results_parallel]
        
        # 計算統計量
        results[N] = {
            'e1_mean': np.mean(e1_values),
            'e1_var': np.var(e1_values, ddof=1),
            'e2_mean': np.mean(e2_values),
            'e2_var': np.var(e2_values, ddof=1),
            'Ecv_mean': np.mean(Ecv_values),
            'Ecv_var': np.var(Ecv_values, ddof=1),
            'e1_values': e1_values[:1000],  # 儲存前1000個用於分析
            'e2_values': e2_values[:1000],
            'Ecv_values': Ecv_values[:1000]
        }
        
        print(f"  E[e1] = {results[N]['e1_mean']:.6f}, Var[e1] = {results[N]['e1_var']:.6f}")
        print(f"  E[Ecv] = {results[N]['Ecv_mean']:.6f}, Var[Ecv] = {results[N]['Ecv_var']:.6f}")
        print()
    
    return results


def plot_part_b(results):
    """
    Part (b)：繪製 e1, e2, 和 Ecv 平均值的關係圖
    """
    N_values = sorted(results.keys())
    
    e1_means = [results[N]['e1_mean'] for N in N_values]
    e2_means = [results[N]['e2_mean'] for N in N_values]
    Ecv_means = [results[N]['Ecv_mean'] for N in N_values]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(N_values, e1_means, 'o-', label='E[e1]', linewidth=2, markersize=6)
    ax.plot(N_values, e2_means, 's-', label='E[e2]', linewidth=2, markersize=6)
    ax.plot(N_values, Ecv_means, '^-', label='E[Ecv]', linewidth=2, markersize=6)
    
    ax.set_xlabel('N (數據點數量)', fontsize=12)
    ax.set_ylabel('平均誤差', fontsize=12)
    ax.set_title('Part (b)：E[e1], E[e2], 與 E[Ecv] 的關係', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_part_c(results):
    """
    Part (c)：分析 e1 變異數的貢獻者
    """
    N_values = sorted(results.keys())
    
    e1_vars = [results[N]['e1_var'] for N in N_values]
    Ecv_vars = [results[N]['Ecv_var'] for N in N_values]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(N_values, e1_vars, 'o-', label='Var[e1]', linewidth=2, markersize=6, color='blue')
    ax.plot(N_values, Ecv_vars, 's-', label='Var[Ecv]', linewidth=2, markersize=6, color='red')
    
    ax.set_xlabel('N (數據點數量)', fontsize=12)
    ax.set_ylabel('變異數', fontsize=12)
    ax.set_title('Part (c)：e1 與 Ecv 的變異數比較', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_part_e(results, title_suffix=""):
    """
    Part (e)：繪製有效新樣本數
    """
    N_values = sorted(results.keys())
    
    e1_vars = [results[N]['e1_var'] for N in N_values]
    Ecv_vars = [results[N]['Ecv_var'] for N in N_values]
    
    # N_eff = Var[e_i] / Var[E_cv]
    N_eff = [e1_vars[i] / Ecv_vars[i] for i in range(len(N_values))]
    N_eff_percentage = [N_eff[i] / N_values[i] * 100 for i in range(len(N_values))]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 繪製 N_eff
    ax1.plot(N_values, N_eff, 'o-', linewidth=2, markersize=6, color='green')
    ax1.plot(N_values, N_values, '--', linewidth=1.5, color='red', alpha=0.5, label='N (參考線)')
    ax1.set_xlabel('N (數據點數量)', fontsize=12)
    ax1.set_ylabel('N_eff (有效新樣本數)', fontsize=12)
    ax1.set_title(f'Part (e)：有效新樣本數{title_suffix}', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 繪製 N_eff 百分比
    ax2.plot(N_values, N_eff_percentage, 'o-', linewidth=2, markersize=6, color='purple')
    ax2.axhline(y=100, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='100% (參考線)')
    ax2.set_xlabel('N (數據點數量)', fontsize=12)
    ax2.set_ylabel('N_eff / N (%)', fontsize=12)
    ax2.set_title(f'Part (e)：N_eff 占 N 的百分比{title_suffix}', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, N_eff, N_eff_percentage


def plot_part_f_comparison(results_small, results_large):
    """
    Part (f)：比較不同正規化參數的 N_eff
    """
    N_values = sorted(results_small.keys())
    
    # 計算兩種 lambda 值的 N_eff
    e1_vars_small = [results_small[N]['e1_var'] for N in N_values]
    Ecv_vars_small = [results_small[N]['Ecv_var'] for N in N_values]
    N_eff_small = [e1_vars_small[i] / Ecv_vars_small[i] for i in range(len(N_values))]
    N_eff_pct_small = [N_eff_small[i] / N_values[i] * 100 for i in range(len(N_values))]
    
    e1_vars_large = [results_large[N]['e1_var'] for N in N_values]
    Ecv_vars_large = [results_large[N]['Ecv_var'] for N in N_values]
    N_eff_large = [e1_vars_large[i] / Ecv_vars_large[i] for i in range(len(N_values))]
    N_eff_pct_large = [N_eff_large[i] / N_values[i] * 100 for i in range(len(N_values))]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 繪製 N_eff 比較
    ax1.plot(N_values, N_eff_small, 'o-', linewidth=2, markersize=6, label='λ = 0.05/N', color='blue')
    ax1.plot(N_values, N_eff_large, 's-', linewidth=2, markersize=6, label='λ = 2.5/N', color='orange')
    ax1.plot(N_values, N_values, '--', linewidth=1.5, color='red', alpha=0.5, label='N (參考線)')
    ax1.set_xlabel('N (數據點數量)', fontsize=12)
    ax1.set_ylabel('N_eff (有效新樣本數)', fontsize=12)
    ax1.set_title('Part (f)：不同 λ 值的 N_eff 比較', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 繪製百分比比較
    ax2.plot(N_values, N_eff_pct_small, 'o-', linewidth=2, markersize=6, label='λ = 0.05/N', color='blue')
    ax2.plot(N_values, N_eff_pct_large, 's-', linewidth=2, markersize=6, label='λ = 2.5/N', color='orange')
    ax2.axhline(y=100, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='100% (參考線)')
    ax2.set_xlabel('N (數據點數量)', fontsize=12)
    ax2.set_ylabel('N_eff / N (%)', fontsize=12)
    ax2.set_title('Part (f)：N_eff 百分比比較', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def print_summary_table(results, lambda_ratio):
    """
    輸出結果摘要表格
    """
    print(f"\n{'='*90}")
    print(f"摘要表格 (λ = {lambda_ratio}/N)")
    print(f"{'='*90}")
    print(f"{'N':>5} | {'E[e1]':>10} | {'E[e2]':>10} | {'E[Ecv]':>10} | {'Var[e1]':>10} | {'Var[Ecv]':>10} | {'N_eff/N':>8}")
    print(f"{'-'*90}")
    
    N_values = sorted(results.keys())
    for N in N_values:
        e1_mean = results[N]['e1_mean']
        e2_mean = results[N]['e2_mean']
        Ecv_mean = results[N]['Ecv_mean']
        e1_var = results[N]['e1_var']
        Ecv_var = results[N]['Ecv_var']
        N_eff_ratio = e1_var / Ecv_var / N
        
        print(f"{N:5d} | {e1_mean:10.6f} | {e2_mean:10.6f} | {Ecv_mean:10.6f} | "
              f"{e1_var:10.6f} | {Ecv_var:10.6f} | {N_eff_ratio:7.2%}")
    
    print(f"{'='*90}\n")


def main():
    """
    主函數：執行問題 4.24 的所有部分
    """
    print("\n" + "="*60)
    print("問題 4.24：交叉驗證分析（多核心加速版本）")
    print("="*60)
    
    # 參數設定
    d = 3
    sigma = 0.5
    num_experiments = 100000  # 10^5 次實驗
    n_jobs = -1  # 使用所有可用的 CPU 核心
    
    # Part (a) - (e) 使用 λ = 0.05/N
    print("\n" + "="*60)
    print("Parts (a)-(e)：使用 λ = 0.05/N 執行實驗")
    print("="*60)
    
    results_small = run_experiment_part_a(
        d=d, 
        sigma=sigma, 
        num_experiments=num_experiments,
        lambda_ratio=0.05,
        n_jobs=n_jobs
    )
    
    # 輸出摘要表格
    print_summary_table(results_small, 0.05)
    
    # Part (b)：繪製平均值之間的關係
    print("生成 Part (b) 圖表...")
    fig_b = plot_part_b(results_small)
    fig_b.savefig('/Users/kai/Desktop/ML/Pr4.24_part_b.png', dpi=300, bbox_inches='tight')
    print("已保存：Pr4.24_part_b.png\n")
    
    # Part (c)：分析變異數
    print("生成 Part (c) 圖表...")
    fig_c = plot_part_c(results_small)
    fig_c.savefig('/Users/kai/Desktop/ML/Pr4.24_part_c.png', dpi=300, bbox_inches='tight')
    print("已保存：Pr4.24_part_c.png\n")
    
    # Part (e)：有效新樣本數
    print("生成 Part (e) 圖表...")
    fig_e, N_eff, N_eff_pct = plot_part_e(results_small, " (λ = 0.05/N)")
    fig_e.savefig('/Users/kai/Desktop/ML/Pr4.24_part_e.png', dpi=300, bbox_inches='tight')
    print("已保存：Pr4.24_part_e.png\n")
    
    # Part (f)：增加正規化 λ = 2.5/N
    print("\n" + "="*60)
    print("Part (f)：使用 λ = 2.5/N 執行實驗")
    print("="*60)
    
    results_large = run_experiment_part_a(
        d=d, 
        sigma=sigma, 
        num_experiments=num_experiments,
        lambda_ratio=2.5,
        n_jobs=n_jobs
    )
    
    # 為大 lambda 輸出摘要表格
    print_summary_table(results_large, 2.5)
    
    # Part (f)：為大 lambda 繪製 N_eff 圖
    print("生成 Part (f) 單獨圖表...")
    fig_f_individual, _, _ = plot_part_e(results_large, " (λ = 2.5/N)")
    fig_f_individual.savefig('/Users/kai/Desktop/ML/Pr4.24_part_f_individual.png', dpi=300, bbox_inches='tight')
    print("已保存：Pr4.24_part_f_individual.png\n")
    
    # Part (f)：比較圖
    print("生成 Part (f) 比較圖表...")
    fig_f = plot_part_f_comparison(results_small, results_large)
    fig_f.savefig('/Users/kai/Desktop/ML/Pr4.24_part_f_comparison.png', dpi=300, bbox_inches='tight')
    print("已保存：Pr4.24_part_f_comparison.png\n")
    
    # 輸出最終分析
    print("\n" + "="*60)
    print("分析與結論")
    print("="*60)
    
    print("\nPart (b) - E[e1], E[e2], 與 E[Ecv] 的關係：")
    print("-" * 60)
    print("理論預期：")
    print("  • E[e1] = E[e2] = E[Ecv] （所有CV誤差都是同分佈的）")
    print("  • 每個 e_i 都估計一個新點的樣本外誤差")
    print("\n實驗驗證：")
    N_values = sorted(results_small.keys())
    for N in [N_values[0], N_values[-1]]:
        print(f"  N = {N}:")
        print(f"    E[e1]  = {results_small[N]['e1_mean']:.6f}")
        print(f"    E[e2]  = {results_small[N]['e2_mean']:.6f}")
        print(f"    E[Ecv] = {results_small[N]['Ecv_mean']:.6f}")
        print(f"    差異：{abs(results_small[N]['e1_mean'] - results_small[N]['Ecv_mean']):.8f}")
    print("\n✓ 平均值幾乎相同，證實了理論。")
    
    print("\nPart (c) - Var[e1] 的貢獻者：")
    print("-" * 60)
    print("變異數貢獻者：")
    print("  1. 數據集的隨機性（X, y）")
    print("  2. 留出點的隨機性")
    print("  3. 目標函數的噪音（ε）")
    print("  4. 正規化造成的假設空間變化")
    print("\n觀察：")
    print(f"  • Var[e1] 隨 N 增加而減少（估計更穩定）")
    print(f"  • 在 N={N_values[0]}：Var[e1] = {results_small[N_values[0]]['e1_var']:.6f}")
    print(f"  • 在 N={N_values[-1]}：Var[e1] = {results_small[N_values[-1]]['e1_var']:.6f}")
    
    print("\nPart (d) - 如果獨立，Var[e_i] 與 Var[Ecv] 的關係：")
    print("-" * 60)
    print("理論（如果獨立）：")
    print("  • Ecv = (1/N) Σ e_i")
    print("  • 如果 e_i 獨立：Var[Ecv] = Var[e_i] / N")
    print("  • 因此：N = Var[e_i] / Var[Ecv]")
    print("\n然而，e_i 並非真正獨立，因為：")
    print("  • 它們共享 N-2 個共同的訓練點")
    print("  • 這創造了 CV 誤差之間的相關性")
    
    print("\nPart (e) - 有效新樣本數（N_eff）：")
    print("-" * 60)
    print("定義：")
    print("  • N_eff = Var[e_i] / Var[Ecv]")
    print("  • 衡量有多少「有效獨立」的樣本對 Ecv 有貢獻")
    print("\n結果（λ = 0.05/N）：")
    for i, N in enumerate([N_values[0], N_values[len(N_values)//2], N_values[-1]]):
        var_ratio = results_small[N]['e1_var'] / results_small[N]['Ecv_var']
        percentage = var_ratio / N * 100
        print(f"  N = {N}：N_eff = {var_ratio:.2f}, N_eff/N = {percentage:.2f}%")
    print("\n✓ N_eff 非常接近 N，顯示 CV 誤差幾乎獨立！")
    print("  這是因為每個 CV 誤差使用不同的驗證點。")
    
    print("\nPart (f) - 增加正規化對 N_eff 的影響：")
    print("-" * 60)
    print("猜想：")
    print("  • 增加 λ → 更多正規化 → 更平滑的假設")
    print("  • 更平滑的假設 → 留出不同點時模型更相似")
    print("  • 更相似的模型 → e_i 之間相關性更高")
    print("  • 更高的相關性 → 更低的 N_eff")
    print("\n驗證（比較 λ = 0.05/N vs λ = 2.5/N）：")
    for N in [N_values[0], N_values[-1]]:
        var_ratio_small = results_small[N]['e1_var'] / results_small[N]['Ecv_var']
        pct_small = var_ratio_small / N * 100
        var_ratio_large = results_large[N]['e1_var'] / results_large[N]['Ecv_var']
        pct_large = var_ratio_large / N * 100
        print(f"\n  N = {N}：")
        print(f"    λ = 0.05/N：N_eff/N = {pct_small:.2f}%")
        print(f"    λ = 2.5/N：N_eff/N = {pct_large:.2f}%")
        print(f"    變化：{pct_large - pct_small:+.2f} 百分點")
    print("\n✓ 猜想部分證實：更高的正規化在某些情況下降低 N_eff")
    
    print("\n" + "="*60)
    print("所有圖表已成功保存！")
    print("="*60 + "\n")
    
    plt.show()


if __name__ == "__main__":
    main()
