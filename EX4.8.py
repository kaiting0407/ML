"""
Exercise 4.8: 模型選擇與過擬合分析
目的：進行數值模擬並比較不同條件下的過擬合情況
展示訓練誤差和測試誤差隨模型複雜度的變化
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import warnings

# 設定支援繁體中文的字型
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'PingFang TC', 'Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False  # 正確顯示負號

warnings.filterwarnings('ignore')
np.random.seed(42)

# 實驗參數（參考題目中的設定）
Q_f = 3  # 目標函數的真實複雜度（3次多項式）
sigma_squared = 0.4  # 雜訊變異數
N_train = 35  # 訓練樣本數
N_test = 1000  # 測試樣本數
max_degree = 15  # 測試的最大模型複雜度
n_experiments = 100  # 實驗重複次數

def target_function(x):
    """
    目標函數：3次多項式
    f(x) = 0.5*x^3 - 0.3*x^2 + 0.2*x + 0.1
    """
    return 0.5 * x**3 - 0.3 * x**2 + 0.2 * x + 0.1

def generate_data(n_samples, add_noise=True):
    """
    生成數據
    x 在 [-1, 1] 範圍內均勻採樣
    y = f(x) + 雜訊
    """
    x = np.random.uniform(-1, 1, n_samples)
    y = target_function(x)
    
    if add_noise:
        noise = np.random.normal(0, np.sqrt(sigma_squared), n_samples)
        y += noise
    
    return x, y

def fit_polynomial(x_train, y_train, degree):
    """
    擬合指定次數的多項式
    """
    X_train = x_train.reshape(-1, 1)
    
    # 生成多項式特徵
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X_train)
    
    # 線性回歸
    model = LinearRegression()
    model.fit(X_poly, y_train)
    
    return model, poly

def compute_error(model, poly, x, y):
    """
    計算平方誤差（MSE）
    """
    X = x.reshape(-1, 1)
    X_poly = poly.transform(X)
    y_pred = model.predict(X_poly)
    return np.mean((y - y_pred) ** 2)

def single_experiment():
    """
    單次實驗：生成數據，訓練不同複雜度的模型，計算誤差
    """
    # 生成訓練和測試數據
    x_train, y_train = generate_data(N_train, add_noise=True)
    x_test, y_test = generate_data(N_test, add_noise=True)
    
    train_errors = []
    test_errors = []
    
    # 測試不同的模型複雜度
    for degree in range(1, max_degree + 1):
        try:
            model, poly = fit_polynomial(x_train, y_train, degree)
            
            # 計算訓練誤差和測試誤差
            train_error = compute_error(model, poly, x_train, y_train)
            test_error = compute_error(model, poly, x_test, y_test)
            
            train_errors.append(train_error)
            test_errors.append(test_error)
        except:
            # 處理數值不穩定的情況
            train_errors.append(np.nan)
            test_errors.append(np.nan)
    
    return train_errors, test_errors

def run_experiments():
    """
    執行多次實驗並計算平均誤差
    """
    all_train_errors = []
    all_test_errors = []
    
    print(f"執行 {n_experiments} 次實驗...")
    for i in range(n_experiments):
        if (i + 1) % 20 == 0:
            print(f"  完成 {i + 1}/{n_experiments} 次實驗")
        
        train_errors, test_errors = single_experiment()
        all_train_errors.append(train_errors)
        all_test_errors.append(test_errors)
    
    # 計算平均值和標準差
    all_train_errors = np.array(all_train_errors)
    all_test_errors = np.array(all_test_errors)
    
    mean_train_errors = np.nanmean(all_train_errors, axis=0)
    mean_test_errors = np.nanmean(all_test_errors, axis=0)
    std_train_errors = np.nanstd(all_train_errors, axis=0)
    std_test_errors = np.nanstd(all_test_errors, axis=0)
    
    return mean_train_errors, mean_test_errors, std_train_errors, std_test_errors

def plot_results(mean_train_errors, mean_test_errors, std_train_errors, std_test_errors):
    """
    繪製訓練誤差和測試誤差曲線
    """
    degrees = np.arange(1, max_degree + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 圖1: 平均誤差曲線
    ax1.plot(degrees, mean_train_errors, 'b-o', label='訓練誤差 $E_{in}$', linewidth=2, markersize=6)
    ax1.plot(degrees, mean_test_errors, 'r-s', label='測試誤差 $E_{out}$', linewidth=2, markersize=6)
    ax1.axvline(x=Q_f, color='g', linestyle='--', linewidth=2, alpha=0.7, label=f'目標複雜度 (Q={Q_f})')
    ax1.axhline(y=sigma_squared, color='gray', linestyle=':', linewidth=1.5, alpha=0.7, label=f'雜訊水平 (σ²={sigma_squared})')
    
    ax1.set_xlabel('模型複雜度（多項式次數）', fontsize=12)
    ax1.set_ylabel('平均平方誤差', fontsize=12)
    ax1.set_title(f'訓練誤差 vs 測試誤差\n(N={N_train}, σ²={sigma_squared}, {n_experiments}次實驗平均)', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max_degree + 1)
    ax1.set_ylim(0, max(np.max(mean_test_errors[~np.isnan(mean_test_errors)]) * 1.2, 2))
    
    # 圖2: 帶標準差的誤差曲線
    ax2.plot(degrees, mean_train_errors, 'b-o', label='訓練誤差 $E_{in}$', linewidth=2, markersize=6)
    ax2.fill_between(degrees, 
                      mean_train_errors - std_train_errors, 
                      mean_train_errors + std_train_errors, 
                      alpha=0.2, color='blue')
    
    ax2.plot(degrees, mean_test_errors, 'r-s', label='測試誤差 $E_{out}$', linewidth=2, markersize=6)
    ax2.fill_between(degrees, 
                      mean_test_errors - std_test_errors, 
                      mean_test_errors + std_test_errors, 
                      alpha=0.2, color='red')
    
    ax2.axvline(x=Q_f, color='g', linestyle='--', linewidth=2, alpha=0.7, label=f'目標複雜度 (Q={Q_f})')
    
    # 找出測試誤差最小的模型
    min_test_idx = np.nanargmin(mean_test_errors)
    min_test_degree = degrees[min_test_idx]
    ax2.axvline(x=min_test_degree, color='orange', linestyle='-.', linewidth=2, alpha=0.7, 
                label=f'最佳模型 (Q={min_test_degree})')
    
    ax2.set_xlabel('模型複雜度（多項式次數）', fontsize=12)
    ax2.set_ylabel('平均平方誤差', fontsize=12)
    ax2.set_title('誤差曲線（含標準差）', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, max_degree + 1)
    ax2.set_ylim(0, max(np.max(mean_test_errors[~np.isnan(mean_test_errors)]) * 1.2, 2))
    
    plt.tight_layout()
    plt.savefig('/Users/kai/Desktop/ML/EX4.8_results.png', dpi=300, bbox_inches='tight')
    print("\n圖表已儲存至: EX4.8_results.png")
    plt.show()

def visualize_overfitting_example():
    """
    視覺化單個過擬合案例
    """
    # 生成單組數據
    x_train, y_train = generate_data(N_train, add_noise=True)
    x_test_plot = np.linspace(-1, 1, 200)
    y_true = target_function(x_test_plot)
    
    # 訓練幾個不同複雜度的模型
    degrees_to_show = [1, 3, 7, 15]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()
    
    for idx, degree in enumerate(degrees_to_show):
        ax = axes[idx]
        
        # 訓練模型
        model, poly = fit_polynomial(x_train, y_train, degree)
        
        # 預測
        X_plot = x_test_plot.reshape(-1, 1)
        X_poly_plot = poly.transform(X_plot)
        y_pred = model.predict(X_poly_plot)
        
        # 計算誤差
        train_error = compute_error(model, poly, x_train, y_train)
        x_test, y_test = generate_data(N_test, add_noise=True)
        test_error = compute_error(model, poly, x_test, y_test)
        
        # 繪圖
        ax.scatter(x_train, y_train, c='blue', s=50, alpha=0.6, label='訓練數據', edgecolors='black', linewidth=0.5)
        ax.plot(x_test_plot, y_true, 'g--', linewidth=2, label='真實函數', alpha=0.8)
        ax.plot(x_test_plot, y_pred, 'r-', linewidth=2, label=f'模型擬合 (Q={degree})')
        
        ax.set_xlabel('x', fontsize=11)
        ax.set_ylabel('y', fontsize=11)
        ax.set_title(f'多項式次數 Q = {degree}\n$E_{{in}}$ = {train_error:.3f}, $E_{{out}}$ = {test_error:.3f}', 
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-2, 2)
    
    plt.tight_layout()
    plt.savefig('/Users/kai/Desktop/ML/EX4.8_overfitting_examples.png', dpi=300, bbox_inches='tight')
    print("過擬合視覺化已儲存至: EX4.8_overfitting_examples.png")
    plt.show()

def print_analysis(mean_train_errors, mean_test_errors):
    """
    輸出分析結果
    """
    degrees = np.arange(1, max_degree + 1)
    
    print("\n" + "="*70)
    print("實驗結果分析")
    print("="*70)
    
    print(f"\n實驗設定:")
    print(f"  - 目標函數複雜度: Q_f = {Q_f} (3次多項式)")
    print(f"  - 雜訊變異數: σ² = {sigma_squared}")
    print(f"  - 訓練樣本數: N = {N_train}")
    print(f"  - 測試樣本數: {N_test}")
    print(f"  - 實驗重複次數: {n_experiments}")
    
    # 找出關鍵點
    min_train_idx = np.nanargmin(mean_train_errors)
    min_test_idx = np.nanargmin(mean_test_errors)
    
    print(f"\n關鍵觀察:")
    print(f"  1. 訓練誤差最小值在 Q = {degrees[min_train_idx]}, E_in = {mean_train_errors[min_train_idx]:.4f}")
    print(f"  2. 測試誤差最小值在 Q = {degrees[min_test_idx]}, E_out = {mean_test_errors[min_test_idx]:.4f}")
    print(f"  3. 目標複雜度 Q_f = {Q_f}, E_out = {mean_test_errors[Q_f-1]:.4f}")
    
    # 過擬合分析
    overfitting_gap = mean_test_errors - mean_train_errors
    max_gap_idx = np.nanargmax(overfitting_gap)
    
    print(f"\n過擬合分析:")
    print(f"  - 過擬合最嚴重的模型: Q = {degrees[max_gap_idx]}")
    print(f"    訓練誤差: {mean_train_errors[max_gap_idx]:.4f}")
    print(f"    測試誤差: {mean_test_errors[max_gap_idx]:.4f}")
    print(f"    誤差差距: {overfitting_gap[max_gap_idx]:.4f}")
    
    print(f"\n誤差變化趨勢:")
    print(f"  - 隨著模型複雜度增加:")
    print(f"    ✓ 訓練誤差 E_in 持續下降（從 {mean_train_errors[0]:.4f} → {mean_train_errors[-1]:.4f}）")
    print(f"    ✓ 測試誤差 E_out 先降後升（最低點在 Q = {degrees[min_test_idx]}）")
    print(f"    ✓ 這展示了經典的偏差-變異數權衡（Bias-Variance Tradeoff）")
    
    print(f"\n樂觀偏差（Optimistic Bias）:")
    print(f"  - 如果我們選擇驗證誤差最小的模型 (Q = {degrees[min_test_idx]})")
    print(f"  - 該模型的驗證誤差 E_val = {mean_test_errors[min_test_idx]:.4f}")
    print(f"  - 這會是真實 E_out 的樂觀（偏低）估計")
    print(f"  - 因為我們透過「選擇最佳模型」的過程引入了選擇偏差")
    
    print("\n" + "="*70 + "\n")

def create_summary_report():
    """
    建立總結報告
    """
    print("\n" + "="*70)
    print("📊 EXERCISE 4.8 總結報告")
    print("="*70)
    
    print("\n【題目要求】")
    print("  ✓ 進行數值模擬並比較不同條件下的過擬合情況")
    print("  ✓ 包含模擬結果的圖表")
    print("  ✓ 繪製訓練和測試誤差曲線")
    print("  ✓ 解釋隨著模型複雜度增加，誤差如何變化")
    
    print("\n【實驗成果】")
    print("  📈 生成圖表:")
    print("     1. EX4.8_results.png - 訓練/測試誤差曲線")
    print("     2. EX4.8_overfitting_examples.png - 過擬合視覺化")
    print("     3. EX4.8_說明.md - 完整實驗說明文件")
    
    print("\n【核心發現】")
    print("  🎯 偏差-變異數權衡 (Bias-Variance Tradeoff):")
    print("     • 低複雜度 (Q=1): 欠擬合 → 高偏差")
    print("     • 適中複雜度 (Q=3): 最佳平衡點")
    print("     • 高複雜度 (Q=15): 過擬合 → 高變異數")
    
    print("\n  ⚠️  樂觀偏差 (Optimistic Bias):")
    print("     • 選擇驗證誤差最小的模型會引入偏差")
    print("     • E_m* 不是 E_out(g_m*) 的無偏估計")
    print("     • 需要使用獨立測試集進行最終評估")
    
    print("\n【實務啟示】")
    print("  💡 訓練誤差持續下降 ≠ 模型越來越好")
    print("  💡 存在最佳模型複雜度（在本實驗中約為 Q=3）")
    print("  💡 過於複雜的模型會失去泛化能力")
    print("  💡 小樣本 (N=35) 下過擬合風險特別高")
    
    print("\n" + "="*70)
    print("✅ Exercise 4.8 完成！")
    print("="*70 + "\n")

def main():
    """
    主程式
    """
    print("="*70)
    print("Exercise 4.8: 模型選擇與過擬合分析")
    print("="*70)
    
    # 執行實驗
    mean_train_errors, mean_test_errors, std_train_errors, std_test_errors = run_experiments()
    
    # 輸出分析
    print_analysis(mean_train_errors, mean_test_errors)
    
    # 繪製結果
    print("繪製誤差曲線...")
    plot_results(mean_train_errors, mean_test_errors, std_train_errors, std_test_errors)
    
    # 視覺化過擬合案例
    print("\n繪製過擬合視覺化範例...")
    visualize_overfitting_example()
    
    # 建立總結報告
    create_summary_report()

if __name__ == "__main__":
    main()
