"""
檢查 Windows 系統上可用的中文字體
"""
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

# 列出所有可用字體
print("=" * 80)
print("檢查系統上可用的中文字體：")
print("=" * 80)

# 獲取所有字體
fonts = fm.findSystemFonts(fontpaths=None, fontext='ttf')

# 尋找常見的中文字體
chinese_fonts = []
target_fonts = ['Microsoft JhengHei', 'Microsoft YaHei', 'SimHei', 'DFKai-SB', 
                'Microsoft YaHei UI', 'SimSun', 'NSimSun', 'FangSong', 'KaiTi']

print("\n正在搜尋中文字體...")
for font_path in fonts:
    try:
        font_name = fm.FontProperties(fname=font_path).get_name()
        for target in target_fonts:
            if target.lower() in font_name.lower():
                chinese_fonts.append((font_name, font_path))
                print(f"✓ 找到：{font_name}")
                print(f"  路徑：{font_path}")
                break
    except:
        pass

print("\n" + "=" * 80)
if chinese_fonts:
    print(f"✓ 找到 {len(chinese_fonts)} 個中文字體")
    print("\n建議使用的字體（按優先順序）：")
    for i, (name, path) in enumerate(chinese_fonts[:3], 1):
        print(f"{i}. {name}")
else:
    print("✗ 未找到常見的中文字體")
    print("系統可能沒有安裝中文字體包")

print("=" * 80)

# 測試字體顯示
print("\n生成測試圖表...")

# 使用找到的第一個中文字體
if chinese_fonts:
    plt.rcParams['font.sans-serif'] = [chinese_fonts[0][0]]
    plt.rcParams['font.family'] = 'sans-serif'
else:
    # 使用默認設定
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Microsoft YaHei']
    plt.rcParams['font.family'] = 'sans-serif'

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'

# 創建測試圖表
fig, ax = plt.subplots(figsize=(10, 6))

x = [1, 2, 3, 4, 5]
y1 = [2, 4, 6, 8, 10]
y2 = [1, 3, 5, 7, 9]

ax.plot(x, y1, 'o-', label=r'$\lambda$ = 0.05/N（測試希臘字母）', linewidth=2, markersize=8)
ax.plot(x, y2, 's-', label=r'$\sigma$ = 0.5（測試希臘字母）', linewidth=2, markersize=8)

ax.set_xlabel('橫軸標籤（測試中文）', fontsize=14, fontweight='bold')
ax.set_ylabel('縱軸標籤（測試中文）', fontsize=14, fontweight='bold')
ax.set_title('測試圖表：中文與希臘字母顯示測試', fontsize=16, fontweight='bold')
ax.legend(fontsize=12, loc='upper left')
ax.grid(True, alpha=0.3)

# 添加文字標註
ax.text(3, 5, '這是中文標註\n包含希臘字母：λ σ μ', 
        fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        ha='center')

plt.tight_layout()
plt.savefig('c:/Users/mike2/Desktop/ML/font_test_result.png', dpi=150, bbox_inches='tight')
print("✓ 測試圖表已保存：font_test_result.png")
print("\n請檢查圖表，確認：")
print("  1. 標題、軸標籤的中文是否正常顯示（不是方框）")
print("  2. 圖例中的希臘字母 λ 和 σ 是否正常顯示")
print("  3. 文字標註中的內容是否清晰可讀")
plt.show()
