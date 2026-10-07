import os
import matplotlib.pyplot as plt
import numpy as np

# Thiết lập style học thuật chuẩn mực
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#e0e0e0'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.7

def generate_slide11_chart(output_path="paper/figures/fig_slide11_core_benchmark.png"):
    fig, ax = plt.subplots(figsize=(13.5, 6.2), dpi=300)

    models = [
        '2B E0\n(Zero-shot)',
        '2B E1\n(Đơn ngữ EN)',
        '2B E2\n(Đơn ngữ VI)',
        '2B E3\n(Song ngữ)',
        '2B E4\n(+Miền VI)',
        '4B E3\n(Song ngữ)',
        '4B E4\n(+Miền VI)',
        'Method 2\n(Shared E4)'
    ]

    vi_tool_acc = [56.34, 93.67, 90.25, 94.00, 93.92, 98.73, 86.22, 59.20]
    vi_arga = [40.48, 65.57, 64.90, 69.76, 69.75, 72.86, 64.94, 30.26]
    en_arga = [60.63, 73.66, 71.36, 73.22, 73.15, 74.71, 66.66, 35.52]

    x = np.arange(len(models))
    width = 0.26

    c_vi_tool = '#9ecae1'  # Xanh lam nhạt
    c_vi_arga = '#1f77b4'  # Xanh lam đậm (Key VI Metric)
    c_en_arga = '#e6550d'  # Cam đậm (Key EN Metric)

    rects1 = ax.bar(x - width, vi_tool_acc, width, label='VI Test: Tool Acc (%)',
                    color=c_vi_tool, edgecolor='#2b5c8f', lw=0.7)
    rects2 = ax.bar(x, vi_arga, width, label='VI Test: ArgA (%)',
                    color=c_vi_arga, edgecolor='#08306b', lw=0.8)
    rects3 = ax.bar(x + width, en_arga, width, label='EN Test: ArgA (%)',
                    color=c_en_arga, edgecolor='#7f2704', lw=0.8)

    ax.set_ylabel('Độ chính xác / Accuracy (%)', fontsize=12, fontweight='bold', labelpad=8)
    ax.set_title('Hiệu năng trên Canonical Core Benchmark (7.712 mẫu song ngữ Test)',
                 fontsize=14, fontweight='bold', pad=18)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10.5, fontweight='bold')
    ax.set_ylim(0, 118)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Đánh số trên đầu cột VI ArgA và EN ArgA
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}%'.replace('.', ','),
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#08306b')

    for rect in rects3:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}%'.replace('.', ','),
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#7f2704')

    # Đánh dấu đỉnh cao nhất ở 4B E3
    ax.annotate('Đỉnh hiệu năng:\nVI: 72,86% | EN: 74,71%',
                xy=(5, 75.0), xytext=(5, 96),
                arrowprops=dict(arrowstyle="->", color="#08519c", lw=1.5),
                ha='center', fontsize=9.5, fontweight='bold', color='#08519c',
                bbox=dict(boxstyle="round,pad=0.3", fc="#eff3ff", ec="#08519c", lw=1.0))

    # Đánh dấu tụt ở 4B E4 do can nhiễu miền
    ax.annotate('Can nhiễu miền Core:\nLỗi cú pháp 9,75%',
                xy=(6, 67.0), xytext=(6, 88),
                arrowprops=dict(arrowstyle="->", color="#b30000", lw=1.5),
                ha='center', fontsize=9.5, fontweight='bold', color='#b30000',
                bbox=dict(boxstyle="round,pad=0.3", fc="#fee5d9", ec="#b30000", lw=1.0))

    # Ghi chú Method 2: Trade-off độ trễ
    ax.annotate('Ưu thế độ trễ:\n61,50 ms (Gấp ~40×)',
                xy=(7, 36.0), xytext=(7, 56),
                arrowprops=dict(arrowstyle="->", color="#006d2c", lw=1.5),
                ha='center', fontsize=9.5, fontweight='bold', color='#006d2c',
                bbox=dict(boxstyle="round,pad=0.3", fc="#edf8e9", ec="#006d2c", lw=1.0))

    ax.legend(loc='upper left', fontsize=10.5, framealpha=0.95, edgecolor='#cccccc')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved chart to {output_path}")

if __name__ == "__main__":
    generate_slide11_chart()
