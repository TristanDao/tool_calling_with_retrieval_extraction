"""
Script tao cac bieu do va so do hoc thuat (Academic Figures) cho bai bao khoa hoc
ve Tool Calling tieng Viet.
Xuat cac file PDF (vector) va PNG (300 DPI) vao thu muc paper/figures/.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Thiet lap style hoc thuat
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#dddddd'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.7

OUTPUT_DIR = "paper/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_fig1_architecture():
    """Figure 1: So do kien truc tong quan (System Architecture Diagram)."""
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)
    ax.axis('off')
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.5)

    # Khung tieu de Input
    p_in = patches.FancyBboxPatch(
        (0.5, 4.9), 11.0, 1.2,
        boxstyle="round,pad=0.2,rounding_size=0.15",
        ec="#1f497d", fc="#e9f1f7", lw=1.5
    )
    ax.add_patch(p_in)
    ax.text(6.0, 5.8, "NGƯỜI DÙNG / TRUY VẤN TIẾNG VIỆT (USER QUERY)", 
            ha='center', va='center', fontsize=12, fontweight='bold', color='#1f497d')
    ax.text(6.0, 5.3, '"Kiểm tra phạt nguội cho xe máy biển số 59P1-12345 tại TP. Hồ Chí Minh"', 
            ha='center', va='center', fontsize=11, fontstyle='italic', color='#333333')
    ax.text(6.0, 4.85, 'Danh mục công cụ: [tra_cuu_phat_nguoi, dat_ve_xe_khach, thanh_toan_tien_dien, ...]', 
            ha='center', va='center', fontsize=9.5, color='#555555')

    # Mui ten phan nhanh
    ax.annotate('', xy=(3.2, 4.3), xytext=(4.5, 4.7),
                arrowprops=dict(arrowstyle="->", lw=2, color="#2c5e8a", shrinkA=5, shrinkB=5))
    ax.annotate('', xy=(8.8, 4.3), xytext=(7.5, 4.7),
                arrowprops=dict(arrowstyle="->", lw=2, color="#b25e2e", shrinkA=5, shrinkB=5))

    # NHANH 1: METHOD 1 (SLM END-TO-END)
    p_m1 = patches.FancyBboxPatch(
        (0.5, 0.4), 5.2, 3.8,
        boxstyle="round,pad=0.2,rounding_size=0.2",
        ec="#b25e2e", fc="#fdf6f0", lw=1.8
    )
    ax.add_patch(p_m1)
    ax.text(3.1, 3.9, "PHƯƠNG PHÁP 1: SLM END-TO-END (Qwen3.5 2B/4B)", 
            ha='center', va='center', fontsize=11, fontweight='bold', color='#b25e2e')

    steps_m1 = [
        ("Định dạng Prompt & JSON Schema", "#f5e1d3"),
        ("Qwen3.5 + QLoRA SFT\n(Học loss chỉ trên Assistant response)", "#e8c2a8"),
        ("Giải mã tự hồi quy (Autoregressive)\nSinh cấu trúc XML native: <tool_call>", "#dc9f7c"),
        ("JSON Tool Call Output\nArguments trích xuất trực tiếp", "#b25e2e", "white")
    ]
    y_pos = 3.3
    for s in steps_m1:
        text, col = s[0], s[1]
        text_col = s[2] if len(s) > 2 else '#222222'
        box = patches.FancyBboxPatch((0.8, y_pos - 0.4), 4.6, 0.55,
                                     boxstyle="round,pad=0.1,rounding_size=0.1",
                                     ec="#b25e2e", fc=col, lw=1.0)
        ax.add_patch(box)
        ax.text(3.1, y_pos - 0.12, text, ha='center', va='center', fontsize=9, color=text_col, fontweight='bold' if text_col=='white' else 'normal')
        if y_pos > 1.2:
            ax.annotate('', xy=(3.1, y_pos - 0.45), xytext=(3.1, y_pos - 0.35),
                        arrowprops=dict(arrowstyle="->", lw=1.2, color="#b25e2e"))
        y_pos -= 0.75

    # NHANH 2: METHOD 2 (BI-ENCODER + CROSS-ENCODER)
    p_m2 = patches.FancyBboxPatch(
        (6.3, 0.4), 5.2, 3.8,
        boxstyle="round,pad=0.2,rounding_size=0.2",
        ec="#1f497d", fc="#f0f5fa", lw=1.8
    )
    ax.add_patch(p_m2)
    ax.text(8.9, 3.9, "PHƯƠNG PHÁP 2: BI-ENCODER + CROSS-ENCODER", 
            ha='center', va='center', fontsize=11, fontweight='bold', color='#1f497d')

    steps_m2 = [
        ("Bi-Encoder BGE-M3 (CachedMNRL)\nTruy hồi Top-K công cụ ứng viên (k ≤ 3)", "#d4e4f2"),
        ("Đầu quyết định No-call / Selection\nNgưỡng hiệu chuẩn τ = 0.35, δ = 0.21", "#b8d2e8"),
        ("Cross-Encoder XLM-R (Hierarchical Heads)\n• has_value • Span head • Enum/Bool head", "#9bbddc"),
        ("Value Normalizer + JSON Validator\nÁnh xạ thực thể Việt & Chuẩn hóa JSON", "#1f497d", "white")
    ]
    y_pos = 3.3
    for s in steps_m2:
        text, col = s[0], s[1]
        text_col = s[2] if len(s) > 2 else '#222222'
        box = patches.FancyBboxPatch((6.6, y_pos - 0.4), 4.6, 0.55,
                                     boxstyle="round,pad=0.1,rounding_size=0.1",
                                     ec="#1f497d", fc=col, lw=1.0)
        ax.add_patch(box)
        ax.text(8.9, y_pos - 0.12, text, ha='center', va='center', fontsize=9, color=text_col, fontweight='bold' if text_col=='white' else 'normal')
        if y_pos > 1.2:
            ax.annotate('', xy=(8.9, y_pos - 0.45), xytext=(8.9, y_pos - 0.35),
                        arrowprops=dict(arrowstyle="->", lw=1.2, color="#1f497d"))
        y_pos -= 0.75

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig1_system_architecture.png"), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(OUTPUT_DIR, "fig1_system_architecture.pdf"), bbox_inches='tight')
    plt.close()
    print("✓ Da xuat fig1_system_architecture (PNG & PDF)")


def plot_fig2_customtools_comparison():
    """Figure 2: So sanh hieu nang tren CustomTools-VI (Seen vs Unseen)."""
    fig, ax = plt.subplots(figsize=(10.5, 6.0), dpi=300)

    models = [
        'Method 2\n(Bi+Cross)',
        'SLM 2B (E4)\n(Qwen3.5)',
        'SLM 4B (E4)\n(Qwen3.5)',
        'GPT-5.6 Luna\n(OpenAI API)',
        'Gemini 3.8 Flash\n(Google API)'
    ]
    
    seen_arga = [67.75, 87.00, 87.92, 78.25, 93.62]
    unseen_arga = [22.75, 86.38, 86.75, 79.50, 92.12]
    seen_acc = [85.25, 93.25, 94.66, 93.25, 100.00]
    unseen_acc = [74.50, 97.00, 96.75, 97.25, 100.00]

    x = np.arange(len(models))
    width = 0.2

    c_seen_acc = '#aec7e8'
    c_seen_arga = '#1f77b4'
    c_unseen_acc = '#ffbb78'
    c_unseen_arga = '#d62728'

    rects1 = ax.bar(x - 1.5*width, seen_acc, width, label='Seen Tool Acc (%)', color=c_seen_acc, edgecolor='black', lw=0.6)
    rects2 = ax.bar(x - 0.5*width, seen_arga, width, label='Seen ArgA / EM (%)', color=c_seen_arga, edgecolor='black', lw=0.6)
    rects3 = ax.bar(x + 0.5*width, unseen_acc, width, label='Unseen Tool Acc (%)', color=c_unseen_acc, edgecolor='black', lw=0.6)
    rects4 = ax.bar(x + 1.5*width, unseen_arga, width, label='Unseen ArgA / EM (%)', color=c_unseen_arga, edgecolor='black', lw=0.6)

    ax.set_ylabel('Độ chính xác / Accuracy (%)', fontsize=11, fontweight='bold')
    ax.set_title('So sánh đối đầu toàn diện trên benchmark CustomTools-VI (Seen vs. Unseen)', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10, fontweight='bold')
    ax.set_ylim(0, 130)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Them gia tri len dinh cac cot ArgA
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold', color='#1f77b4')

    for rect in rects4:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold', color='#d62728')

    # Chu thich Zero-Shot Gap cho Method 2 va SLM
    ax.annotate('Sụp đổ Zero-Shot:\n-45.0% ArgA', xy=(0 + 1.5*width, 24), xytext=(-0.1, 46),
                arrowprops=dict(arrowstyle="->", color="black", lw=1),
                fontsize=8.5, fontweight='bold', color='#b30000', ha='left',
                bbox=dict(boxstyle="round,pad=0.2", fc="#ffe6e6", ec="#b30000", lw=0.8))

    ax.annotate('Zero-Shot bền bỉ:\nArgA Gap +1.17%', xy=(2 + 1.5*width, 88), xytext=(2.0, 110),
                arrowprops=dict(arrowstyle="->", color="darkgreen", lw=1.2),
                fontsize=8.5, fontweight='bold', color='darkgreen', ha='center',
                bbox=dict(boxstyle="round,pad=0.2", fc="#e6ffe6", ec="darkgreen", lw=0.8))

    ax.legend(loc='upper left', ncol=2, fontsize=8.5, framealpha=0.95)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig2_performance_comparison.png"), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(OUTPUT_DIR, "fig2_performance_comparison.pdf"), bbox_inches='tight')
    plt.close()
    print("✓ Da xuat fig2_performance_comparison (PNG & PDF)")


def plot_fig3_stress_test():
    """Figure 3: Ket qua Stress Test doi dau giua SLM va Method 2 (N = 3 -> 1000)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=300)

    N_values = [3, 10, 50, 100, 500, 1000]
    x_indices = np.arange(len(N_values))

    # Data
    m2_tool_acc = [100.0, 98.0, 96.0, 95.0, 78.0, 68.0]
    m2_arga = [81.0, 79.0, 78.0, 77.0, 62.0, 54.0]
    m2_latency = [58.16, 57.15, 55.82, 58.55, 87.44, 91.88]

    slm_tool_acc = [94.0, 93.0, 93.0, 92.0, np.nan, np.nan]
    slm_arga = [85.5, 84.5, 85.5, 83.0, np.nan, np.nan]
    slm_latency = [1258.94, 1604.68, 4703.39, 9318.21, np.nan, np.nan]

    # PANEL A: DO CHINH XAC (ArgA & Tool Acc)
    ax1.plot(x_indices, m2_tool_acc, marker='o', color='#1f77b4', lw=2, label='Method 2: Tool Selection Acc (%)')
    ax1.plot(x_indices, m2_arga, marker='s', color='#1f77b4', lw=2, linestyle='--', label='Method 2: ArgA / Exact Match (%)')

    ax1.plot(x_indices[:4], slm_tool_acc[:4], marker='^', color='#d62728', lw=2, label='SLM (2B E4): Tool Selection Acc (%)')
    ax1.plot(x_indices[:4], slm_arga[:4], marker='d', color='#d62728', lw=2, linestyle='--', label='SLM (2B E4): ArgA / Exact Match (%)')

    # Vung OOM cho SLM
    ax1.axvspan(3.5, 5.5, color='#feebe8', alpha=0.6)
    ax1.text(4.5, 75, "SLM Sụp Đổ\n(CUDA OOM)\n>32GB VRAM", ha='center', va='center', 
             fontsize=10, fontweight='bold', color='#990000',
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#990000", lw=1))

    ax1.set_xlabel('Số lượng công cụ trong prompt ($N$)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Độ chính xác / Accuracy (%)', fontsize=11, fontweight='bold')
    ax1.set_title('(a) Độ chính xác chọn công cụ & trích xuất tham số', fontsize=11, fontweight='bold')
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels([f'$N={n}$' for n in N_values], fontsize=10, fontweight='bold')
    ax1.set_ylim(45, 105)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='lower left', fontsize=8.5)

    # PANEL B: DO TRE SUY LUAN (P50 Latency)
    ax2.plot(x_indices, m2_latency, marker='o', color='#1f77b4', lw=2.2, label='Method 2 (Bi+Cross Pipeline)')
    ax2.plot(x_indices[:4], slm_latency[:4], marker='^', color='#d62728', lw=2.2, label='SLM (Qwen3.5-2B E4)')

    ax2.set_yscale('log')
    ax2.axvspan(3.5, 5.5, color='#feebe8', alpha=0.6)
    ax2.text(4.5, 500, "Vùng OOM\nKhông thể phục vụ", ha='center', va='center', 
             fontsize=10, fontweight='bold', color='#990000')

    # Mui ten toc do tai N=100
    ax2.annotate('Nhanh gấp 159.2 lần\n(58.6ms vs 9.3s)', xy=(3, 60), xytext=(2.2, 500),
                 arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.2),
                 fontsize=9, fontweight='bold', color='#1f77b4',
                 bbox=dict(boxstyle="round,pad=0.2", fc="#e6f2ff", ec="#1f77b4", lw=0.8))

    ax2.set_xlabel('Số lượng công cụ trong prompt ($N$)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Độ trễ P50 (ms) - Thang đo Log', fontsize=11, fontweight='bold')
    ax2.set_title('(b) Độ trễ suy luận P50 (ms) theo quy mô công cụ', fontsize=11, fontweight='bold')
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels([f'$N={n}$' for n in N_values], fontsize=10, fontweight='bold')
    ax2.grid(True, which="both", ls="--", alpha=0.4)
    ax2.legend(loc='upper left', fontsize=9)

    plt.suptitle("Stress Test đối đầu trực diện: SLM vs. Method 2 ($N=3 \to 1.000$ công cụ)", 
                 fontsize=13, fontweight='bold', y=1.00)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig3_stress_test_curves.png"), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(OUTPUT_DIR, "fig3_stress_test_curves.pdf"), bbox_inches='tight')
    plt.close()
    print("✓ Da xuat fig3_stress_test_curves (PNG & PDF)")


def plot_fig4_scaling_syntax():
    """Figure 4: Nghien cuu mo rong quy mo (2B vs 4B) va triet tieu loi cu phap."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=300)

    # PANEL A: ACCURACY CEILING ON CORE BENCHMARK
    categories = ['Core VI\nTool Acc', 'Core VI\nArgA', 'Core EN\nTool Acc', 'Core EN\nArgA']
    m2b = [94.00, 69.76, 94.27, 73.22]
    m4b = [98.73, 72.86, 96.63, 74.71]

    x = np.arange(len(categories))
    width = 0.32

    ax1.bar(x - width/2, m2b, width, label='Qwen3.5-2B (E3)', color='#9ecae1', edgecolor='black', lw=0.7)
    bars2 = ax1.bar(x + width/2, m4b, width, label='Qwen3.5-4B (E3)', color='#2171b5', edgecolor='black', lw=0.7)

    for bar in bars2:
        h = bar.get_height()
        ax1.annotate(f'{h:.2f}%', xy=(bar.get_x() + bar.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    ax1.set_ylabel('Độ chính xác / Accuracy (%)', fontsize=11, fontweight='bold')
    ax1.set_title('(a) Nâng trần độ chính xác trên Core Benchmark', fontsize=11, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=9.5, fontweight='bold')
    ax1.set_ylim(55, 126)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    ax1.legend(loc='upper left', fontsize=9, framealpha=0.95)

    # PANEL B: SYNTAX ERROR SUPPRESSION
    err_cats = ['Core VI Test', 'Core EN Test']
    err_2b = [4.31, 4.35]
    err_4b = [0.32, 1.97]

    x2 = np.arange(len(err_cats))
    ax2.bar(x2 - width/2, err_2b, width, label='Qwen3.5-2B (E3)', color='#fc9272', edgecolor='black', lw=0.7)
    bars_err = ax2.bar(x2 + width/2, err_4b, width, label='Qwen3.5-4B (E3)', color='#cb181d', edgecolor='black', lw=0.7)

    for bar in bars_err:
        h = bar.get_height()
        ax2.annotate(f'{h:.2f}%', xy=(bar.get_x() + bar.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#cb181d')

    ax2.annotate('Giảm hơn 13 lần:\n4.31% → 0.32%', xy=(x2[0] + width/2, 0.35), xytext=(0.55, 4.2),
                 arrowprops=dict(arrowstyle="->", color="#990000", lw=1.2),
                 fontsize=9, fontweight='bold', color='#990000',
                 bbox=dict(boxstyle="round,pad=0.2", fc="#ffe6e6", ec="#990000", lw=0.8))

    ax2.set_ylabel('Tỷ lệ lỗi cú pháp / Syntax Error Rate (%)', fontsize=11, fontweight='bold')
    ax2.set_title('(b) Triệt tiêu lỗi cú pháp JSON (Syntax Error)', fontsize=11, fontweight='bold')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(err_cats, fontsize=9.5, fontweight='bold')
    ax2.set_ylim(0, 7.0)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    ax2.legend(loc='upper left', fontsize=9, framealpha=0.95)

    plt.suptitle("Nghiên cứu mở rộng quy mô (Model Scaling Study: 2B vs. 4B)", fontsize=13, fontweight='bold', y=1.00)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig4_model_scaling.png"), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(OUTPUT_DIR, "fig4_model_scaling.pdf"), bbox_inches='tight')
    plt.close()
    print("✓ Da xuat fig4_model_scaling (PNG & PDF)")


if __name__ == "__main__":
    print("==================================================")
    print("BẮT ĐẦU XUẤT CÁC HÌNH ẢNH HỌC THUẬT CHO BÀI BÁO")
    print("==================================================")
    plot_fig1_architecture()
    plot_fig2_customtools_comparison()
    plot_fig3_stress_test()
    plot_fig4_scaling_syntax()
    print("==================================================")
    print(f"HOÀN TẤT 100%! CÁC FILE ĐƯỢC LƯU TẠI: {OUTPUT_DIR}/")
    print("==================================================")
