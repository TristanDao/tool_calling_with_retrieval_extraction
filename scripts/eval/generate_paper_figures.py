"""
Script tạo các biểu đồ và sơ đồ học thuật (Academic Figures) cho bài báo khoa học
về Tool Calling tiếng Việt.
Hỗ trợ cả 2 ngôn ngữ: Tiếng Việt (vi) và Tiếng Anh (en).
Xuất các file SVG, PDF (vector) và PNG (300 DPI) vào các thư mục hình của bài báo.
"""

import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Thiết lập style học thuật chuẩn mực
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#dddddd'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.7


def plot_fig1_architecture(lang: str = "vi", output_dir: str = "paper/figures") -> None:
    is_en = lang == "en"
    fig, ax = plt.subplots(figsize=(14, 8.2), dpi=300)
    ax.axis("off")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8.2)

    colors = {
        "ink": "#203247",
        "muted": "#536579",
        "line": "#718096",
        "panel": "#f8fafc",
        "panel_edge": "#cbd5e1",
        "input": "#e8f1f8",
        "model": "#d9e7f2",
        "retrieval": "#e4edf6",
        "head": "#e8e4f4",
        "post": "#e8f1e8",
        "accent1": "#a64b24",
        "accent2": "#245b83",
        "white": "#ffffff",
    }

    def box(x: float, y: float, width: float, height: float, label: str, fill: str, edge: str, size: float = 9.0, weight: str = "normal") -> None:
        shape = patches.FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.04,rounding_size=0.09",
            ec=edge, fc=fill, lw=1.15,
        )
        ax.add_patch(shape)
        ax.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=size, color=colors["ink"], fontweight=weight, linespacing=1.2)

    def arrow(start: tuple[float, float], end: tuple[float, float], color: str = "line", connection: str = "arc3,rad=0") -> None:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "-|>", "lw": 1.2, "color": colors.get(color, color), "shrinkA": 2, "shrinkB": 2, "connectionstyle": connection})

    def panel(x: float, y: float, width: float, height: float, title: str, accent: str) -> None:
        shape = patches.FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.07,rounding_size=0.12",
            ec=colors["panel_edge"], fc=colors["panel"], lw=1.0,
        )
        ax.add_patch(shape)
        ax.text(x + 0.22, y + height - 0.28, title, ha="left", va="center", fontsize=11.2, color=accent, fontweight="bold")
        ax.plot([x + 0.2, x + width - 0.2], [y + height - 0.54, y + height - 0.54], color=colors["panel_edge"], lw=0.8)

    title_m1 = "(a) End-to-end SLM" if is_en else "(a) SLM đầu-cuối"
    title_m2 = "(b) Retrieval + schema-aware extraction" if is_en else "(b) Truy hồi + trích xuất theo schema"
    panel(0.25, 0.3, 4.15, 7.55, title_m1, colors["accent1"])
    panel(4.6, 0.3, 9.15, 7.55, title_m2, colors["accent2"])

    q_label = "User request\n‘Check traffic fines for plate 59P1-12345’" if is_en else "Yêu cầu người dùng\n‘Kiểm tra phạt nguội xe 59P1-12345’"
    schema_label = "Tool schemas\nprovided in prompt" if is_en else "Tool Schema\nđưa vào prompt"
    model_label = "Qwen3.5 2B / 4B\nQLoRA fine-tuning\n(response-only loss)" if is_en else "Qwen3.5 2B / 4B\nTinh chỉnh QLoRA\n(response-only loss)"
    generation_label = "Assistant generation\nautoregressive decoding" if is_en else "Sinh phản hồi assistant\ngiải mã tự hồi quy"
    response_label = "Tool-call tags" if is_en else "Thẻ gọi công cụ"
    natural_label = "Natural-language reply\n(no tool call)" if is_en else "Phản hồi tự nhiên\n(không gọi công cụ)"
    parser_label = "Rule-based parser" if is_en else "Bộ phân tích thẻ"
    call_label = "Structured tool call" if is_en else "Lời gọi có cấu trúc"

    box(0.55, 6.27, 2.23, 0.68, q_label, colors["input"], colors["panel_edge"], 8.5)
    box(2.94, 6.27, 1.18, 0.68, schema_label, colors["input"], colors["panel_edge"], 8.2)
    box(0.9, 4.96, 2.95, 0.88, model_label, colors["model"], colors["accent1"], 9.2, "bold")
    box(0.9, 3.65, 2.95, 0.78, generation_label, colors["white"], colors["panel_edge"], 9.0)
    box(0.62, 2.13, 1.72, 0.82, response_label, "#f7e8df", colors["accent1"], 8.8)
    box(2.48, 2.13, 1.64, 0.82, natural_label, colors["white"], colors["panel_edge"], 8.3)
    box(0.62, 0.82, 1.72, 0.68, parser_label, "#f7e8df", colors["accent1"], 8.8)
    box(2.48, 0.82, 1.64, 0.68, call_label, "#f7e8df", colors["accent1"], 8.8, "bold")
    arrow((1.65, 6.25), (1.65, 5.88), "accent1")
    arrow((3.52, 6.25), (3.12, 5.88), "accent1", "arc3,rad=0.12")
    arrow((2.37, 4.94), (2.37, 4.45), "accent1")
    arrow((2.37, 3.63), (1.48, 2.98), "accent1", "arc3,rad=0.08")
    arrow((2.37, 3.63), (3.28, 2.98), "line", "arc3,rad=-0.08")
    arrow((1.48, 2.11), (1.48, 1.52), "accent1")
    arrow((2.35, 1.16), (2.47, 1.16), "accent1")

    tool_store = "Tool catalog\n(descriptions)" if is_en else "Kho công cụ\n(mô tả)"
    q_encode = "Query\nencoder" if is_en else "Mã hóa\ntruy vấn"
    tool_encode = "Tool\nencoder" if is_en else "Mã hóa\ncông cụ"
    index_label = "Precomputed tool embeddings" if is_en else "Vector công cụ đã mã hóa"
    similarity = "Cosine ranking\n+ calibrated gate" if is_en else "Xếp hạng cosine\n+ ngưỡng kích hoạt"
    candidates = "Top-K candidates\n(k ≤ 3)" if is_en else "Công cụ ứng viên\n(Top-K, k ≤ 3)"
    query_context = "Same user query" if is_en else "Cùng truy vấn người dùng"
    box(4.9, 6.73, 1.48, 0.68, query_context, colors["input"], colors["panel_edge"], 8.5)
    box(4.9, 5.83, 1.48, 0.65, q_encode, colors["retrieval"], colors["accent2"], 8.5)
    box(6.65, 6.25, 1.75, 0.82, tool_store, colors["input"], colors["panel_edge"], 8.4)
    box(8.62, 6.25, 1.85, 0.82, tool_encode, colors["retrieval"], colors["accent2"], 8.5)
    box(10.69, 6.25, 1.62, 0.82, index_label, colors["retrieval"], colors["accent2"], 8.0)
    box(7.08, 5.02, 2.45, 0.73, similarity, "#d5e3ef", colors["accent2"], 8.7, "bold")
    box(5.03, 5.02, 1.8, 0.73, candidates, colors["retrieval"], colors["accent2"], 8.4)
    box(10.12, 5.02, 2.25, 0.73, "No candidate above threshold\n→ no-call []" if is_en else "Không có công cụ đạt ngưỡng\n→ no-call []", colors["white"], colors["panel_edge"], 7.9)
    arrow((5.64, 6.71), (5.64, 6.51), "accent2")
    arrow((8.42, 6.65), (8.62, 6.65), "accent2")
    arrow((10.47, 6.65), (10.69, 6.65), "accent2")
    arrow((6.38, 6.15), (7.08, 5.75), "accent2", "arc3,rad=0.05")
    arrow((11.48, 6.22), (9.53, 5.75), "accent2", "arc3,rad=-0.04")
    arrow((7.08, 5.38), (6.83, 5.38), "accent2")
    arrow((9.53, 5.38), (10.12, 5.38), "line")

    extraction_title = "Per-parameter cross-encoding and hierarchical prediction" if is_en else "Cross-Encoder và dự đoán phân cấp theo từng tham số"
    pair_label = "XLM-R\nQuery + parameter\nprompt / schema" if is_en else "XLM-R\nTruy vấn + mô tả\ntham số trong schema"
    presence_label = "has_value\npresent?" if is_en else "has_value\ncó giá trị?"
    routing_label = "Route by\nschema type" if is_en else "Chọn nhánh theo\nkiểu schema"
    span_label = "Span\nhead" if is_en else "Span\nhead"
    enum_label = "Enum\nhead" if is_en else "Enum\nhead"
    bool_label = "Boolean\nhead" if is_en else "Boolean\nhead"
    normalizer_label = "Value normalizer" if is_en else "Chuẩn hóa giá trị"
    assemble_label = "Schema-based\nassembly" if is_en else "Lắp ghép theo\nschema"
    json_label = "Structured JSON\nfunction call" if is_en else "Lời gọi hàm\nJSON"
    subpanel = patches.FancyBboxPatch((4.9, 1.8), 8.58, 2.7, boxstyle="round,pad=0.04,rounding_size=0.08", ec=colors["panel_edge"], fc=colors["white"], lw=0.9)
    ax.add_patch(subpanel)
    ax.text(6.95, 4.23, extraction_title, ha="left", va="center", fontsize=9.1, color=colors["accent2"], fontweight="bold")
    box(5.12, 2.88, 1.45, 0.82, pair_label, colors["input"], colors["panel_edge"], 8.0)
    box(6.8, 2.88, 1.3, 0.82, presence_label, colors["head"], colors["accent2"], 8.4, "bold")
    box(8.34, 2.88, 1.35, 0.82, routing_label, colors["model"], colors["accent2"], 8.0)
    box(10.02, 3.13, 0.94, 0.62, span_label, colors["head"], colors["accent2"], 8.2)
    box(11.1, 3.13, 0.94, 0.62, enum_label, colors["head"], colors["accent2"], 8.2)
    box(12.18, 3.13, 1.0, 0.62, bool_label, colors["head"], colors["accent2"], 8.0)
    box(9.65, 2.02, 1.5, 0.62, normalizer_label, colors["post"], "#6d8965", 8.2)
    box(11.42, 2.02, 1.0, 0.62, assemble_label, colors["post"], "#6d8965", 7.8)
    box(12.6, 2.02, 0.7, 0.62, json_label, colors["post"], "#6d8965", 7.3, "bold")
    arrow((11.17, 2.32), (11.42, 2.32), "line")
    arrow((12.42, 2.32), (12.6, 2.32), "line")
    arrow((5.93, 5.0), (5.93, 4.53), "accent2")
    arrow((5.93, 4.5), (5.93, 3.72), "accent2")
    ax.plot([5.0, 4.75], [6.73, 6.5], color=colors["accent2"], lw=1.2)
    ax.plot([4.75, 4.75], [6.5, 3.29], color=colors["accent2"], lw=1.2)
    arrow((4.75, 3.29), (5.12, 3.29), "accent2")
    arrow((6.57, 3.29), (6.8, 3.29), "accent2")
    arrow((8.1, 3.29), (8.34, 3.29), "accent2")
    arrow((9.69, 3.29), (9.87, 3.29), "accent2")
    ax.plot([9.87, 9.87], [3.29, 3.91], color=colors["accent2"], lw=1.2)
    ax.plot([9.87, 12.68], [3.91, 3.91], color=colors["accent2"], lw=1.2)
    for head_x in (10.49, 11.57, 12.68):
        arrow((head_x, 3.91), (head_x, 3.77), "accent2")
    ax.plot([10.12, 12.68], [2.88, 2.88], color=colors["line"], lw=1.0)
    for head_x in (10.49, 11.57, 12.68):
        arrow((head_x, 3.12), (head_x, 2.9), "accent2")
    arrow((11.38, 2.88), (11.38, 2.66), "accent2")
    ax.text(7.45, 2.56, "if present" if is_en else "nếu có", ha="center", va="center", fontsize=7.5, color=colors["muted"])
    ax.text(12.8, 4.78, "τ = 0.35 · δ = 0.21" if is_en else "τ = 0.35 · δ = 0.21", ha="center", va="center", fontsize=7.6, color=colors["muted"])

    plt.tight_layout(pad=0.2)
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "fig1_system_architecture.png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(output_dir, "fig1_system_architecture.pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(output_dir, "fig1_system_architecture.svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_fig2_customtools_comparison(lang="vi", output_dir="paper/figures"):
    """Figure 2: So sánh hiệu năng trên CustomTools-VI (Seen vs Unseen)."""
    fig, ax = plt.subplots(figsize=(10.5, 6.0), dpi=300)
    is_en = (lang == "en")

    models = [
        'Method 2\n(Bi+Cross)',
        'SLM 2B (E4)\n(Qwen3.5)',
        'SLM 4B (E4)\n(Qwen3.5)',
        'GPT-5.6 Luna\n(OpenAI API)',
        'Gemini 3.8 Flash\n(Google API)'
    ]
    
    seen_acc = [92.25, 93.25, 92.50, 93.25, 100.00]
    seen_arga = [85.38, 87.00, 86.62, 78.25, 93.62]
    unseen_acc = [79.50, 97.00, 96.75, 97.25, 100.00]
    unseen_arga = [60.25, 86.38, 86.75, 79.50, 92.12]

    x = np.arange(len(models))
    width = 0.2

    c_seen_acc = '#aec7e8'
    c_seen_arga = '#1f77b4'
    c_unseen_acc = '#ffbb78'
    c_unseen_arga = '#d62728'

    rects1 = ax.bar(x - 1.5*width, seen_acc, width, label='Seen Tool Acc (%)', color=c_seen_acc, edgecolor='black', lw=0.6)
    rects2 = ax.bar(x - 0.5*width, seen_arga, width, label='Seen ArgA-all (%)', color=c_seen_arga, edgecolor='black', lw=0.6)
    rects3 = ax.bar(x + 0.5*width, unseen_acc, width, label='Unseen Tool Acc (%)', color=c_unseen_acc, edgecolor='black', lw=0.6)
    rects4 = ax.bar(x + 1.5*width, unseen_arga, width, label='Unseen ArgA-all (%)', color=c_unseen_arga, edgecolor='black', lw=0.6)

    ylabel = 'Accuracy (%)' if is_en else 'Độ chính xác / Accuracy (%)'
    title = 'Comprehensive Empirical Comparison on CustomTools-VI Benchmark (Seen vs. Unseen)' if is_en else 'So sánh đối đầu toàn diện trên benchmark CustomTools-VI (Seen vs. Unseen)'
    
    ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10, fontweight='bold')
    ax.set_ylim(0, 130)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Thêm giá trị lên đỉnh các cột ArgA
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold', color='#1f77b4')

    for rect in rects4:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold', color='#d62728')

    # Chú thích Zero-Shot Gap cho Method 2 và SLM
    txt_drop = 'Zero-Shot Drop:\n-25.13% ArgA' if is_en else 'Suy giảm Zero-Shot:\n-25.13% ArgA'
    txt_robust = 'Robust Zero-Shot:\nArgA Gap +0.13%' if is_en else 'Zero-Shot bền bỉ:\nArgA Gap +0.13%'

    ax.annotate(txt_drop, xy=(0 + 1.5*width, 60.25), xytext=(-0.15, 78),
                arrowprops=dict(arrowstyle="->", color="black", lw=1),
                fontsize=8.5, fontweight='bold', color='#b30000', ha='left',
                bbox=dict(boxstyle="round,pad=0.2", fc="#ffe6e6", ec="#b30000", lw=0.8))

    ax.annotate(txt_robust, xy=(2 + 1.5*width, 88), xytext=(2.0, 110),
                arrowprops=dict(arrowstyle="->", color="darkgreen", lw=1.2),
                fontsize=8.5, fontweight='bold', color='darkgreen', ha='center',
                bbox=dict(boxstyle="round,pad=0.2", fc="#e6ffe6", ec="darkgreen", lw=0.8))

    ax.legend(loc='upper left', ncol=2, fontsize=8.5, framealpha=0.95)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "fig2_performance_comparison.png"), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, "fig2_performance_comparison.pdf"), bbox_inches='tight')
    plt.close()
    print(f"✓ [{lang.upper()}] Exported fig2_performance_comparison to {output_dir}")


def plot_fig3_stress_test(lang="vi", output_dir="paper/figures"):
    """Figure 3: Kết quả Stress Test đối đầu giữa SLM và Method 2 (N = 3 -> 1000)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=300)
    is_en = (lang == "en")

    N_values = [3, 10, 50, 100, 500, 1000]
    x_indices = np.arange(len(N_values))

    # Data
    m2_tool_acc = [89.00, 89.00, 89.00, 89.00, 87.00, 87.00]
    m2_arga = [87.50, 87.50, 87.50, 87.00, 85.00, 84.00]
    m2_latency = [55.13, 55.05, 56.07, 61.01, 92.27, 107.68]

    slm_tool_acc = [94.0, 93.0, 93.0, 92.0, np.nan, np.nan]
    slm_arga = [85.5, 84.5, 85.5, 83.0, np.nan, np.nan]
    slm_latency = [1258.94, 1604.68, 4703.39, 9318.21, np.nan, np.nan]

    # PANEL A: ĐỘ CHÍNH XÁC (ArgA & Tool Acc)
    lbl_m2_tool = 'Method 2: Tool Selection Acc (%)'
    lbl_m2_arga = 'Method 2: ArgA / Exact Match (%)'
    lbl_slm_tool = 'SLM (2B E4): Tool Selection Acc (%)'
    lbl_slm_arga = 'SLM (2B E4): ArgA / Exact Match (%)'

    ax1.plot(x_indices, m2_tool_acc, marker='o', color='#1f77b4', lw=2, label=lbl_m2_tool)
    ax1.plot(x_indices, m2_arga, marker='s', color='#1f77b4', lw=2, linestyle='--', label=lbl_m2_arga)

    ax1.plot(x_indices[:4], slm_tool_acc[:4], marker='^', color='#d62728', lw=2, label=lbl_slm_tool)
    ax1.plot(x_indices[:4], slm_arga[:4], marker='d', color='#d62728', lw=2, linestyle='--', label=lbl_slm_arga)

    # Vùng OOM cho SLM
    ax1.axvspan(3.5, 5.5, color='#feebe8', alpha=0.6)
    txt_oom_box = "SLM Memory Limit\n(CUDA OOM)\nAt N ≥ 500 on 16GB T4" if is_en else "Giới hạn bộ nhớ SLM\n(CUDA OOM)\nTại N ≥ 500 trên T4 16GB"
    ax1.text(4.5, 75, txt_oom_box, ha='center', va='center', 
             fontsize=10, fontweight='bold', color='#990000',
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#990000", lw=1))

    xlabel_a = 'Number of tools in prompt ($N$)' if is_en else 'Số lượng công cụ trong prompt ($N$)'
    ylabel_a = 'Accuracy (%)' if is_en else 'Độ chính xác / Accuracy (%)'
    title_a = '(a) Tool Selection & Argument Accuracy' if is_en else '(a) Độ chính xác chọn công cụ & trích xuất tham số'

    ax1.set_xlabel(xlabel_a, fontsize=11, fontweight='bold')
    ax1.set_ylabel(ylabel_a, fontsize=11, fontweight='bold')
    ax1.set_title(title_a, fontsize=11, fontweight='bold')
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels([f'$N={n}$' for n in N_values], fontsize=10, fontweight='bold')
    ax1.set_ylim(45, 105)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='lower left', fontsize=8.5)

    # PANEL B: ĐỘ TRỄ SUY LUẬN (P50 Latency)
    lbl_m2_lat = 'Method 2 (Bi+Cross, P50)'
    lbl_slm_lat = 'SLM (Qwen3.5-2B E4, batch-normalized)'

    ax2.plot(x_indices, m2_latency, marker='o', color='#1f77b4', lw=2.2, label=lbl_m2_lat)
    ax2.plot(x_indices[:4], slm_latency[:4], marker='^', color='#d62728', lw=2.2, label=lbl_slm_lat)

    ax2.set_yscale('log')
    ax2.axvspan(3.5, 5.5, color='#feebe8', alpha=0.6)
    txt_oom_zone = "OOM Zone\nService Inoperable" if is_en else "Vùng OOM\nKhông thể phục vụ"
    ax2.text(4.5, 500, txt_oom_zone, ha='center', va='center', 
             fontsize=10, fontweight='bold', color='#990000')

    xlabel_b = 'Number of tools in prompt ($N$)' if is_en else 'Số lượng công cụ trong prompt ($N$)'
    ylabel_b = 'Latency (ms) - Log Scale' if is_en else 'Độ trễ (ms) - Thang đo Log'
    title_b = '(b) Inference Latency (Log Scale)' if is_en else '(b) Độ trễ suy luận (Thang đo Log)'

    ax2.set_xlabel(xlabel_b, fontsize=11, fontweight='bold')
    ax2.set_ylabel(ylabel_b, fontsize=11, fontweight='bold')
    ax2.set_title(title_b, fontsize=11, fontweight='bold')
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels([f'$N={n}$' for n in N_values], fontsize=10, fontweight='bold')
    ax2.grid(True, which="both", ls="--", alpha=0.4)
    ax2.legend(loc='upper left', fontsize=9)

    suptitle = "Direct Head-to-Head Stress Test: SLM vs. Method 2 (N = 3 → 1,000 tools)" if is_en else "Stress Test đối đầu trực diện: SLM vs. Method 2 (N = 3 → 1,000 công cụ)"
    plt.suptitle(suptitle, fontsize=13, fontweight='bold', y=1.00)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "fig3_stress_test_curves.png"), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, "fig3_stress_test_curves.pdf"), bbox_inches='tight')
    plt.close()
    print(f"✓ [{lang.upper()}] Exported fig3_stress_test_curves to {output_dir}")


def plot_fig4_scaling_syntax(lang="vi", output_dir="paper/figures"):
    """Figure 4: Nghiên cứu mở rộng quy mô (2B vs 4B) và triệt tiêu lỗi cú pháp."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=300)
    is_en = (lang == "en")

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

    ylabel_a = 'Accuracy (%)' if is_en else 'Độ chính xác / Accuracy (%)'
    title_a = '(a) Elevating Accuracy Ceiling on Core Benchmark' if is_en else '(a) Nâng trần độ chính xác trên Core Benchmark'

    ax1.set_ylabel(ylabel_a, fontsize=11, fontweight='bold')
    ax1.set_title(title_a, fontsize=11, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=9.5, fontweight='bold')
    ax1.set_ylim(55, 126)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    ax1.legend(loc='upper left', fontsize=9, framealpha=0.95)

    # PANEL B: SYNTAX ERROR SUPPRESSION
    err_cats = ['Core VI Test', 'Core EN Test']
    err_2b = [4.66, 4.73]
    err_4b = [0.32, 1.97]

    x2 = np.arange(len(err_cats))
    ax2.bar(x2 - width/2, err_2b, width, label='Qwen3.5-2B (E3)', color='#fc9272', edgecolor='black', lw=0.7)
    bars_err = ax2.bar(x2 + width/2, err_4b, width, label='Qwen3.5-4B (E3)', color='#cb181d', edgecolor='black', lw=0.7)

    for bar in bars_err:
        h = bar.get_height()
        ax2.annotate(f'{h:.2f}%', xy=(bar.get_x() + bar.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#cb181d')

    txt_reduc = 'Over 14× Reduction:\n4.66% → 0.32%' if is_en else 'Giảm hơn 14 lần:\n4.66% → 0.32%'
    ax2.annotate(txt_reduc, xy=(x2[0] + width/2, 0.35), xytext=(0.55, 4.2),
                 arrowprops=dict(arrowstyle="->", color="#990000", lw=1.2),
                 fontsize=9, fontweight='bold', color='#990000',
                 bbox=dict(boxstyle="round,pad=0.2", fc="#ffe6e6", ec="#990000", lw=0.8))

    ylabel_b = 'Syntax Error Rate (%)' if is_en else 'Tỷ lệ lỗi cú pháp / Syntax Error Rate (%)'
    title_b = '(b) Output Format Errors' if is_en else '(b) Lỗi định dạng đầu ra'

    ax2.set_ylabel(ylabel_b, fontsize=11, fontweight='bold')
    ax2.set_title(title_b, fontsize=11, fontweight='bold')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(err_cats, fontsize=9.5, fontweight='bold')
    ax2.set_ylim(0, 7.0)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    ax2.legend(loc='upper left', fontsize=9, framealpha=0.95)

    suptitle = "Model Scaling Study (Qwen3.5-2B vs. Qwen3.5-4B)" if is_en else "Nghiên cứu mở rộng quy mô (Model Scaling Study: 2B vs. 4B)"
    plt.suptitle(suptitle, fontsize=13, fontweight='bold', y=1.00)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "fig4_model_scaling.png"), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, "fig4_model_scaling.pdf"), bbox_inches='tight')
    plt.close()
    print(f"✓ [{lang.upper()}] Exported fig4_model_scaling to {output_dir}")


def generate_all():
    print("==================================================")
    print("GENERATING ACADEMIC FIGURES (VI & EN)")
    print("==================================================")
    
    # 1. Vietnamese figures -> paper/figures and latex/paper_vi/figures
    for d in ["paper/figures", "latex/paper_vi/figures"]:
        plot_fig1_architecture(lang="vi", output_dir=d)
        plot_fig2_customtools_comparison(lang="vi", output_dir=d)
        plot_fig3_stress_test(lang="vi", output_dir=d)
        plot_fig4_scaling_syntax(lang="vi", output_dir=d)

    # 2. English figures -> paper/figures_en and latex/paper_en/figures
    for d in ["paper/figures_en", "latex/paper_en/figures"]:
        plot_fig1_architecture(lang="en", output_dir=d)
        plot_fig2_customtools_comparison(lang="en", output_dir=d)
        plot_fig3_stress_test(lang="en", output_dir=d)
        plot_fig4_scaling_syntax(lang="en", output_dir=d)

    print("==================================================")
    print("ALL FIGURES (VI & EN) GENERATED SUCCESSFULLY!")
    print("==================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=["vi", "en", "all"], default="all")
    parser.add_argument("--overwrite-fig1", action="store_true", help="Regenerate Figure 1 architecture diagrams")
    args = parser.parse_args()

    if args.overwrite_fig1:
        for d in ["paper/figures", "latex/paper_vi/figures"]:
            plot_fig1_architecture(lang="vi", output_dir=d)
        for d in ["paper/figures_en", "latex/paper_en/figures"]:
            plot_fig1_architecture(lang="en", output_dir=d)

    if args.lang == "all":
        generate_all()
    elif args.lang == "vi":
        for d in ["paper/figures", "latex/paper_vi/figures"]:
            plot_fig2_customtools_comparison(lang="vi", output_dir=d)
            plot_fig3_stress_test(lang="vi", output_dir=d)
            plot_fig4_scaling_syntax(lang="vi", output_dir=d)
    elif args.lang == "en":
        for d in ["paper/figures_en", "latex/paper_en/figures"]:
            plot_fig2_customtools_comparison(lang="en", output_dir=d)
            plot_fig3_stress_test(lang="en", output_dir=d)
            plot_fig4_scaling_syntax(lang="en", output_dir=d)
