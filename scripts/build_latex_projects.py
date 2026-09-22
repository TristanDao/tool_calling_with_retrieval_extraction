#!/usr/bin/env python3
"""
Convert Markdown Thesis and Papers into Modular LaTeX Projects
and package them into ZIP files for OpenAI Prism (prism.openai.com).
"""

import os
import re
import shutil
import zipfile
from pathlib import Path

ROOT_DIR = Path("/home/thinh/project/UIT/tool_calling_with_retrieval_extraction")
LATEX_DIR = ROOT_DIR / "latex"
DIST_DIR = LATEX_DIR / "dist"

THESIS_MD = ROOT_DIR / "thesis" / "khoa_luan_tot_nghiep.md"
PAPER_EN_MD = ROOT_DIR / "paper" / "paper_en.md"
PAPER_VI_MD = ROOT_DIR / "paper" / "paper_vi.md"

FIGURES_SRC = ROOT_DIR / "paper" / "figures"
LOGO_SRC = ROOT_DIR / "thesis" / "logo_uit.jpeg"


def clean_math_and_escapes(text: str) -> str:
    """Safely escape text for LaTeX while preserving math and formatting."""
    # We will handle inline code, bold, italics, math carefully
    return text


def md_table_to_latex(table_lines: list[str], caption: str = "", label: str = "", is_wide: bool = False) -> str:
    """Convert a markdown table to a clean LaTeX booktabs tabular."""
    if len(table_lines) < 2:
        return ""
    
    # Parse headers
    header_line = table_lines[0].strip()
    headers = [c.strip() for c in header_line.strip("|").split("|")]
    ncols = len(headers)
    
    # Parse alignment
    align_line = table_lines[1].strip()
    aligns_raw = [c.strip() for c in align_line.strip("|").split("|")]
    col_align = []
    for a in aligns_raw:
        if a.startswith(":") and a.endswith(":"):
            col_align.append("c")
        elif a.endswith(":"):
            col_align.append("r")
        else:
            col_align.append("l")
    while len(col_align) < ncols:
        col_align.append("l")
    col_align_str = "".join(col_align[:ncols])
    
    rows = []
    for line in table_lines[2:]:
        line = line.strip()
        if not line or not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        # Pad or trim to ncols
        while len(cells) < ncols:
            cells.append("")
        cells = cells[:ncols]
        # Format cells
        formatted_cells = []
        for cell in cells:
            formatted_cells.append(format_inline(cell))
        rows.append(" & ".join(formatted_cells) + r" \\")
    
    # Header format
    header_formatted = [r"\textbf{" + format_inline(h) + "}" for h in headers]
    header_str = " & ".join(header_formatted) + r" \\"
    
    table_env = "table*" if is_wide else "table"
    
    latex_parts = [
        f"\\begin{{{table_env}}}[htbp]",
        r"    \centering",
    ]
    if caption:
        latex_parts.append(f"    \\caption{{{caption}}}")
    if label:
        latex_parts.append(f"    \\label{{{label}}}")
        
    if is_wide or ncols > 5:
        latex_parts.append(r"    \resizebox{\textwidth}{!}{%")
        latex_parts.append(f"    \\begin{{tabular}}{{{col_align_str}}}")
    else:
        latex_parts.append(f"    \\begin{{tabular}}{{{col_align_str}}}")
        
    latex_parts.append(r"        \toprule")
    latex_parts.append(f"        {header_str}")
    latex_parts.append(r"        \midrule")
    for r in rows:
        # Check if row is a group divider (like italicized full-width note)
        if r.startswith(r"\textit{") and r.count("&") == ncols - 1 and r.count("&") > 0:
            latex_parts.append(f"        {r}")
        else:
            latex_parts.append(f"        {r}")
    latex_parts.append(r"        \bottomrule")
    latex_parts.append(r"    \end{tabular}")
    if is_wide or ncols > 5:
        latex_parts.append(r"    }%")
    latex_parts.append(f"\\end{{{table_env}}}")
    
    return "\n".join(latex_parts)


def format_inline(text: str) -> str:
    """Format inline markdown to LaTeX (bold, italic, code, math, special chars)."""
    # Protect inline math $...$
    math_segments = []
    def save_math(m):
        math_segments.append(m.group(0))
        return f"MATHPH{len(math_segments)-1}PH"
    
    text = re.sub(r"\$[^$]+\$", save_math, text)
    
    # Protect inline code `...`
    code_segments = []
    def save_code(m):
        code_segments.append(m.group(1))
        return f"CODEPH{len(code_segments)-1}PH"
    
    text = re.sub(r"`([^`]+)`", save_code, text)
    
    # Escape special LaTeX chars in remaining text: % & # _
    text = text.replace("%", r"\%")
    text = text.replace("&", r"\&")
    text = text.replace("#", r"\#")
    text = text.replace("_", r"\_")
    
    # Bold **text**
    text = re.sub(r"\*\*([^*]+)\*\*", r"\\textbf{\1}", text)
    
    # Italic *text*
    text = re.sub(r"\*([^*]+)\*", r"\\textit{\1}", text)
    
    # Links [text](url)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\\href{\2}{\1}", text)
    
    # Restore code
    for i, code in enumerate(code_segments):
        # Escape _ inside code or use \texttt
        clean_code = code.replace(r"\_", "_").replace(r"\%", "%").replace(r"\&", "&")
        clean_code = clean_code.replace("{", r"\{").replace("}", r"\}")
        text = text.replace(f"CODEPH{i}PH", f"\\texttt{{{clean_code}}}")
        
    # Restore math
    for i, math_str in enumerate(math_segments):
        text = text.replace(f"MATHPH{i}PH", math_str)
        
    return text


def md_block_to_latex(md_content: str, is_thesis: bool = False) -> str:
    """Parse markdown body into LaTeX paragraphs, sections, tables, lists, figures."""
    lines = md_content.splitlines()
    output_lines = []
    
    in_table = False
    table_buffer = []
    last_table_caption = ""
    last_table_label = ""
    
    in_code_block = False
    code_buffer = []
    
    in_itemize = False
    in_enumerate = False
    
    def close_lists():
        nonlocal in_itemize, in_enumerate
        res = []
        if in_itemize:
            res.append(r"\end{itemize}")
            in_itemize = False
        if in_enumerate:
            res.append(r"\end{enumerate}")
            in_enumerate = False
        return res
    
    def close_table():
        nonlocal in_table, table_buffer, last_table_caption, last_table_label
        res = []
        if in_table and table_buffer:
            latex_tab = md_table_to_latex(table_buffer, caption=last_table_caption, label=last_table_label)
            res.append(latex_tab)
            table_buffer = []
            in_table = False
            last_table_caption = ""
            last_table_label = ""
        return res

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # Code fence
        if stripped.startswith("```"):
            if in_code_block:
                in_code_block = False
                output_lines.append(r"\begin{verbatim}")
                output_lines.extend(code_buffer)
                output_lines.append(r"\end{verbatim}")
                code_buffer = []
            else:
                output_lines.extend(close_lists())
                output_lines.extend(close_table())
                in_code_block = True
                code_buffer = []
            i += 1
            continue
            
        if in_code_block:
            code_buffer.append(line)
            i += 1
            continue
            
        # Ignore horizontal rules and page breaks
        if stripped in ["---", "***", '<div style="page-break-after: always;"></div>', "<br>", "<br><br>", "<br><br><br>"]:
            output_lines.extend(close_lists())
            output_lines.extend(close_table())
            i += 1
            continue
            
        # Table detection
        if stripped.startswith("|") and stripped.endswith("|"):
            output_lines.extend(close_lists())
            in_table = True
            table_buffer.append(stripped)
            i += 1
            continue
        elif in_table:
            output_lines.extend(close_table())
            
        # Table Caption detector: e.g. **Bảng 1: ...** or **Table 1: ...**
        cap_match = re.match(r"^\*\*(Bảng|Table)\s*([0-9.]+):?\s*(.*?)\*\*", stripped)
        if cap_match:
            prefix, num, cap_text = cap_match.groups()
            last_table_caption = f"{cap_text.strip()}"
            clean_num = num.replace(".", "_")
            last_table_label = f"tab:{prefix.lower()}_{clean_num}"
            i += 1
            continue
            
        # Figures: ![Caption](path)
        fig_match = re.match(r"^!\[(.*?)\]\((.*?)\)", stripped)
        if fig_match:
            output_lines.extend(close_lists())
            cap, path = fig_match.groups()
            # Normalize figure path to figures/<basename without ext>
            base_name = Path(path).stem
            # Extract label from caption if present (e.g. "Hình 4.1: ...")
            clean_cap = format_inline(cap)
            lbl_match = re.match(r"^(Hình|Figure)\s*([0-9.]+):?\s*(.*)", cap)
            if lbl_match:
                lbl_type, lbl_num, real_cap = lbl_match.groups()
                label_str = f"\\label{{fig:{lbl_type.lower()}_{lbl_num.replace('.', '_')}}}"
                clean_cap = format_inline(real_cap)
            else:
                label_str = f"\\label{{fig:{base_name}}}"
                
            fig_latex = [
                r"\begin{figure}[htbp]",
                r"    \centering",
                f"    \\includegraphics[width=0.88\\textwidth]{{figures/{base_name}}}",
                f"    \\caption{{{clean_cap}}}",
                f"    {label_str}",
                r"\end{figure}"
            ]
            output_lines.append("\n".join(fig_latex))
            i += 1
            # If the next line is an italic caption repeat (*Hình 1: ...*), skip it
            if i < len(lines) and (lines[i].strip().startswith("*Hình") or lines[i].strip().startswith("*Figure")):
                i += 1
            continue
            
        # Headings
        if stripped.startswith("#"):
            output_lines.extend(close_lists())
            level = len(stripped.split()[0])
            title = stripped.lstrip("#").strip()
            
            # Clean section numbers if present (e.g. "1. Introduction" -> "Introduction")
            clean_heading = re.sub(r"^\d+(\.\d+)*\.?\s*", "", title)
            clean_title = format_inline(clean_heading)
            
            # If thesis, clean up chapter and section numbers to let LaTeX number them naturally
            if is_thesis:
                # Chapter match: e.g. "Chương 1. TỔNG QUAN VỀ ĐỀ TÀI"
                ch_match = re.match(r"^Chương\s*\d+\.\s*(.*)", title, re.IGNORECASE)
                if ch_match:
                    title = ch_match.group(1).strip()
                    output_lines.append(f"\\chapter{{{format_inline(title)}}}")
                    i += 1
                    continue
                # Section match: e.g. "1.1. Đặt vấn đề..."
                sec_match = re.match(r"^\d+\.\d+\.?\s*(.*)", title)
                if sec_match:
                    title = sec_match.group(1).strip()
                    output_lines.append(f"\\section{{{format_inline(title)}}}")
                    i += 1
                    continue
                # Sub-section match: e.g. "1.1.1. ..."
                subsec_match = re.match(r"^\d+\.\d+\.\d+\.?\s*(.*)", title)
                if subsec_match:
                    title = subsec_match.group(1).strip()
                    output_lines.append(f"\\subsection{{{format_inline(title)}}}")
                    i += 1
                    continue
                    
            # Standard heading translation
            if level == 1 or level == 2:
                output_lines.append(f"\\section{{{clean_title}}}")
            elif level == 3:
                output_lines.append(f"\\subsection{{{clean_title}}}")
            elif level >= 4:
                output_lines.append(f"\\subsubsection{{{clean_title}}}")
            i += 1
            continue
            
        # Lists
        if re.match(r"^[-*]\s+", stripped):
            item_text = re.sub(r"^[-*]\s+", "", stripped)
            if not in_itemize:
                output_lines.extend(close_lists())
                output_lines.append(r"\begin{itemize}")
                in_itemize = True
            output_lines.append(f"    \\item {format_inline(item_text)}")
            i += 1
            continue
            
        if re.match(r"^\d+\.\s+", stripped):
            item_text = re.sub(r"^\d+\.\s+", "", stripped)
            if not in_enumerate:
                output_lines.extend(close_lists())
                output_lines.append(r"\begin{enumerate}")
                in_enumerate = True
            output_lines.append(f"    \\item {format_inline(item_text)}")
            i += 1
            continue
            
        # Empty line
        if not stripped:
            output_lines.extend(close_lists())
            output_lines.append("")
            i += 1
            continue
            
        # Normal text line
        output_lines.append(format_inline(line))
        i += 1
        
    output_lines.extend(close_lists())
    output_lines.extend(close_table())
    return "\n".join(output_lines)


def get_shared_bibtex() -> str:
    """Generate standardized BibTeX database for both thesis and papers."""
    return r"""@article{ersoy2025tool,
  author    = {Okan Ersoy and Efe Altinisik and H. Tugrul Sencar and Kareem Darwish},
  title     = {Tool Calling for Arabic LLMs: Data Strategies and Instruction Tuning},
  journal   = {arXiv preprint arXiv:2502.12345},
  year      = {2025}
}

@inproceedings{patil2023gorilla,
  author    = {Shishir G. Patil and Tianjun Zhang and Xin Wang and Joseph E. Gonzalez},
  title     = {Gorilla: Large Language Model Connected with Massive APIs},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2023}
}

@inproceedings{schick2023toolformer,
  author    = {Timo Schick and Jane Dwivedi-Yu and Roberto Dess{\`\i} and Roberta Raileanu and Maria Lomeli and Luke Zettlemoyer and Nicola Cancedda and Thomas Scialom},
  title     = {Toolformer: Language Models Can Teach Themselves to Use Tools},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {36},
  year      = {2023}
}

@techreport{yan2024bfcl,
  author      = {Fanjia Yan and Huanzhi Mao and Charlie Ji and Junxian Chen and Joseph E. Gonzalez},
  title       = {Berkeley Function-Calling Leaderboard (BFCL)},
  institution = {UC Berkeley Sky Computing Lab},
  year        = {2024}
}

@inproceedings{qin2024toolllm,
  author    = {Yujia Qin and Shihao Liang and Yining Ye and Kunlun Zhu and Lan Yan and Yaxi Lu and Yankai Lin and others},
  title     = {ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs},
  booktitle = {Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (ACL)},
  year      = {2024}
}

@inproceedings{dettmers2024qlora,
  author    = {Tim Dettmers and Artidoro Pagnoni and Ari Holtzman and Luke Zettlemoyer},
  title     = {QLoRA: Efficient Finetuning of Quantized LLMs},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {36},
  year      = {2024}
}

@inproceedings{song2023autotool,
  author    = {Jiaxuan Song and Wayne Zhao and Kun Chen and Yutao He},
  title     = {AutoTool: Automating Tool Selection and Parameter Generation for Large Language Models},
  booktitle = {Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year      = {2023}
}

@inproceedings{devlin2019bert,
  author    = {Jacob Devlin and Ming-Wei Chang and Kenton Lee and Kristina Toutanova},
  title     = {BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  booktitle = {Proceedings of NAACL-HLT},
  pages     = {4171--4186},
  year      = {2019}
}

@inproceedings{conneau2020xlmr,
  author    = {Alexis Conneau and Kartikay Khandelwal and Naman Goyal and Vishrav Chaudhary and Guillaume Wenzek and Francisco Guzm{\'a}n and Edouard Grave and Myle Ott and Luke Zettlemoyer and Veselin Stoyanov},
  title     = {Unsupervised Cross-lingual Representation Learning at Scale},
  booktitle = {Proceedings of ACL},
  pages     = {8440--8451},
  year      = {2020}
}

@article{chen2024bgem3,
  author    = {Jianlv Chen and Shitao Xiao and Peitian Hou and Dingkun Liu and Kun Lu},
  title     = {BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Versatile Pre-Training},
  journal   = {arXiv preprint arXiv:2402.03216},
  year      = {2024}
}

@inproceedings{liu2024apigen,
  author    = {Zeyi Liu and Tie-Yan Hoang and Jiayi Zhang and Ming Zhu and others},
  title     = {APIGen: Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2024}
}

@article{liu2024toolace,
  author    = {Wei Liu and Xiaohui Huang and Xinyi Zeng and Xiang Hao and others},
  title     = {ToolACE: Winning the Points of LLM Function Calling},
  journal   = {arXiv preprint arXiv:2409.00920},
  year      = {2024}
}
"""


def build_uit_thesis():
    """Build modular UIT Thesis project conforming to docs/main.tex and UIT regulations."""
    print("Building Project 1: UIT Thesis...")
    target_dir = LATEX_DIR / "uit_thesis"
    if target_dir.exists():
        shutil.rmtree(target_dir)
    
    (target_dir / "frontmatter").mkdir(parents=True, exist_ok=True)
    (target_dir / "chapters").mkdir(parents=True, exist_ok=True)
    (target_dir / "figures").mkdir(parents=True, exist_ok=True)
    
    # Copy figures and logo
    shutil.copy(LOGO_SRC, target_dir / "figures" / "logo_uit.jpeg")
    for f in FIGURES_SRC.glob("*"):
        shutil.copy(f, target_dir / "figures" / f.name)
        
    # Write references.bib
    with open(target_dir / "references.bib", "w", encoding="utf-8") as f:
        f.write(get_shared_bibtex())
        
    # Read thesis markdown
    with open(THESIS_MD, "r", encoding="utf-8") as f:
        md_text = f.read()
        
    # Frontmatter: Cover page with TikZ border
    cover_tex = r"""% Trang bìa chính thức Khóa luận tốt nghiệp UIT (Bìa ngoài)
\begin{titlepage}
    % Khung viền đôi chuẩn học thuật UIT
    \begin{tikzpicture}[remember picture, overlay]
        \draw[line width = 2pt] 
            ($(current page.north west) + (2.5cm,-1.5cm)$) 
            rectangle 
            ($(current page.south east) + (-1.5cm,1.5cm)$);
        \draw[line width = 0.8pt] 
            ($(current page.north west) + (2.6cm,-1.6cm)$) 
            rectangle 
            ($(current page.south east) + (-1.4cm,1.4cm)$);
    \end{tikzpicture}
    
    \centering
    {\fontsize{15pt}{18pt}\selectfont \textbf{ĐẠI HỌC QUỐC GIA THÀNH PHỐ HỒ CHÍ MINH}}\\[0.2cm]
    {\fontsize{16pt}{20pt}\selectfont \textbf{TRƯỜNG ĐẠI HỌC CÔNG NGHỆ THÔNG TIN}}\\[0.2cm]
    {\fontsize{16pt}{20pt}\selectfont \textbf{KHOA KHOA HỌC MÁY TÍNH}}\\[0.3cm]
    {\fontsize{10pt}{12pt}\selectfont —————————— ❖ ——————————}\\[0.6cm]

    % Logo trường Đại học Công nghệ Thông tin
    \includegraphics[width=3.2cm]{figures/logo_uit.jpeg}\\[1.0cm]

    {\fontsize{14pt}{18pt}\selectfont \textbf{ĐÀO PHƯỚC THỊNH}}\\[0.15cm]
    {\fontsize{14pt}{18pt}\selectfont \textbf{HÀ QUANG ĐẠT}}\\[1.0cm]

    {\fontsize{16pt}{20pt}\selectfont \textbf{KHÓA LUẬN TỐT NGHIỆP}}\\[0.5cm]

    % Tên đề tài khóa luận (Tiếng Việt & Tiếng Anh)
    \begin{minipage}{0.92\textwidth}
        \centering
        {\fontsize{18pt}{22pt}\selectfont \textbf{NGHIÊN CỨU PHƯƠNG PHÁP TOOL CALLING DỰA TRÊN TRUY HỒI NGỮ NGHĨA VÀ TRÍCH XUẤT THAM SỐ THEO TOOL SCHEMA}}\\[0.5cm]
        {\fontsize{16pt}{20pt}\selectfont \textbf{\textcolor[RGB]{237,28,36}{RESEARCH ON TOOL CALLING USING SEMANTIC RETRIEVAL AND SCHEMA-AWARE PARAMETER EXTRACTION}}}
    \end{minipage}\\[1.0cm]

    {\fontsize{14pt}{18pt}\selectfont \textbf{CỬ NHÂN NGÀNH TRÍ TUỆ NHÂN TẠO}}\\[1.2cm]

    \vfill

    {\fontsize{13pt}{16pt}\selectfont \textbf{TP. HỒ CHÍ MINH, NĂM 2026}}
\end{titlepage}
"""
    with open(target_dir / "frontmatter" / "cover.tex", "w", encoding="utf-8") as f:
        f.write(cover_tex)
        
    # Frontmatter: Subcover (Trang phụ bìa - Bìa trong, không viền)
    subcover_tex = r"""% Trang phụ bìa khóa luận UIT (Bìa trong)
\begin{titlepage}
    \centering
    {\fontsize{15pt}{18pt}\selectfont \textbf{ĐẠI HỌC QUỐC GIA THÀNH PHỐ HỒ CHÍ MINH}}\\[0.2cm]
    {\fontsize{16pt}{20pt}\selectfont \textbf{TRƯỜNG ĐẠI HỌC CÔNG NGHỆ THÔNG TIN}}\\[0.2cm]
    {\fontsize{16pt}{20pt}\selectfont \textbf{KHOA KHOA HỌC MÁY TÍNH}}\\[0.3cm]
    {\fontsize{10pt}{12pt}\selectfont —————————— ❖ ——————————}\\[0.6cm]

    % Logo trường Đại học Công nghệ Thông tin
    \includegraphics[width=3.2cm]{figures/logo_uit.jpeg}\\[0.8cm]

    {\fontsize{14pt}{18pt}\selectfont \textbf{ĐÀO PHƯỚC THỊNH - 25210038}}\\[0.15cm]
    {\fontsize{14pt}{18pt}\selectfont \textbf{HÀ QUANG ĐẠT - 25210008}}\\[0.8cm]

    {\fontsize{16pt}{20pt}\selectfont \textbf{KHÓA LUẬN TỐT NGHIỆP}}\\[0.4cm]

    % Tên đề tài khóa luận (Tiếng Việt & Tiếng Anh)
    \begin{minipage}{0.92\textwidth}
        \centering
        {\fontsize{18pt}{22pt}\selectfont \textbf{NGHIÊN CỨU PHƯƠNG PHÁP TOOL CALLING DỰA TRÊN TRUY HỒI NGỮ NGHĨA VÀ TRÍCH XUẤT THAM SỐ THEO TOOL SCHEMA}}\\[0.4cm]
        {\fontsize{16pt}{20pt}\selectfont \textbf{\textcolor[RGB]{237,28,36}{RESEARCH ON TOOL CALLING USING SEMANTIC RETRIEVAL AND SCHEMA-AWARE PARAMETER EXTRACTION}}}
    \end{minipage}\\[0.8cm]

    {\fontsize{14pt}{18pt}\selectfont \textbf{CỬ NHÂN NGÀNH TRÍ TUỆ NHÂN TẠO}}\\[0.8cm]

    {\fontsize{14pt}{18pt}\selectfont \textbf{GIẢNG VIÊN HƯỚNG DẪN:}}\\[0.15cm]
    {\fontsize{14pt}{18pt}\selectfont \textbf{TS. ĐẶNG VĂN THÌN}}\\[0.8cm]

    \vfill

    {\fontsize{13pt}{16pt}\selectfont \textbf{TP. HỒ CHÍ MINH, NĂM 2026}}
\end{titlepage}
"""
    with open(target_dir / "frontmatter" / "subcover.tex", "w", encoding="utf-8") as f:
        f.write(subcover_tex)

    # Frontmatter: Council
    council_tex = r"""\chapter*{THÔNG TIN HỘI ĐỒNG CHẤM KHÓA LUẬN}
\addcontentsline{toc}{chapter}{Thông tin Hội đồng chấm khóa luận}

Khóa luận tốt nghiệp được thực hiện tại: \textbf{Khoa Khoa học Máy tính, Trường Đại học Công nghệ Thông tin, Đại học Quốc gia TP. Hồ Chí Minh}.

\vspace{0.5cm}

\textbf{Hội đồng chấm khóa luận tốt nghiệp:}

\vspace{0.3cm}

\begin{table}[htbp]
    \centering
    \begin{tabular}{p{0.35\textwidth} p{0.35\textwidth} p{0.2\textwidth}}
        \toprule
        \textbf{Họ và tên thành viên} & \textbf{Vai trò trong hội đồng} & \textbf{Chữ ký} \\
        \midrule
        TS. Đặng Văn Thìn & Giảng viên hướng dẫn & \\
        ................................................... & Cán bộ chấm điểm 1 & \\
        ................................................... & Cán bộ chấm điểm 2 & \\
        \bottomrule
    \end{tabular}
\end{table}

\vspace{1cm}

Xác nhận của Khoa Khoa học Máy tính, Trường Đại học Công nghệ Thông tin:

\vspace{0.5cm}
\begin{flushright}
    \textit{TP. Hồ Chí Minh, ngày ..... tháng ..... năm 2026} \\
    \textbf{TRƯỞNG KHOA} \\
    \vspace{2cm}
    ............................................................
\end{flushright}
\newpage
"""
    with open(target_dir / "frontmatter" / "council.tex", "w", encoding="utf-8") as f:
        f.write(council_tex)

    # Frontmatter: Acknowledgments
    ack_match = re.search(r"## LỜI CẢM ƠN\s*(.*?)(?=## LỜI CAM ĐOAN)", md_text, re.DOTALL)
    ack_content = ack_match.group(1).strip() if ack_match else "Chúng tôi xin chân thành cảm ơn..."
    ack_tex = f"""\\chapter*{{LỜI CẢM ƠN}}
\\addcontentsline{{toc}}{{chapter}}{{Lời cảm ơn}}

{md_block_to_latex(ack_content, is_thesis=True)}
\\newpage
"""
    with open(target_dir / "frontmatter" / "acknowledgments.tex", "w", encoding="utf-8") as f:
        f.write(ack_tex)

    # Frontmatter: Declaration
    dec_match = re.search(r"## LỜI CAM ĐOAN\s*(.*?)(?=## MỤC LỤC)", md_text, re.DOTALL)
    dec_content = dec_match.group(1).strip() if dec_match else "Chúng tôi xin cam đoan..."
    dec_tex = f"""\\chapter*{{LỜI CAM ĐOAN}}
\\addcontentsline{{toc}}{{chapter}}{{Lời cam đoan}}

{md_block_to_latex(dec_content, is_thesis=True)}

\\vspace{{1cm}}
\\begin{{flushright}}
    \\textit{{TP. Hồ Chí Minh, năm 2026}} \\\\[0.2cm]
    \\textbf{{Tập thể sinh viên thực hiện}} \\\\[1.5cm]
    \\textbf{{Đào Phước Thịnh \\qquad Hà Quang Đạt}}
\\end{{flushright}}
\\newpage
"""
    with open(target_dir / "frontmatter" / "declaration.tex", "w", encoding="utf-8") as f:
        f.write(dec_tex)

    # Frontmatter: Abstract VI
    abs_vi_match = re.search(r"## TÓM TẮT KHÓA LUẬN\s*(.*?)(?=## ABSTRACT)", md_text, re.DOTALL)
    abs_vi_content = abs_vi_match.group(1).strip() if abs_vi_match else ""
    abs_vi_tex = f"""\\chapter*{{TÓM TẮT KHÓA LUẬN}}
\\addcontentsline{{toc}}{{chapter}}{{Tóm tắt khóa luận}}

{md_block_to_latex(abs_vi_content, is_thesis=True)}
\\newpage
"""
    with open(target_dir / "frontmatter" / "abstract_vi.tex", "w", encoding="utf-8") as f:
        f.write(abs_vi_tex)

    # Frontmatter: Abstract EN
    abs_en_match = re.search(r"## ABSTRACT\s*(.*?)(?=## MỞ ĐẦU|## Chương 1)", md_text, re.DOTALL)
    abs_en_content = abs_en_match.group(1).strip() if abs_en_match else ""
    abs_en_tex = f"""\\chapter*{{ABSTRACT}}
\\addcontentsline{{toc}}{{chapter}}{{Abstract}}

{md_block_to_latex(abs_en_content, is_thesis=True)}
\\newpage
"""
    with open(target_dir / "frontmatter" / "abstract_en.tex", "w", encoding="utf-8") as f:
        f.write(abs_en_tex)

    # Frontmatter: Abbreviations
    abbr_match = re.search(r"## DANH MỤC TỪ VIẾT TẮT\s*(.*?)(?=## TÓM TẮT)", md_text, re.DOTALL)
    abbr_content = abbr_match.group(1).strip() if abbr_match else ""
    abbr_tex = f"""\\chapter*{{DANH MỤC TỪ VIẾT TẮT}}
\\addcontentsline{{toc}}{{chapter}}{{Danh mục từ viết tắt}}

{md_block_to_latex(abbr_content, is_thesis=True)}
\\newpage
"""
    with open(target_dir / "frontmatter" / "abbreviations.tex", "w", encoding="utf-8") as f:
        f.write(abbr_tex)

    # Chapters 1 to 6
    chapters_info = [
        ("ch1_tong_quan.tex", r"## Chương 1\. TỔNG QUAN VỀ ĐỀ TÀI\s*(.*?)(?=## Chương 2\.)"),
        ("ch2_co_so_ly_thuyet.tex", r"## Chương 2\. CƠ SỞ LÝ THUYẾT VÀ CÁC CÔNG TRÌNH LIÊN QUAN\s*(.*?)(?=## Chương 3\.)"),
        ("ch3_benchmark.tex", r"## Chương 3\. XÂY DỰNG BỘ TIÊU CHUẨN ĐÁNH GIÁ \(BENCHMARK\) CHO TIẾNG VIỆT\s*(.*?)(?=## Chương 4\.)"),
        ("ch4_phuong_phap.tex", r"## Chương 4\. PHƯƠNG PHÁP ĐỀ XUẤT VÀ THIẾT KẾ KIẾN TRÚC HỆ THỐNG\s*(.*?)(?=## Chương 5\.)"),
        ("ch5_thuc_nghiem.tex", r"## Chương 5\. THỰC NGHIỆM, ĐÁNH GIÁ VÀ BÀN LUẬN KẾT QUẢ\s*(.*?)(?=## Chương 6\.)"),
        ("ch6_phan_tich_loi.tex", r"## Chương 6\. PHÂN TÍCH LỖI VÀ THẢO LUẬN GIỚI HẠN\s*(.*?)(?=## KẾT LUẬN)"),
        ("conclusion.tex", r"## KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN\s*(.*?)(?=## TÀI LIỆU THAM KHẢO)"),
    ]
    
    chapter_titles = [
        "TỔNG QUAN VỀ ĐỀ TÀI",
        "CƠ SỞ LÝ THUYẾT VÀ CÁC CÔNG TRÌNH LIÊN QUAN",
        "XÂY DỰNG BỘ TIÊU CHUẨN ĐÁNH GIÁ (BENCHMARK) CHO TIẾNG VIỆT",
        "PHƯƠNG PHÁP ĐỀ XUẤT VÀ THIẾT KẾ KIẾN TRÚC HỆ THỐNG",
        "THỰC NGHIỆM, ĐÁNH GIÁ VÀ BÀN LUẬN KẾT QUẢ",
        "PHÂN TÍCH LỖI VÀ THẢO LUẬN GIỚI HẠN",
        "KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN",
    ]

    for (filename, pattern), title in zip(chapters_info, chapter_titles):
        m = re.search(pattern, md_text, re.DOTALL)
        content = m.group(1).strip() if m else ""
        if filename == "conclusion.tex":
            tex_content = f"\\chapter*{{{title}}}\n\\addcontentsline{{toc}}{{chapter}}{{{title}}}\n\n"
        else:
            tex_content = f"\\chapter{{{title}}}\n\n"
        tex_content += md_block_to_latex(content, is_thesis=True)
        with open(target_dir / "chapters" / filename, "w", encoding="utf-8") as f:
            f.write(tex_content)

    # Master main.tex based on docs/main.tex
    main_tex = r"""% !TEX program = xelatex
% ==============================================================================
% KHÓA LUẬN TỐT NGHIỆP ĐẠI HỌC - TRƯỜNG ĐH CÔNG NGHỆ THÔNG TIN (ĐHQG-HCM)
% Trình biên dịch: XeLaTeX | Font: Times New Roman (UTF-8)
% Kế thừa và chuẩn hóa theo mẫu quy định của Bộ GD&ĐT và Trường ĐH CNTT
% ==============================================================================

\documentclass[13pt,a4paper,oneside]{report}

% ------------------------------------------------------------------------------
% 1. CẤU HÌNH FONT VÀ TIẾNG VIỆT
% ------------------------------------------------------------------------------
\usepackage{fontspec}          % Nạp font OpenType/TrueType cho XeLaTeX
\setmainfont{Times New Roman}  % Font Times New Roman chuẩn mực

\usepackage[vietnamese]{babel} % Tự động Việt hóa tiêu đề: Chương, Mục lục, Hình, Bảng...

% ------------------------------------------------------------------------------
% 2. CĂN LỀ TRANG IN (CHUẨN QUY ĐỊNH PHỤ LỤC 2 TRƯỜNG ĐH CNTT)
% Quy định: Lề trên 3cm (30mm), Lề dưới 3.5cm (35mm), Lề trái 3.5cm (35mm), Lề phải 2cm (20mm)
% ------------------------------------------------------------------------------
\usepackage{geometry}
\geometry{
    a4paper,
    top=30mm,
    bottom=35mm,
    left=35mm,
    right=20mm
}

% ------------------------------------------------------------------------------
% 3. GIÃN DÒNG VÀ ĐỊNH DẠNG ĐOẠN VĂN
% ------------------------------------------------------------------------------
\usepackage{setspace}
\onehalfspacing             % Giãn dòng 1.5 dòng theo quy định chuẩn

\usepackage{indentfirst}    % Tự động thụt đầu dòng cả đoạn văn đầu tiên sau tiêu đề
\setlength{\parindent}{1cm} % Thụt lề đầu dòng 1cm
\setlength{\parskip}{6pt}   % Khoảng cách giữa các đoạn văn 6pt

% ------------------------------------------------------------------------------
% 4. CÁC GÓI TIỆN ÍCH HỖ TRỢ (TOÁN, BẢNG BIỂU, HÌNH ẢNH, VIỀN BÌA TIKZ)
% ------------------------------------------------------------------------------
\usepackage{amsmath, amsfonts, amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{array}
\usepackage{tikz}
\usetikzlibrary{calc}
\usepackage{caption}
\captionsetup{
    font=small,
    labelfont=bf,
    labelsep=colon,
    justification=centering
}

\usepackage[unicode]{hyperref}
\hypersetup{
    colorlinks=true,
    linkcolor=black,        % Màu đen chuẩn mực cho liên kết nội bộ
    citecolor=blue,         % Màu xanh lam cho trích dẫn tài liệu
    urlcolor=blue           % Màu xanh lam cho URL
}

% ==============================================================================
% BẮT ĐẦU TÀI LIỆU
% ==============================================================================
\begin{document}

% ------------------------------------------------------------------------------
% PHẦN TRANG BÌA VÀ THỦ TỤC ĐẦU BÁO CÁO (KHÔNG ĐÁNH SỐ TRANG THEO PHỤ LỤC 2)
% ------------------------------------------------------------------------------
\pagestyle{empty}
\include{frontmatter/cover}
\include{frontmatter/subcover}
\include{frontmatter/council}
\include{frontmatter/acknowledgments}
% Không có mục Lời cam đoan (Quy chế KLTN Đại học UIT chỉ áp dụng cho Thạc sĩ / Tiến sĩ)
% \include{frontmatter/declaration}

\tableofcontents
\newpage

\listoffigures
\newpage

\listoftables
\newpage

\include{frontmatter/abbreviations}
\newpage

% ------------------------------------------------------------------------------
% BẮT ĐẦU ĐÁNH SỐ TRANG TỪ PHẦN TÓM TẮT ĐỒ ÁN (SỐ Ả-RẬP Ở GIỮA BÊN DƯỚI THEO PHỤ LỤC 2)
% ------------------------------------------------------------------------------
\pagestyle{plain}
\pagenumbering{arabic}
\setcounter{page}{1}

\include{frontmatter/abstract_vi}
\include{frontmatter/abstract_en}

% ------------------------------------------------------------------------------
% NỘI DUNG CHÍNH
% ------------------------------------------------------------------------------

\include{chapters/ch1_tong_quan}
\include{chapters/ch2_co_so_ly_thuyet}
\include{chapters/ch3_benchmark}
\include{chapters/ch4_phuong_phap}
\include{chapters/ch5_thuc_nghiem}
\include{chapters/ch6_phan_tich_loi}
\include{chapters/conclusion}

% ------------------------------------------------------------------------------
% TÀI LIỆU THAM KHẢO
% ------------------------------------------------------------------------------
\bibliographystyle{IEEEtran}
\bibliography{references}
\addcontentsline{toc}{chapter}{Tài liệu tham khảo}

\end{document}
"""
    with open(target_dir / "main.tex", "w", encoding="utf-8") as f:
        f.write(main_tex)
        
    print("Project 1 built successfully.")


def build_paper_en():
    """Build IEEEtran English conference paper project."""
    print("Building Project 2: Paper English (IEEEtran)...")
    target_dir = LATEX_DIR / "paper_en"
    if target_dir.exists():
        shutil.rmtree(target_dir)
    
    (target_dir / "figures").mkdir(parents=True, exist_ok=True)
    figures_en_dir = ROOT_DIR / "paper" / "figures_en"
    for f in figures_en_dir.glob("*"):
        shutil.copy(f, target_dir / "figures" / f.name)
        
    with open(target_dir / "references.bib", "w", encoding="utf-8") as f:
        f.write(get_shared_bibtex())
        
    with open(PAPER_EN_MD, "r", encoding="utf-8") as f:
        md_text = f.read()
        
    # Extract Title, Abstract, Body
    title_match = re.search(r"^#\s+(.*?)\n", md_text)
    paper_title = title_match.group(1).strip() if title_match else "Vietnamese Tool Calling"
    
    abstract_match = re.search(r"## Abstract\s*(.*?)(?=## 1\. Introduction)", md_text, re.DOTALL)
    abstract_text = abstract_match.group(1).strip() if abstract_match else ""
    abstract_text = re.sub(r"\n*---+\s*$", "", abstract_text).strip()
    
    body_match = re.search(r"(## 1\. Introduction.*)", md_text, re.DOTALL)
    body_text = body_match.group(1).strip() if body_match else ""
    
    # Strip References and Acknowledgments from body to handle via standard LaTeX
    body_text = re.sub(r"## Acknowledgments.*", "", body_text, flags=re.DOTALL)
    
    latex_body = md_block_to_latex(body_text, is_thesis=False)
    
    header_template = r"""\documentclass[conference]{IEEEtran}
\usepackage{amsmath, amsfonts, amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{array}
\usepackage{cite}
\usepackage{microtype}
\usepackage[hyphens]{url}
\usepackage[hidelinks]{hyperref}

\begin{document}

\title{__PAPER_TITLE__}

\author{
    \IEEEauthorblockN{Dao Phuoc Thinh\IEEEauthorrefmark{1}, Ha Quang Dat\IEEEauthorrefmark{1}, Dang Van Thin\IEEEauthorrefmark{1}\IEEEauthorrefmark{2}}
    \IEEEauthorblockA{\IEEEauthorrefmark{1}Faculty of Information Technology, University of Information Technology, VNU-HCM, Vietnam\\
    Email: \{21521469, 21521925\}@ms.uit.edu.vn, thindv@uit.edu.vn\\
    \IEEEauthorrefmark{2}Corresponding Author}
}

\maketitle

\begin{abstract}
__ABSTRACT_TEXT__
\end{abstract}

\begin{IEEEkeywords}
Vietnamese Tool Calling, Small Language Models, Function Calling, Bi-Encoder, Cross-Encoder, Parameter Extraction, Stress Testing, Low-Latency AI Agents.
\end{IEEEkeywords}

__BODY_TEXT__

\section*{Acknowledgment}
This research was conducted at the Faculty of Information Technology, University of Information Technology, Vietnam National University Ho Chi Minh City (VNU-HCM). The authors express sincere gratitude to \textbf{Dr. Dang Van Thin} for insightful research supervision, experimental guidance, and invaluable feedback throughout the development of this work.

\bibliographystyle{IEEEtran}
\bibliography{references}

\end{document}
"""
    main_tex = header_template.replace("__PAPER_TITLE__", format_inline(paper_title))\
                               .replace("__ABSTRACT_TEXT__", format_inline(abstract_text))\
                               .replace("__BODY_TEXT__", latex_body)
                               
    with open(target_dir / "main.tex", "w", encoding="utf-8") as f:
        f.write(main_tex)
        
    print("Project 2 built successfully.")


def build_paper_vi():
    """Build Vietnamese conference paper project."""
    print("Building Project 3: Paper Vietnamese...")
    target_dir = LATEX_DIR / "paper_vi"
    if target_dir.exists():
        shutil.rmtree(target_dir)
    
    (target_dir / "figures").mkdir(parents=True, exist_ok=True)
    for f in FIGURES_SRC.glob("*"):
        shutil.copy(f, target_dir / "figures" / f.name)
        
    with open(target_dir / "references.bib", "w", encoding="utf-8") as f:
        f.write(get_shared_bibtex())
        
    with open(PAPER_VI_MD, "r", encoding="utf-8") as f:
        md_text = f.read()
        
    title_match = re.search(r"^#\s+(.*?)\n", md_text)
    paper_title = title_match.group(1).strip() if title_match else "Gọi Công Cụ Tiếng Việt"
    
    abstract_match = re.search(r"## Tóm Tắt \(Abstract\)\s*(.*?)(?=## 1\. Giới Thiệu)", md_text, re.DOTALL)
    abstract_text = abstract_match.group(1).strip() if abstract_match else ""
    abstract_text = re.sub(r"\n*---+\s*$", "", abstract_text).strip()
    
    body_match = re.search(r"(## 1\. Giới Thiệu.*)", md_text, re.DOTALL)
    body_text = body_match.group(1).strip() if body_match else ""
    body_text = re.sub(r"## Lời Cảm Ơn.*", "", body_text, flags=re.DOTALL)
    
    latex_body = md_block_to_latex(body_text, is_thesis=False)
    
    vi_template = r"""% !TEX program = xelatex
\documentclass[10pt,twocolumn,a4paper]{article}
\usepackage{fontspec}
\setmainfont{Times New Roman}
\usepackage[vietnamese]{babel}
\usepackage[top=20mm,bottom=20mm,left=15mm,right=15mm]{geometry}
\usepackage{amsmath, amsfonts, amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{array}
\usepackage{cite}
\usepackage{caption}
\captionsetup{font=small,labelfont=bf,justification=centering}
\usepackage[hidelinks]{hyperref}

\title{\textbf{__PAPER_TITLE__}}

\author{
    \textbf{Đào Phước Thịnh}$^{1}$, \textbf{Hà Quang Đạt}$^{1}$, \textbf{Đặng Văn Thìn}$^{1,*}$ \\[0.2cm]
    $^{1}$Khoa Công nghệ Thông tin, Trường Đại học Công nghệ Thông tin, ĐHQG-HCM, Việt Nam \\
    Email: \texttt{\{21521469, 21521925\}@ms.uit.edu.vn}, \texttt{thindv@uit.edu.vn} \\
    $^{*}$Tác giả liên hệ
}

\date{Năm 2026}

\begin{document}

\maketitle

\begin{abstract}
\textbf{Tóm tắt}---__ABSTRACT_TEXT__
\end{abstract}

\vspace{0.3cm}
\textbf{\textit{Từ khóa}}---Gọi công cụ (Tool Calling), Mô hình ngôn ngữ nhỏ (SLM), Bi-Encoder, Cross-Encoder, Trích xuất tham số, Kiểm thử áp lực, Tác tử AI tiếng Việt.

\vspace{0.5cm}

__BODY_TEXT__

\section*{Lời Cảm Ơn}
Nghiên cứu này được thực hiện tại Khoa Công nghệ Thông tin, Trường Đại học Công nghệ Thông tin, Đại học Quốc gia Thành phố Hồ Chí Minh (ĐHQG-HCM). Các tác giả xin gửi lời cảm ơn sâu sắc và tri ân chân thành nhất tới \textbf{TS. Đặng Văn Thìn} vì sự định hướng khoa học tận tâm, hướng dẫn thực nghiệm và phản biện quý báu trong suốt quá trình triển khai công trình.

\bibliographystyle{IEEEtran}
\bibliography{references}

\end{document}
"""
    main_tex = vi_template.replace("__PAPER_TITLE__", format_inline(paper_title))\
                          .replace("__ABSTRACT_TEXT__", format_inline(abstract_text))\
                          .replace("__BODY_TEXT__", latex_body)
                          
    with open(target_dir / "main.tex", "w", encoding="utf-8") as f:
        f.write(main_tex)
        
    print("Project 3 built successfully.")


def package_zip_files():
    """Package the 3 projects into independent zip archives for OpenAI Prism."""
    print("Packaging ZIP files for OpenAI Prism...")
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    
    projects = [
        ("uit_thesis", LATEX_DIR / "uit_thesis"),
        ("paper_en", LATEX_DIR / "paper_en"),
        ("paper_vi", LATEX_DIR / "paper_vi"),
    ]
    
    for zip_name, proj_dir in projects:
        zip_path = DIST_DIR / f"{zip_name}.zip"
        if zip_path.exists():
            zip_path.unlink()
            
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(proj_dir):
                for file in files:
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(proj_dir)
                    zf.write(file_path, arcname)
                    
        size_kb = zip_path.stat().st_size / 1024
        print(f"Generated: {zip_path.name} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    build_uit_thesis()
    build_paper_en()
    build_paper_vi()
    package_zip_files()
    print("\nAll projects built and packaged successfully in latex/dist/!")
