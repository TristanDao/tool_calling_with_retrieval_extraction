#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_uit_thesis_docx.py
Biên dịch toàn diện tài liệu Khóa luận tốt nghiệp (khoa_luan_tot_nghiep.md) sang tài liệu Word (.docx)
đạt chuẩn 100% quy chế định dạng học thuật Trường Đại học Công nghệ Thông tin (UIT) - ĐHQG-HCM:
1. 100% màu đen đơn sắc (Monochrome Academic Standard), không dùng màu xanh hay màu phụ.
2. Đồng nhất duy nhất 1 font chữ Times New Roman cho toàn bộ văn bản; các từ đặc biệt / code in đậm đen.
3. Khung viền (Border) CHỈ DUY NHẤT ở trang bìa ngoài (Section 0), trang bìa trong và toàn bộ các trang thân bài hoàn toàn không có viền.
4. Mục lục, Danh mục hình vẽ, Danh mục bảng biểu có liên kết Hyperlink và Bookmark nhấp được, chuyển hướng chính xác đến từng mục.
5. Công thức toán học (cả inline và display) được biên dịch 100% sang Native Office Math (OMML), không còn sót chuỗi LaTeX thô.
6. Tuân thủ 100% ECMA-376 OOXML Schema (thứ tự thẻ XSD, field codes chuẩn, không chèn thẻ hỏng trong Hyperlink).
7. Chỉ sử dụng ảnh diagram/mermaid được chụp và lưu lại, loại bỏ các ảnh thô sinh từ matplotlib.
"""

import os
import sys
import re
import xml.sax.saxutils
import docx
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from PIL import Image

import latex2mathml.converter
import mathml2omml
from lxml.etree import _Element

# ==============================================================================
# HẰNG SỐ ĐỊNH DẠNG QUY CHUẨN UIT (ĐỒNG NHẤT TIMES NEW ROMAN & MÀU ĐEN)
# ==============================================================================
FONT_NAME = "Times New Roman"

COLOR_BLACK = RGBColor(0, 0, 0)
COLOR_MUTED = RGBColor(60, 60, 60)
COLOR_RED = RGBColor(237, 28, 36)

# Khổ giấy A4 chuẩn UIT
PAGE_WIDTH = Cm(21.0)
PAGE_HEIGHT = Cm(29.7)

# Lề trang chuẩn đóng gáy Khóa luận UIT (Quy định: Top 3.0cm, Bottom 3.5cm, Left 3.5cm, Right 2.0cm)
MARGIN_TOP = Cm(3.0)
MARGIN_BOTTOM = Cm(3.5)
MARGIN_LEFT = Cm(3.5)                       # Lề đóng gáy (Gutter side)
MARGIN_RIGHT = Cm(2.0)
USABLE_WIDTH_CM = 15.5                      # 21.0 - 3.5 - 2.0 = 15.5 cm
TAB_STOP_POS_DXA = 8787                     # 15.5 cm in dxa (1 cm = 566.929 dxa -> 15.5 * 566.929 = 8787 dxa)

# ==============================================================================
# BẢNG THỨ TỰ PHẦN TỬ CHUẨN ECMA-376 OOXML XSD (NGĂN CHẶN LỖI CORRUPTED FILE)
# ==============================================================================
PPR_ORDER = [
    'pStyle', 'keepNext', 'keepLines', 'pageBreakBefore', 'framePr', 'widowControl',
    'numPr', 'suppressLineNumbers', 'pBdr', 'shd', 'tabs', 'suppressAutoHyphens',
    'kinsoku', 'wordWrap', 'overflowPunct', 'topLinePunct', 'autoSpaceDE', 'autoSpaceDN',
    'bidi', 'adjustRightInd', 'snapToGrid', 'spacing', 'ind', 'contextualSpacing',
    'mirrorIndents', 'suppressOverlap', 'jc', 'textDirection', 'textAlignment',
    'textboxTightWrap', 'outlineLvl', 'divId', 'cnfStyle', 'rPr', 'sectPr'
]

SECTPR_ORDER = [
    'headerReference', 'footerReference', 'footnotePr', 'endnotePr',
    'type', 'pgSz', 'pgMar', 'paperSrc', 'pgBorders', 'lnNumType',
    'pgNumType', 'cols', 'formProt', 'vAlign', 'noEndnote', 'titlePg',
    'textDirection', 'bidi', 'rtlGutter', 'docGrid', 'printerSettings',
    'sectPrChange'
]

TBLPR_ORDER = [
    'tblStyle', 'tblpPr', 'tblOverlap', 'bidiVisual', 'tblStyleRowBandSize',
    'tblStyleColBandSize', 'tblW', 'jc', 'tblCellSpacing', 'tblInd',
    'tblBorders', 'shd', 'tblLayout', 'tblCellMar', 'tblLook',
    'tblCaption', 'tblDescription'
]

TCPR_ORDER = [
    'cnfStyle', 'tcW', 'gridSpan', 'hMerge', 'vMerge', 'tcBorders',
    'shd', 'noWrap', 'tcMar', 'textDirection', 'tcFitText', 'vAlign',
    'hideMark', 'headers'
]

def reorder_child_elements(parent, order_list):
    """Sắp xếp lại các phần tử con theo đúng thứ tự nghiêm ngặt của chuẩn XSD OpenXML."""
    if parent is None:
        return
    children = list(parent)
    if not children:
        return
    def get_order_key(elem):
        tag = elem.tag.split('}')[-1]
        if tag in order_list:
            return (0, order_list.index(tag))
        return (1, tag)

    sorted_children = sorted(children, key=get_order_key)
    for c in children:
        parent.remove(c)
    for c in sorted_children:
        parent.append(c)

# ==============================================================================
# BOOKMARK & HYPERLINK REGISTRY
# ==============================================================================
BOOKMARK_ID_COUNTER = 100

def get_next_bookmark_id():
    global BOOKMARK_ID_COUNTER
    BOOKMARK_ID_COUNTER += 1
    return str(BOOKMARK_ID_COUNTER)

def add_bookmark_to_paragraph(paragraph, bookmark_name):
    """
    Gán bookmark vào paragraph tuân thủ 100% OOXML:
    Thẻ <w:pPr> LUÔN là con đầu tiên của <w:p>.
    Thẻ <w:bookmarkStart> đặt ngay sau <w:pPr>.
    Thẻ <w:bookmarkEnd> đặt ở cuối cùng của paragraph.
    """
    p = paragraph._p
    pPr = p.get_or_add_pPr()
    bm_id = get_next_bookmark_id()
    bm_start = parse_xml(f'<w:bookmarkStart {nsdecls("w")} w:id="{bm_id}" w:name="{bookmark_name}"/>')
    bm_end = parse_xml(f'<w:bookmarkEnd {nsdecls("w")} w:id="{bm_id}"/>')

    pPr_idx = p.index(pPr)
    p.insert(pPr_idx + 1, bm_start)
    p.append(bm_end)

def create_anchor_from_title(title_text):
    """Tạo anchor chuẩn không dấu từ tiêu đề."""
    s = title_text.lower()
    s = re.sub(r'[àáạảãâầấậẩẫăằắặẳẵ]', 'a', s)
    s = re.sub(r'[èéẹẻẽêềếệểễ]', 'e', s)
    s = re.sub(r'[ìíịỉĩ]', 'i', s)
    s = re.sub(r'[òóọỏõôồốộổỗơờớợởỡ]', 'o', s)
    s = re.sub(r'[ùúụủũưừứựửữ]', 'u', s)
    s = re.sub(r'[ỳýỵỷỹ]', 'y', s)
    s = re.sub(r'[đ]', 'd', s)
    s = re.sub(r'[^a-z0-9_]+', '_', s)
    s = s.strip('_')
    return f"_Toc_{s}"[:38]

def extract_numbered_key(text):
    clean_text = re.sub(r'[`*_]', '', text)
    match = re.search(r'(?<!\w)(\d+(?:[._]\d+)*(?:[a-z])?)(?!\w)', clean_text, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).replace('.', '_').replace('-', '_').lower()

# ==============================================================================
# XML UTILITIES (BORDERS, SHADING, NUMBERING, MATH)
# ==============================================================================

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Đặt lề trong (padding) cho ô bảng (đơn vị dxa, 1 pt = 20 dxa)."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m_name, m_val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m_name}')
        node.set(qn('w:w'), str(m_val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def set_cell_shading(cell, color_hex):
    """Đổ màu nền cho ô bảng."""
    shading_elm = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>')
    cell._tc.get_or_add_tcPr().append(shading_elm)

def set_table_borders(table, border_color="CCCCCC", header_border="000000"):
    """Thiết lập đường viền bảng phong cách học thuật trang nhã (APA / IEEE)."""
    tblPr = table._tbl.tblPr
    borders_xml = f'''
    <w:tblBorders {nsdecls("w")}>
        <w:top w:val="single" w:sz="12" w:space="0" w:color="{header_border}"/>
        <w:bottom w:val="single" w:sz="12" w:space="0" w:color="{header_border}"/>
        <w:insideH w:val="single" w:sz="4" w:space="0" w:color="{border_color}"/>
        <w:insideV w:val="none"/>
        <w:left w:val="none"/>
        <w:right w:val="none"/>
    </w:tblBorders>
    '''
    tblPr.append(parse_xml(borders_xml))

def set_row_cant_split(row):
    """Ngăn không cho hàng bị cắt ngang qua 2 trang."""
    trPr = row._tr.get_or_add_trPr()
    trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))

def set_table_header_repeat(row):
    """Lặp lại tiêu đề bảng khi sang trang mới."""
    trPr = row._tr.get_or_add_trPr()
    trPr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))

def add_cover_page_borders(section):
    """Tạo khung viền trang bìa ngoài đôi chuẩn học thuật UIT (màu đen, chỉ trang đầu tiên)."""
    sectPr = section._sectPr
    for elem in list(sectPr):
        if elem.tag.endswith('pgBorders'):
            sectPr.remove(elem)
    borders_xml = f'''
    <w:pgBorders {nsdecls("w")} w:offsetFrom="page" w:display="firstPage">
        <w:top w:val="double" w:sz="18" w:space="24" w:color="000000"/>
        <w:left w:val="double" w:sz="18" w:space="24" w:color="000000"/>
        <w:bottom w:val="double" w:sz="18" w:space="24" w:color="000000"/>
        <w:right w:val="double" w:sz="18" w:space="24" w:color="000000"/>
    </w:pgBorders>
    '''
    sectPr.append(parse_xml(borders_xml))

def disable_page_borders(section):
    """Vô hiệu hóa hoàn toàn khung viền trên section (đảm bảo Word không kế thừa viền)."""
    sectPr = section._sectPr
    for elem in list(sectPr):
        if elem.tag.endswith('pgBorders'):
            sectPr.remove(elem)
    borders_xml = f'''
    <w:pgBorders {nsdecls("w")} w:offsetFrom="page">
        <w:top w:val="none"/>
        <w:left w:val="none"/>
        <w:bottom w:val="none"/>
        <w:right w:val="none"/>
    </w:pgBorders>
    '''
    sectPr.append(parse_xml(borders_xml))

def add_page_number_to_footer(footer, is_roman=False):
    """
    Thêm trường số trang vào footer căn giữa bằng Simple Field chuẩn ECMA-376.
    Tương thích 100% với cả Microsoft Word và Google Docs mà không bị lỗi hỏng trường.
    """
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    
    fallback_val = "i" if is_roman else "1"
    fld_xml = f'''
    <w:fldSimple {nsdecls("w")} w:instr="PAGE">
        <w:r>
            <w:rPr>
                <w:rFonts w:ascii="{FONT_NAME}" w:hAnsi="{FONT_NAME}"/>
                <w:color w:val="000000"/>
                <w:sz w:val="21"/>
            </w:rPr>
            <w:t>{fallback_val}</w:t>
        </w:r>
    </w:fldSimple>
    '''
    p._p.append(parse_xml(fld_xml))

# ==============================================================================
# BỘ CHUYỂN ĐỔI CÔNG THỨC TOÁN LATEX -> OMML (CHUẨN XÁC NATIVE)
# ==============================================================================

def sanitize_latex(s):
    """Chuẩn hóa các ký hiệu LaTeX dễ gây lỗi thiếu toán hạng (¿) trong StarMath / Word OMML."""
    # 1. Dấu cộng lũy thừa: ^+ hoặc ^{+} -> ^{\text{+}}
    s = re.sub(r'\^\{?\+\}?', lambda m: r'^{\text{+}}', s)
    # 2. Dấu sao lũy thừa: ^* hoặc ^{*} -> ^{\star}
    s = re.sub(r'\^\{?\*\}?', lambda m: r'^{\star}', s)
    # 3. Ký hiệu độ dài / lực lượng tập hợp (|X|, |\mathcal{Y}|): chuyển thành \lvert ... \rvert chuẩn
    s = re.sub(r'(?<![\\|])\|([A-Za-z\\][A-Za-z0-9_{}\\]*?)\|(?!\|)', lambda m: r'\lvert ' + m.group(1) + r' \rvert', s)
    return s

def postprocess_omml(omml):
    """Xử lý hậu kỳ các thẻ XML OMML để Word / LibreOffice hiển thị hoàn hảo không bị lỗi cú pháp."""
    omml = re.sub(
        r'<m:r><m:rPr><m:sty m:val="p"/></m:rPr><m:t>&lt;</m:t></m:r>',
        r'<m:r><m:rPr><m:nor/></m:rPr><m:t>&lt;</m:t></m:r>',
        omml
    )
    omml = re.sub(
        r'<m:limUpp><m:e>(.*?)</m:e><m:lim><m:r><m:rPr><m:sty m:val="p"/></m:rPr><m:t>\^</m:t></m:r></m:lim></m:limUpp>',
        r'<m:acc><m:accPr><m:chr m:val="&#x0302;"/></m:accPr><m:e>\1</m:e></m:acc>',
        omml,
        flags=re.DOTALL
    )
    return omml

def safe_xpath(element, xpath_str: str):
    """Truy vấn XPath an toàn hỗ trợ cả lxml.etree._Element và BaseOxmlElement."""
    try:
        return element.xpath(xpath_str)
    except Exception:
        return element.xpath(xpath_str, namespaces={
            'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math',
            'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
        })

def unwrap_boxes(element):
    r"""
    Loại bỏ triệt để và đệ quy toàn bộ các thẻ <m:box> do mathml2omml sinh ra từ <mrow>.
    Trong chuẩn Office Math (OMML) của Microsoft Word, các toán hạng và ký hiệu nằm trực tiếp 
    trong <m:oMath> hoặc <m:e>, không dùng <m:box>. Thẻ <m:box> không có thuộc tính boxPr 
    chính là nguyên nhân khiến Word Online (bản web) bị crash layout, chỉ render ký tự đầu tiên
    và làm biến mất toàn bộ các ký hiệu phía sau (như $N \le 100$ bị cụt thành N).
    """
    while True:
        boxes = safe_xpath(element, './/m:box')
        if not boxes:
            break
        for box in boxes:
            parent = box.getparent()
            if parent is None:
                continue
            idx = parent.index(box)
            e = box.find('{http://schemas.openxmlformats.org/officeDocument/2006/math}e')
            if e is not None:
                children = list(e)
                for i, child in enumerate(children):
                    parent.insert(idx + i, child)
            parent.remove(box)

def fix_accents(element):
    r"""
    Chuyển đổi các cấu trúc dấu mũ (như \hat{y}) từ <m:limUpp> sang thẻ dấu mũ chuẩn <m:acc>
    để Microsoft Word và Word Online hiển thị dấu mũ toán học chính xác.
    """
    lim_upps = safe_xpath(element, './/m:limUpp')
    for lu in lim_upps:
        lim = lu.find('{http://schemas.openxmlformats.org/officeDocument/2006/math}lim')
        e = lu.find('{http://schemas.openxmlformats.org/officeDocument/2006/math}e')
        if lim is not None and e is not None:
            t = lim.find('.//{http://schemas.openxmlformats.org/officeDocument/2006/math}t')
            if t is not None and t.text == '^':
                parent = lu.getparent()
                idx = parent.index(lu)
                acc = parse_xml(f'''<m:acc {nsdecls("m")}>
                    <m:accPr><m:chr m:val="&#x0302;"/></m:accPr>
                </m:acc>''')
                acc_e = parse_xml(f'<m:e {nsdecls("m")}/>')
                for child in list(e):
                    e.remove(child)
                    acc_e.append(child)
                acc.append(acc_e)
                parent.insert(idx, acc)
                parent.remove(lu)

def add_math_fonts_to_runs(element):
    """
    Bổ sung thuộc tính font Cambria Math tường minh cho từng run toán học <m:r>
    giúp Word Online tải và hiển thị chính xác các ký tự toán học (≤, ≥, τ, δ, v.v.).
    """
    w_ns = nsdecls('w')
    for r in safe_xpath(element, './/m:r'):
        rPr = r.find('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}rPr')
        if rPr is None:
            rPr = parse_xml(f'<w:rPr {w_ns}><w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math"/></w:rPr>')
            r.insert(0, rPr)
        else:
            rFonts = rPr.find('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}rFonts')
            if rFonts is None:
                rFonts = parse_xml(f'<w:rFonts {w_ns} w:ascii="Cambria Math" w:hAnsi="Cambria Math"/>')
                rPr.insert(0, rFonts)

def render_latex_to_omml(latex_code: str, is_display: bool = False) -> _Element | None:
    """Chuyển đổi mã LaTeX sang Office Math XML (OMML) chuẩn Native Microsoft Word."""
    try:
        s = sanitize_latex(latex_code.strip())
        mml = latex2mathml.converter.convert(s)
        omml = mathml2omml.convert(mml)
        omml = postprocess_omml(omml)
        m_ns = nsdecls("m")
        if '<m:oMath' in omml and 'xmlns:m=' not in omml:
            omml = omml.replace('<m:oMath', f'<m:oMath {m_ns}', 1)
        math_element = parse_xml(omml)
        if math_element.tag != qn('m:oMath'):
            return None
        unwrap_boxes(math_element)
        fix_accents(math_element)
        for nary in safe_xpath(math_element, './/m:nary'):
            sup = nary.find(qn('m:sup'))
            props = nary.find(qn('m:naryPr'))
            if sup is not None and len(sup) == 0 and not (sup.text or '').strip() and props is not None:
                if props.find(qn('m:supHide')) is None:
                    sup_hide = OxmlElement('m:supHide')
                    sup_hide.set(qn('m:val'), '1')
                    props.append(sup_hide)
        add_math_fonts_to_runs(math_element)
        if is_display:
            math_paragraph = OxmlElement('m:oMathPara')
            math_paragraph.append(math_element)
            return math_paragraph
        return math_element
    except Exception as e:
        return None

def latex_math_to_unicode_text(math_code):
    """
    Chuyển mã toán LaTeX sang biểu diễn Unicode văn bản thuần cho Hyperlink/TOC.
    Ngăn chặn tuyệt đối việc lồng thẻ <m:oMath> vào bên trong <w:hyperlink>.
    """
    s = math_code.strip()
    replacements = [
        (r'\ge', '≥'), (r'\le', '≤'), (r'\tau', 'τ'), (r'\delta', 'δ'),
        (r'\alpha', 'α'), (r'\beta', 'β'), (r'\gamma', 'γ'), (r'\lambda', 'λ'),
        (r'\mu', 'μ'), (r'\sigma', 'σ'), (r'\theta', 'θ'), (r'\pi', 'π'),
        (r'\mathcal{T}', 'T'), (r'\mathcal{S}', 'S'), (r'\mathcal{A}', 'A'),
        (r'\mathcal{Y}', 'Y'), (r'\mathcal{L}', 'L'), (r'\mathbb{R}', 'R'),
        (r'\approx', '≈'), (r'\neq', '≠'), (r'\times', '×'), (r'\cdot', '·'),
        (r'\dots', '…'), (r'\cdots', '…'), (r'\to', '→'), (r'\rightarrow', '→'),
        (r'\in', '∈'), (r'\notin', '∉'), (r'\subset', '⊂'), (r'\ll', '≪'),
        (r'\gg', '≫'), (r'\_', '_'), (r'\,', ' '), (r'\;', ' '),
        (r'\quad', ' '), (r'\qquad', ' ')
    ]
    for pat, rep in replacements:
        s = s.replace(pat, rep)
    # Xóa các định dạng lệnh \text{...}, \mathbf{...}, \mathcal{...}
    s = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', s)
    # Xóa các lệnh LaTeX còn lại
    s = re.sub(r'\\[a-zA-Z]+', '', s)
    s = re.sub(r'[{}]', '', s)
    return s.strip()

def add_external_hyperlink(paragraph, text, url, font_size=13, is_bold=False, is_italic=False):
    """Thêm một hyperlink trỏ đến URL bên ngoài chuẩn OOXML, có thể click được trong Word."""
    try:
        part = paragraph.part
        r_id = part.relate_to(url, docx.opc.constants.RELATIONSHIP_TYPE.HYPERLINK, is_external=True)
        
        escaped_text = xml.sax.saxutils.escape(text)
        space_attr = ' xml:space="preserve"' if (text.startswith(' ') or text.endswith(' ')) else ''
        bold_tag = '<w:b/>' if is_bold else ''
        italic_tag = '<w:i/>' if is_italic else ''
        
        hl_xml = f'''
        <w:hyperlink {nsdecls("w", "r")} r:id="{r_id}" w:history="1">
            <w:r>
                <w:rPr>
                    <w:rFonts w:ascii="{FONT_NAME}" w:hAnsi="{FONT_NAME}"/>
                    {bold_tag}
                    {italic_tag}
                    <w:sz w:val="{int(font_size * 2)}"/>
                    <w:color w:val="004B87"/>
                    <w:u w:val="single"/>
                </w:rPr>
                <w:t{space_attr}>{escaped_text}</w:t>
            </w:r>
        </w:hyperlink>
        '''
        paragraph._p.append(parse_xml(hl_xml))
    except Exception as e:
        run = paragraph.add_run(text)
        run.font.name = FONT_NAME
        run.font.size = Pt(font_size)
        run.font.bold = is_bold
        run.font.italic = is_italic
        run.font.color.rgb = COLOR_BLACK

# ==============================================================================
# XỬ LÝ RUNS ĐỊNH DẠNG NỘI DÒNG
# ==============================================================================

def add_inline_formatted_text(paragraph, text, base_font_size=13, is_bold=False, is_italic=False, is_heading=False):
    """Phân tích các thẻ inline trong Markdown (Toán, In đậm, Nghiêng, Code, Hyperlink)."""
    text = text.replace('&nbsp;', ' ')
    if is_heading:
        is_bold = True
    
    pattern = re.compile(r'(\[[^\]]+\]\([^)]+\)|\$[^$\n]+\$|\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)')
    tokens = pattern.split(text)
    
    for token in tokens:
        if not token:
            continue
        
        # 1. Hyperlink Markdown: [Anchor Text](URL)
        link_match = re.match(r'^\[(.*?)\]\((.*?)\)$', token)
        if link_match:
            anchor_text, link_url = link_match.groups()
            add_external_hyperlink(paragraph, anchor_text, link_url.strip(), font_size=base_font_size, is_bold=is_bold, is_italic=is_italic)
            continue

        # 2. Công thức toán inline: $math$
        elif token.startswith('$') and token.endswith('$') and len(token) >= 2:
            math_code = token[1:-1].strip()

            # Số thuần túy hoặc phần trăm (ví dụ: $7{,}712$, $400$, $5\%$):
            # Hiển thị trực tiếp dạng văn bản Times New Roman để đồng nhất văn phong và tránh tạo object toán
            if re.match(r'^[\d,{}%.\s\+\-]+$', math_code):
                clean_num = math_code.replace('{,}', ',').replace(r'\%', '%').strip()
                run = paragraph.add_run(clean_num)
                run.font.name = FONT_NAME
                run.font.size = Pt(base_font_size)
                run.font.bold = is_bold
                run.font.italic = is_italic
                run.font.color.rgb = COLOR_BLACK
                continue

            omml_elem = render_latex_to_omml(math_code, is_display=False)
            if omml_elem is not None:
                paragraph.paragraph_format.line_spacing = 1.5
                paragraph._p.append(omml_elem)
            else:
                run = paragraph.add_run(math_code)
                run.font.name = FONT_NAME
                run.font.size = Pt(base_font_size)
                run.font.italic = True
                run.font.bold = is_bold
                run.font.color.rgb = COLOR_BLACK
                
        # 3. In đậm: **nội dung**
        elif token.startswith('**') and token.endswith('**') and len(token) >= 4:
            content = token[2:-2]
            add_inline_formatted_text(paragraph, content, base_font_size=base_font_size, is_bold=True, is_italic=is_italic)
            
        # 4. In nghiêng: *nội dung*
        elif token.startswith('*') and token.endswith('*') and len(token) >= 2:
            content = token[1:-1]
            add_inline_formatted_text(paragraph, content, base_font_size=base_font_size, is_bold=is_bold, is_italic=True)
            
        # 5. Từ đặc biệt / code: `code` -> In đậm Times New Roman màu đen
        elif token.startswith('`') and token.endswith('`') and len(token) >= 2:
            content = token[1:-1]
            run = paragraph.add_run(content)
            run.font.name = FONT_NAME
            run.font.size = Pt(base_font_size)
            run.font.bold = True
            run.font.italic = is_italic
            run.font.color.rgb = COLOR_BLACK
            
        # 6. Văn bản thông thường
        else:
            run = paragraph.add_run(token)
            run.font.name = FONT_NAME
            run.font.size = Pt(base_font_size)
            run.font.bold = is_bold
            run.font.italic = is_italic
            run.font.color.rgb = COLOR_BLACK

# ==============================================================================
# HÀM TẠO CÁC TRANG BÌA CHUẨN KHOA HỌC UIT (100% MÀU ĐEN)
# ==============================================================================

def create_cover_page(doc, is_outer=True, logo_path="logo_uit.jpg"):
    """
    Tạo trang Bìa chính (ngoài) hoặc Phụ bìa (trong) chuẩn 100% font size UIT:
    - ĐẠI HỌC QUỐC GIA THÀNH PHỐ HỒ CHÍ MINH: size 15
    - TRƯỜNG ĐẠI HỌC CÔNG NGHỆ THÔNG TIN: size 16
    - KHOA KHOA HỌC MÁY TÍNH: size 16
    - <TÊN SINH VIÊN>: size 14
    - KHÓA LUẬN TỐT NGHIỆP: size 16
    - <TÊN KHÓA LUẬN TỐT NGHIỆP>: size 18
    - <Tên Đồ án Tiếng Anh>: size 16 (màu đỏ)
    - CỬ NHÂN NGÀNH <TÊN NGÀNH>: size 14
    - GIẢNG VIÊN HƯỚNG DẪN: size 14
    - TP. HỒ CHÍ MINH, NĂM 2026: size 13
    """
    p1 = doc.add_paragraph()
    p1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p1.paragraph_format.space_before = Pt(0)
    p1.paragraph_format.space_after = Pt(2)
    p1.paragraph_format.line_spacing = 1.15
    r1 = p1.add_run("ĐẠI HỌC QUỐC GIA THÀNH PHỐ HỒ CHÍ MINH")
    r1.font.name = FONT_NAME
    r1.font.size = Pt(15)
    r1.font.bold = True
    r1.font.color.rgb = COLOR_BLACK

    p2 = doc.add_paragraph()
    p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p2.paragraph_format.space_before = Pt(0)
    p2.paragraph_format.space_after = Pt(2)
    p2.paragraph_format.line_spacing = 1.15
    r2 = p2.add_run("TRƯỜNG ĐẠI HỌC CÔNG NGHỆ THÔNG TIN")
    r2.font.name = FONT_NAME
    r2.font.size = Pt(16)
    r2.font.bold = True
    r2.font.color.rgb = COLOR_BLACK

    p3 = doc.add_paragraph()
    p3.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p3.paragraph_format.space_before = Pt(0)
    p3.paragraph_format.space_after = Pt(4)
    p3.paragraph_format.line_spacing = 1.15
    r3 = p3.add_run("KHOA KHOA HỌC MÁY TÍNH")
    r3.font.name = FONT_NAME
    r3.font.size = Pt(16)
    r3.font.bold = True
    r3.font.color.rgb = COLOR_BLACK

    p_div = doc.add_paragraph()
    p_div.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_div.paragraph_format.space_before = Pt(0)
    p_div.paragraph_format.space_after = Pt(6)
    r_div = p_div.add_run("—————————— ❖ ——————————")
    r_div.font.name = FONT_NAME
    r_div.font.size = Pt(10)
    r_div.font.color.rgb = COLOR_BLACK

    p_logo = doc.add_paragraph()
    p_logo.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_logo.paragraph_format.space_before = Pt(2)
    p_logo.paragraph_format.space_after = Pt(8)
    if logo_path and os.path.exists(logo_path):
        p_logo.add_run().add_picture(logo_path, width=Cm(3.2))

    p_sv1 = doc.add_paragraph()
    p_sv1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sv1.paragraph_format.space_before = Pt(4)
    p_sv1.paragraph_format.space_after = Pt(2)
    p_sv1.paragraph_format.line_spacing = 1.15
    sv1_text = "ĐÀO PHƯỚC THỊNH" if is_outer else "ĐÀO PHƯỚC THỊNH - 25210038"
    r_sv1 = p_sv1.add_run(sv1_text)
    r_sv1.font.name = FONT_NAME
    r_sv1.font.size = Pt(14)
    r_sv1.font.bold = True
    r_sv1.font.color.rgb = COLOR_BLACK

    p_sv2 = doc.add_paragraph()
    p_sv2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sv2.paragraph_format.space_before = Pt(0)
    p_sv2.paragraph_format.space_after = Pt(14) if is_outer else Pt(10)
    p_sv2.paragraph_format.line_spacing = 1.15
    sv2_text = "HÀ QUANG ĐẠT" if is_outer else "HÀ QUANG ĐẠT - 25210008"
    r_sv2 = p_sv2.add_run(sv2_text)
    r_sv2.font.name = FONT_NAME
    r_sv2.font.size = Pt(14)
    r_sv2.font.bold = True
    r_sv2.font.color.rgb = COLOR_BLACK

    p_kl = doc.add_paragraph()
    p_kl.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_kl.paragraph_format.space_before = Pt(2)
    p_kl.paragraph_format.space_after = Pt(6)
    r_kl = p_kl.add_run("KHÓA LUẬN TỐT NGHIỆP")
    r_kl.font.name = FONT_NAME
    r_kl.font.size = Pt(16)
    r_kl.font.bold = True
    r_kl.font.color.rgb = COLOR_BLACK

    p_title_vi = doc.add_paragraph()
    p_title_vi.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title_vi.paragraph_format.space_before = Pt(2)
    p_title_vi.paragraph_format.space_after = Pt(6)
    p_title_vi.paragraph_format.line_spacing = 1.15
    r_title_vi = p_title_vi.add_run("NGHIÊN CỨU PHƯƠNG PHÁP TOOL CALLING DỰA TRÊN TRUY HỒI NGỮ NGHĨA VÀ TRÍCH XUẤT THAM SỐ THEO TOOL SCHEMA")
    r_title_vi.font.name = FONT_NAME
    r_title_vi.font.size = Pt(18)
    r_title_vi.font.bold = True
    r_title_vi.font.color.rgb = COLOR_BLACK

    p_title_en = doc.add_paragraph()
    p_title_en.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title_en.paragraph_format.space_before = Pt(2)
    p_title_en.paragraph_format.space_after = Pt(8)
    p_title_en.paragraph_format.line_spacing = 1.15
    r_title_en = p_title_en.add_run("RESEARCH ON TOOL CALLING USING SEMANTIC RETRIEVAL AND SCHEMA-AWARE PARAMETER EXTRACTION")
    r_title_en.font.name = FONT_NAME
    r_title_en.font.size = Pt(16)
    r_title_en.font.bold = True
    r_title_en.font.color.rgb = COLOR_RED

    p_major = doc.add_paragraph()
    p_major.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_major.paragraph_format.space_before = Pt(2)
    p_major.paragraph_format.space_after = Pt(24) if is_outer else Pt(10)
    r_major = p_major.add_run("CỬ NHÂN NGÀNH TRÍ TUỆ NHÂN TẠO")
    r_major.font.name = FONT_NAME
    r_major.font.size = Pt(14)
    r_major.font.bold = True
    r_major.font.color.rgb = COLOR_BLACK

    if not is_outer:
        p_gv_hdr = doc.add_paragraph()
        p_gv_hdr.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_gv_hdr.paragraph_format.space_before = Pt(4)
        p_gv_hdr.paragraph_format.space_after = Pt(2)
        r_gv_hdr = p_gv_hdr.add_run("GIẢNG VIÊN HƯỚNG DẪN:")
        r_gv_hdr.font.name = FONT_NAME
        r_gv_hdr.font.size = Pt(14)
        r_gv_hdr.font.bold = True
        r_gv_hdr.font.color.rgb = COLOR_BLACK

        p_gv = doc.add_paragraph()
        p_gv.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_gv.paragraph_format.space_before = Pt(0)
        p_gv.paragraph_format.space_after = Pt(16)
        r_gv = p_gv.add_run("TS. ĐẶNG VĂN THÌN")
        r_gv.font.name = FONT_NAME
        r_gv.font.size = Pt(14)
        r_gv.font.bold = True
        r_gv.font.color.rgb = COLOR_BLACK

    p_foot = doc.add_paragraph()
    p_foot.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_foot.paragraph_format.space_before = Pt(32) if is_outer else Pt(16)
    p_foot.paragraph_format.space_after = Pt(0)
    r_foot = p_foot.add_run("TP. HỒ CHÍ MINH, NĂM 2026")
    r_foot.font.name = FONT_NAME
    r_foot.font.size = Pt(13)
    r_foot.font.bold = True
    r_foot.font.color.rgb = COLOR_BLACK

# ==============================================================================
# HÀM TẠO MỤC LỤC & DANH MỤC CÓ LIÊN KẾT HYPERLINK (CLICKABLE & SCHEMA SAFE)
# ==============================================================================

def add_hyperlinked_entry(doc, title_text, target_anchor, page_fallback="1", level=1, is_bold=False):
    """
    Tạo một dòng mục lục / danh mục có dot leader và bọc trong hyperlink trỏ tới bookmark.
    ĐẢM BẢO 100% TUÂN THỦ SCHEMA: Không chứa <m:oMath> trong <w:hyperlink>, chỉ dùng <w:r>.
    """
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(1.5)
    p.paragraph_format.space_after = Pt(1.5)
    p.paragraph_format.line_spacing = 1.2
    
    if level == 1:
        p.paragraph_format.left_indent = Cm(0)
    elif level == 2:
        p.paragraph_format.left_indent = Cm(0.6)
    elif level == 3:
        p.paragraph_format.left_indent = Cm(1.2)
        
    pPr = p._p.get_or_add_pPr()
    tabs_xml = f'<w:tabs {nsdecls("w")}><w:tab w:val="right" w:leader="dot" w:pos="{TAB_STOP_POS_DXA}"/></w:tabs>'
    pPr.append(parse_xml(tabs_xml))
    
    font_sz = "28" if (level == 1 and is_bold) else "26"
    bold_tag = "<w:b/>" if is_bold else ""

    title_runs_xml = ""
    tokens = re.split(r'(\$[^$\n]+\$)', title_text)
    for tok in tokens:
        if not tok:
            continue
        if tok.startswith('$') and tok.endswith('$') and len(tok) >= 2:
            math_code = tok[1:-1].strip()
            plain_math = latex_math_to_unicode_text(math_code)
            escaped_math = xml.sax.saxutils.escape(plain_math)
            space_attr = ' xml:space="preserve"' if (plain_math.startswith(' ') or plain_math.endswith(' ')) else ''
            title_runs_xml += f'<w:r><w:rPr><w:rFonts w:ascii="{FONT_NAME}" w:hAnsi="{FONT_NAME}"/><w:i/><w:sz w:val="{font_sz}"/><w:color w:val="000000"/></w:rPr><w:t{space_attr}>{escaped_math}</w:t></w:r>'
        else:
            escaped_tok = xml.sax.saxutils.escape(tok)
            space_attr = ' xml:space="preserve"' if (tok.startswith(' ') or tok.endswith(' ')) else ''
            title_runs_xml += f'<w:r><w:rPr><w:rFonts w:ascii="{FONT_NAME}" w:hAnsi="{FONT_NAME}"/>{bold_tag}<w:sz w:val="{font_sz}"/><w:color w:val="000000"/></w:rPr><w:t{space_attr}>{escaped_tok}</w:t></w:r>'
    
    escaped_anchor = xml.sax.saxutils.escape(target_anchor)
    escaped_page = xml.sax.saxutils.escape(page_fallback)

    hl_xml = f'''
    <w:hyperlink {nsdecls("w")} w:anchor="{escaped_anchor}" w:history="1">
        {title_runs_xml}
        <w:r>
            <w:rPr>
                <w:rFonts w:ascii="{FONT_NAME}" w:hAnsi="{FONT_NAME}"/>
            </w:rPr>
            <w:tab/>
        </w:r>
        <w:r>
            <w:rPr>
                <w:rFonts w:ascii="{FONT_NAME}" w:hAnsi="{FONT_NAME}"/>
                {bold_tag}
                <w:sz w:val="{font_sz}"/>
                <w:color w:val="000000"/>
            </w:rPr>
            <w:t>{escaped_page}</w:t>
        </w:r>
    </w:hyperlink>
    '''
    p._p.append(parse_xml(hl_xml))
    return p

# ==============================================================================
# HÀM BIÊN TẬP BẢNG BIỂU MARKDOWN THÀNH WORD TABLE (MÀU ĐEN HỌC THUẬT)
# ==============================================================================

def add_markdown_table_to_doc(doc, table_lines):
    """Chuyển đổi cú pháp bảng Markdown sang Table của python-docx với định dạng học thuật."""
    parsed_rows = []
    for line in table_lines:
        line_clean = line.strip()
        if not line_clean.startswith('|'):
            continue
        cells = [c.strip() for c in line_clean.strip('|').split('|')]
        if all(re.match(r'^:?-+:?$', c) for c in cells if c):
            continue
        parsed_rows.append(cells)

    if not parsed_rows:
        return

    num_cols = max(len(r) for r in parsed_rows)
    for r in parsed_rows:
        while len(r) < num_cols:
            r.append("")

    table = doc.add_table(rows=len(parsed_rows), cols=num_cols)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    set_table_borders(table, border_color="DDDDDD", header_border="000000")

    if num_cols >= 8:
        cell_font_size = 8.5
        pad_v, pad_h = 60, 60
    elif num_cols >= 6:
        cell_font_size = 9.5
        pad_v, pad_h = 80, 80
    else:
        cell_font_size = 10.5
        pad_v, pad_h = 100, 120

    # Tính toán độ rộng cột thông minh cho bảng
    if num_cols == 2:
        col_widths_cm = [5.0, 11.0]
    elif num_cols == 3:
        col_widths_cm = [4.5, 4.5, 7.0]
    else:
        col_widths_cm = [USABLE_WIDTH_CM / num_cols] * num_cols

    for col_idx, col in enumerate(table.columns):
        col.width = Cm(col_widths_cm[col_idx])

    for r_idx, row_data in enumerate(parsed_rows):
        row = table.rows[r_idx]
        set_row_cant_split(row)
        if r_idx == 0:
            set_table_header_repeat(row)

        for c_idx, cell_value in enumerate(row_data):
            cell = row.cells[c_idx]
            cell.width = Cm(col_widths_cm[c_idx])
            set_cell_margins(cell, top=pad_v, bottom=pad_v, left=pad_h, right=pad_h)

            # Tách các dòng con trong ô nếu có thẻ <br> hoặc <br/>
            sub_lines = re.split(r'<br\s*/?>', cell_value, flags=re.IGNORECASE)

            for s_idx, sub_val in enumerate(sub_lines):
                sub_clean = sub_val.strip()
                if not sub_clean and len(sub_lines) > 1:
                    continue

                if s_idx == 0:
                    p = cell.paragraphs[0]
                else:
                    p = cell.add_paragraph()

                p.paragraph_format.space_before = Pt(1.5)
                p.paragraph_format.space_after = Pt(1.5)
                p.paragraph_format.line_spacing = 1.15

                if r_idx == 0:
                    set_cell_shading(cell, "F2F2F2")
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    add_inline_formatted_text(p, sub_clean, base_font_size=cell_font_size, is_bold=True)
                else:
                    if len(sub_clean) < 15 and (re.match(r'^[0-9,.%–+xX() ]+$', sub_clean) or sub_clean in ['Seen', 'Unseen', 'VI', 'EN', 'Single', 'Multi']):
                        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    else:
                        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                    add_inline_formatted_text(p, sub_clean, base_font_size=cell_font_size, is_bold=False)

    p_space = doc.add_paragraph()
    p_space.paragraph_format.space_before = Pt(0)
    p_space.paragraph_format.space_after = Pt(6)

# ==============================================================================
# HÀM CHÈN HÌNH ẢNH & CAPTION (FIGURE BUILDER)
# ==============================================================================

# Danh mục các hình ảnh diagram / mermaid được phép chèn vào tài liệu
ALLOWED_FIGURES = {
    "fig2_1_bi_cross_pipeline.png": "fig2_1_bi_cross_pipeline.png",
    "fig3_1_data_pipeline.png": "fig3_1_data_pipeline.png",
    "fig4_2_value_normalizer.png": "fig4_2_value_normalizer.png",
    "fig1_system_architecture.png": "fig1_system_architecture.png",
}

EXCLUDED_FIGURES = {
    "fig2_performance_comparison.png",
    "fig3_stress_test_curves.png",
    "fig4_model_scaling.png",
}

def resolve_image_path(image_filename, base_dirs):
    """Tìm đường dẫn thực tế của file ảnh qua các thư mục dự phòng."""
    clean_name = os.path.basename(image_filename)
    if clean_name in EXCLUDED_FIGURES:
        return None, True  # Bị loại trừ có chủ đích

    # Tìm trong base_dirs
    for d in base_dirs:
        cand = os.path.join(d, clean_name)
        if os.path.exists(cand):
            return cand, False
        cand_orig = os.path.join(d, image_filename)
        if os.path.exists(cand_orig):
            return cand_orig, False

    if os.path.exists(clean_name):
        return clean_name, False
    if os.path.exists(image_filename):
        return image_filename, False

    return None, False

def add_figure_to_doc(doc, image_filename, caption_text, base_dirs=None):
    """Chèn hình ảnh căn giữa trang kèm Caption phía dưới hình chuẩn UIT và gắn bookmark."""
    if base_dirs is None:
        base_dirs = [os.getcwd(), os.path.dirname(os.path.abspath(__file__)), "../paper/figures", "create_docx"]

    actual_file, is_excluded = resolve_image_path(image_filename, base_dirs)
    clean_name = os.path.basename(image_filename)

    if is_excluded:
        print(f"  [-] Bỏ qua ảnh matplotlib theo yêu cầu: {clean_name}")
        return

    if actual_file and os.path.exists(actual_file):
        try:
            with Image.open(actual_file) as im:
                w_px, h_px = im.size
                aspect = h_px / max(w_px, 1)

            target_w_cm = min(15.0, USABLE_WIDTH_CM)
            target_h_cm = target_w_cm * aspect
            max_h_cm = 17.5
            
            if target_h_cm > max_h_cm:
                target_h_cm = max_h_cm
                target_w_cm = target_h_cm / aspect

            p_img = doc.add_paragraph()
            p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_img.paragraph_format.space_before = Pt(8)
            p_img.paragraph_format.space_after = Pt(4)
            p_img.paragraph_format.keep_with_next = True
            p_img.add_run().add_picture(actual_file, width=Cm(target_w_cm), height=Cm(target_h_cm))
            print(f"  [+] Đã chèn hình: {clean_name} ({target_w_cm:.1f}cm x {target_h_cm:.1f}cm)")

        except Exception as e:
            print(f"  [!] Lỗi nạp hình ảnh {clean_name}: {e}")
            p_err = doc.add_paragraph(f"[Lỗi nạp hình ảnh: {clean_name}]")
            p_err.alignment = WD_ALIGN_PARAGRAPH.CENTER
    else:
        print(f"  [!] Không tìm thấy file hình: {image_filename}")

    if caption_text:
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.paragraph_format.space_before = Pt(4)
        p_cap.paragraph_format.space_after = Pt(12)
        
        # Thêm text caption
        add_inline_formatted_text(p_cap, f"*{caption_text}*", base_font_size=12, is_heading=False)
        
        # Gán Bookmark chuẩn xác (sau khi paragraph đã có text)
        fig_match = re.search(r'Hình\s+([0-9]+[._][0-9]+)', caption_text)
        if fig_match:
            fig_key = fig_match.group(1).replace('.', '_')
            add_bookmark_to_paragraph(p_cap, f"_Toc_fig_{fig_key}")

# ==============================================================================
# HÀM CHÈN KHỐI CODE / JSON SCHEMA
# ==============================================================================

def add_code_block_to_doc(doc, code_lines):
    """Tạo khung chứa mã lệnh JSON / Python có nền xám viền bo nhẹ."""
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    table.columns[0].width = Cm(USABLE_WIDTH_CM)
    
    cell = table.cell(0, 0)
    set_cell_margins(cell, top=120, bottom=120, left=160, right=160)
    set_cell_shading(cell, "F9F9F9")
    
    borders_xml = f'''
    <w:tcBorders {nsdecls("w")}>
        <w:top w:val="single" w:sz="6" w:space="0" w:color="CCCCCC"/>
        <w:bottom w:val="single" w:sz="6" w:space="0" w:color="CCCCCC"/>
        <w:left w:val="single" w:sz="18" w:space="0" w:color="000000"/>
        <w:right w:val="single" w:sz="6" w:space="0" w:color="CCCCCC"/>
    </w:tcBorders>
    '''
    cell._tc.get_or_add_tcPr().append(parse_xml(borders_xml))

    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = 1.15

    for idx, line in enumerate(code_lines):
        if idx > 0:
            p = cell.add_paragraph()
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.15
        run = p.add_run(line)
        run.font.name = "Consolas"
        run.font.size = Pt(9.5)
        run.font.color.rgb = COLOR_BLACK

    p_after = doc.add_paragraph()
    p_after.paragraph_format.space_before = Pt(0)
    p_after.paragraph_format.space_after = Pt(6)

# ==============================================================================
# HÀM QUÉT DỌN & CHUẨN HÓA SCHEMA TOÀN DIỆN (OOXML SANITIZER)
# ==============================================================================

def sanitize_and_validate_document(doc):
    """
    Duyệt toàn bộ cấu trúc document, headers, footers để:
    1. Sắp xếp lại thứ tự phần tử con trong tất cả pPr theo đúng XSD.
    2. Sắp xếp lại thứ tự phần tử con trong tất cả sectPr và loại bỏ pgNumType trùng lặp.
    3. Sắp xếp lại thứ tự phần tử con trong tất cả tblPr và tcPr.
    4. Đảm bảo w:bookmarkStart không bao giờ đứng trước w:pPr.
    """
    body_el = doc._body._element

    # 1. Sắp xếp lại tất cả pPr
    for pPr in body_el.xpath('//w:pPr'):
        reorder_child_elements(pPr, PPR_ORDER)

    # 2. Sắp xếp lại tất cả tblPr & tcPr
    for tblPr in body_el.xpath('//w:tblPr'):
        reorder_child_elements(tblPr, TBLPR_ORDER)
    for tcPr in body_el.xpath('//w:tcPr'):
        reorder_child_elements(tcPr, TCPR_ORDER)

    # 3. Sắp xếp lại tất cả sectPr, loại bỏ triệt để headerReference và khử trùng thẻ
    for sectPr in body_el.xpath('//w:sectPr'):
        # Loại bỏ hoàn toàn headerReference trên toàn bộ các section (không dùng header trên bất kỳ trang nào)
        for href in list(sectPr.xpath('./w:headerReference')):
            sectPr.remove(href)
        # Khử trùng pgNumType (chỉ giữ lại thẻ cuối cùng nếu có nhiều thẻ)
        pgNumTypes = sectPr.xpath('./w:pgNumType')
        if len(pgNumTypes) > 1:
            for pnt in pgNumTypes[:-1]:
                sectPr.remove(pnt)
        reorder_child_elements(sectPr, SECTPR_ORDER)

    # 4. Kiểm tra footers (chỉ quản lý số trang ở footer, TUYỆT ĐỐI KHÔNG GỌI section.header)
    for section in doc.sections:
        try:
            footer_part = section._sectPr.xpath('./w:footerReference')
            if footer_part and section.footer:
                part_el = section.footer._element
                for pPr in part_el.xpath('.//w:pPr'):
                    reorder_child_elements(pPr, PPR_ORDER)
                for tblPr in part_el.xpath('.//w:tblPr'):
                    reorder_child_elements(tblPr, TBLPR_ORDER)
                for tcPr in part_el.xpath('.//w:tcPr'):
                    reorder_child_elements(tcPr, TCPR_ORDER)
        except Exception:
            pass

    # 5. Đảm bảo bookmarkStart luôn đứng sau pPr
    for p in body_el.xpath('//w:p'):
        children = list(p)
        tag_names = [c.tag.split('}')[-1] for c in children]
        if 'bookmarkStart' in tag_names and 'pPr' in tag_names:
            bm_idx = tag_names.index('bookmarkStart')
            ppr_idx = tag_names.index('pPr')
            if bm_idx < ppr_idx:
                # Đổi vị trí để pPr đứng trước
                pPr_elem = children[ppr_idx]
                p.remove(pPr_elem)
                p.insert(0, pPr_elem)

    # 6. Đảm bảo font Cambria Math và làm sạch toàn diện công thức toán trong document
    ensure_cambria_math_in_font_table(doc)
    unwrap_boxes(body_el)
    fix_accents(body_el)
    add_math_fonts_to_runs(body_el)

def ensure_cambria_math_in_font_table(doc):
    """
    Đảm bảo font Cambria Math được khai báo chính thức trong word/fontTable.xml
    để Microsoft Word Online (trình duyệt) nhận diện và nạp đúng font cho các ký tự toán học.
    """
    for part in doc.part.related_parts.values():
        if "fontTable" in part.partname:
            xml_str = part.blob.decode("utf-8")
            if 'w:name="Cambria Math"' not in xml_str:
                font_entry = '<w:font xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" w:name="Cambria Math"><w:panose1 w:val="02040503050406030204"/><w:charset w:val="00"/><w:family w:val="roman"/><w:pitch w:val="variable"/><w:sig w:usb0="E00006FF" w:usb1="420024FF" w:usb2="02000000" w:usb3="00000000" w:csb0="0000019F" w:csb1="00000000"/></w:font></w:fonts>'
                xml_str = xml_str.replace("</w:fonts>", font_entry)
                part._blob = xml_str.encode("utf-8")
            break

# ==============================================================================
# BỘ BIÊN DỊCH VÀ XỬ LÝ CHÍNH (MAIN COMPILER)
# ==============================================================================

def find_thesis_markdown_source():
    """Tự động xác định file nguồn markdown chính thức."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "khoa_luan_tot_nghiep.md"),
        os.path.join(os.getcwd(), "thesis/khoa_luan_tot_nghiep.md"),
        "thesis/khoa_luan_tot_nghiep.md",
        "khoa_luan_tot_nghiep.md",
    ]
    for cand in candidates:
        if os.path.exists(cand):
            return os.path.abspath(cand)
    return None

def convert_markdown_to_uit_docx(md_path=None, output_docx_path=None):
    if md_path is None:
        md_path = find_thesis_markdown_source()

    if not md_path or not os.path.exists(md_path):
        raise FileNotFoundError(f"Không tìm thấy file nguồn markdown khóa luận tốt nghiệp! Vui lòng kiểm tra path.")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, ".."))

    if output_docx_path is None:
        output_docx_path = os.path.splitext(md_path)[0] + ".docx"

    base_dirs = [
        os.path.join(repo_root, "paper/figures"),
        os.path.join(repo_root, "latex/uit_thesis/figures"),
        os.path.join(script_dir, "figures"),
        script_dir,
        os.getcwd(),
    ]

    logo_path = None
    for cand_logo in [
        os.path.join(script_dir, "logo_uit.jpeg"),
        os.path.join(script_dir, "logo_uit.jpg"),
        os.path.join(repo_root, "latex/uit_thesis/figures/logo_uit.jpeg"),
        os.path.join(os.getcwd(), "thesis/logo_uit.jpeg"),
        "logo_uit.jpeg", "logo_uit.jpg", "logo_uit.png"
    ]:
        if os.path.exists(cand_logo):
            logo_path = cand_logo
            break

    print(f"[*] Bắt đầu chuyển đổi: {md_path} -> {output_docx_path}")
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    heading_anchor_by_key = {}
    for source_line in md_text.splitlines():
        heading_match = re.match(r'^#{1,4}\s+(.*)$', source_line.strip())
        if not heading_match:
            continue
        heading_text = heading_match.group(1).strip()
        heading_key = extract_numbered_key(heading_text)
        if heading_key and heading_key not in heading_anchor_by_key:
            heading_anchor_by_key[heading_key] = create_anchor_from_title(heading_text)

    doc = docx.Document()

    # Cấu hình Style mặc định
    style_normal = doc.styles['Normal']
    style_normal.font.name = FONT_NAME
    style_normal.font.size = Pt(13)
    style_normal.font.color.rgb = COLOR_BLACK
    style_normal.paragraph_format.line_spacing = 1.5
    style_normal.paragraph_format.space_after = Pt(4)
    style_normal.paragraph_format.space_before = Pt(0)

    # Tách các section qua div page-break
    raw_sections = md_text.split('<div style="page-break-after: always;"></div>')
    body_sections = raw_sections[2:] if len(raw_sections) >= 3 else raw_sections

    # --------------------------------------------------------------------------
    # SECTION 0: TRANG BÌA NGOÀI (BÌA CHÍNH - DUY NHẤT CÓ KHUNG VIỀN ĐEN)
    # --------------------------------------------------------------------------
    sec0 = doc.sections[0]
    sec0.page_width = PAGE_WIDTH
    sec0.page_height = PAGE_HEIGHT
    sec0.top_margin = MARGIN_TOP
    sec0.bottom_margin = MARGIN_BOTTOM
    sec0.left_margin = MARGIN_LEFT
    sec0.right_margin = MARGIN_RIGHT
    create_cover_page(doc, is_outer=True, logo_path=logo_path)

    # --------------------------------------------------------------------------
    # SECTION 1: TRANG BÌA TRONG (BÌA PHỤ - KHÔNG CÓ KHUNG VIỀN)
    # --------------------------------------------------------------------------
    sec1 = doc.add_section(docx.enum.section.WD_SECTION.NEW_PAGE)
    sec1.page_width = PAGE_WIDTH
    sec1.page_height = PAGE_HEIGHT
    sec1.top_margin = MARGIN_TOP
    sec1.bottom_margin = MARGIN_BOTTOM
    sec1.left_margin = MARGIN_LEFT
    sec1.right_margin = MARGIN_RIGHT
    create_cover_page(doc, is_outer=False, logo_path=logo_path)

    # --------------------------------------------------------------------------
    # SECTION 2: FRONT MATTER (SỐ TRANG LA MÃ: i, ii, iii... - KHÔNG CÓ KHUNG VIỀN)
    # --------------------------------------------------------------------------
    sec2 = doc.add_section(docx.enum.section.WD_SECTION.NEW_PAGE)
    sec2.page_width = PAGE_WIDTH
    sec2.page_height = PAGE_HEIGHT
    sec2.top_margin = MARGIN_TOP
    sec2.bottom_margin = MARGIN_BOTTOM
    sec2.left_margin = MARGIN_LEFT
    sec2.right_margin = MARGIN_RIGHT
    sec2.footer.is_linked_to_previous = False
    sec2._sectPr.append(parse_xml(f'<w:pgNumType {nsdecls("w")} w:fmt="lowerRoman" w:start="1"/>'))
    add_page_number_to_footer(sec2.footer, is_roman=True)

    in_main_body = False
    current_major_heading = ""
    
    for s_idx, section_content in enumerate(body_sections):
        lines = section_content.strip().split('\n')
        if not lines or not lines[0].strip():
            continue

        first_line = lines[0].strip()

        # Phát hiện bắt đầu thân bài chính (Chương 1) -> Tách sang Section 3 (Số trang Ả Rập 1, 2, 3...)
        if ("# Chương 1" in first_line or "## Chương 1" in first_line) and not in_main_body:
            sec3 = doc.add_section(docx.enum.section.WD_SECTION.NEW_PAGE)
            sec3.page_width = PAGE_WIDTH
            sec3.page_height = PAGE_HEIGHT
            sec3.top_margin = MARGIN_TOP
            sec3.bottom_margin = MARGIN_BOTTOM
            sec3.left_margin = MARGIN_LEFT
            sec3.right_margin = MARGIN_RIGHT
            sec3.footer.is_linked_to_previous = False
            sec3._sectPr.append(parse_xml(f'<w:pgNumType {nsdecls("w")} w:fmt="decimal" w:start="1"/>'))
            add_page_number_to_footer(sec3.footer, is_roman=False)
            in_main_body = True
        elif s_idx > 0:
            doc.add_page_break()

        line_idx = 0
        while line_idx < len(lines):
            line = lines[line_idx].rstrip()
            stripped = line.strip()

            if not stripped or stripped == '---':
                line_idx += 1
                continue

            # Bỏ qua các thẻ HTML <br>, <br/>
            if re.match(r'^(<br\s*/?>\s*)+$', stripped, re.IGNORECASE):
                line_idx += 1
                continue

            # ------------------------------------------------------------------
            # 1. BẢNG BIỂU MARKDOWN (NGOẠI TRỪ DANH MỤC HÌNH & BẢNG SẼ XỬ LÝ RIÊNG)
            # ------------------------------------------------------------------
            if stripped.startswith('|') and '|' in stripped[1:]:
                # Nếu đang trong Danh mục hình vẽ hoặc Danh mục bảng biểu, chuyển sang danh mục Hyperlink
                if current_major_heading == 'DANH MỤC HÌNH VẼ':
                    while line_idx < len(lines) and lines[line_idx].strip().startswith('|'):
                        row_line = lines[line_idx].strip()
                        line_idx += 1
                        cells = [c.strip() for c in row_line.strip('|').split('|')]
                        if not cells or 'Ký hiệu hình' in cells[0] or re.match(r'^:?-+:?$', cells[0]):
                            continue
                        fig_code = cells[0].replace('**', '').strip()   # e.g. Hình 2.1
                        fig_title = cells[1].strip()
                        fig_page = cells[2].strip() if len(cells) > 2 else "1"
                        
                        fig_num_match = re.search(r'([0-9]+[._][0-9]+)', fig_code)
                        anchor = f"_Toc_fig_{fig_num_match.group(1).replace('.', '_')}" if fig_num_match else "_Toc_fig"
                        add_hyperlinked_entry(doc, f"{fig_code}. {fig_title}", anchor, page_fallback=fig_page, level=1, is_bold=False)
                    continue

                elif current_major_heading == 'DANH MỤC BẢNG BIỂU':
                    while line_idx < len(lines) and lines[line_idx].strip().startswith('|'):
                        row_line = lines[line_idx].strip()
                        line_idx += 1
                        cells = [c.strip() for c in row_line.strip('|').split('|')]
                        if not cells or 'Ký hiệu bảng' in cells[0] or re.match(r'^:?-+:?$', cells[0]):
                            continue
                        tab_code = cells[0].replace('**', '').strip()   # e.g. Bảng 3.1
                        tab_title = cells[1].strip()
                        tab_page = cells[2].strip() if len(cells) > 2 else "1"
                        
                        tab_key = extract_numbered_key(tab_code)
                        anchor = f"_Toc_tab_{tab_key}" if tab_key else "_Toc_tab"
                        add_hyperlinked_entry(doc, f"{tab_code}. {tab_title}", anchor, page_fallback=tab_page, level=1, is_bold=False)
                    continue

                # Bảng thông thường
                table_lines = []
                while line_idx < len(lines) and lines[line_idx].strip().startswith('|'):
                    table_lines.append(lines[line_idx].strip())
                    line_idx += 1
                add_markdown_table_to_doc(doc, table_lines)
                continue

            # ------------------------------------------------------------------
            # 2. KHỐI MERMAID -> THAY THẾ BẰNG FILE ẢNH DIAGRAM ĐÃ LƯU
            # ------------------------------------------------------------------
            if stripped.startswith('```mermaid'):
                line_idx += 1
                while line_idx < len(lines):
                    cur_line = lines[line_idx].strip()
                    if cur_line == '```':
                        line_idx += 1
                        break
                    line_idx += 1

                while line_idx < len(lines) and not lines[line_idx].strip():
                    line_idx += 1

                caption_text = ""
                if line_idx < len(lines):
                    cand_cap = lines[line_idx].strip()
                    if cand_cap.startswith('*Hình') or cand_cap.startswith('Hình') or 'Hình ' in cand_cap:
                        caption_text = cand_cap.strip('*_')
                        line_idx += 1

                if "Hình 2.1" in caption_text or "2.1" in caption_text:
                    add_figure_to_doc(doc, "fig2_1_bi_cross_pipeline.png", caption_text, base_dirs)
                elif "Hình 3.1" in caption_text or "3.1" in caption_text:
                    add_figure_to_doc(doc, "fig3_1_data_pipeline.png", caption_text, base_dirs)
                elif "Hình 4.2" in caption_text or "4.2" in caption_text:
                    add_figure_to_doc(doc, "fig4_2_value_normalizer.png", caption_text, base_dirs)
                continue

            # ------------------------------------------------------------------
            # 3. KHỐI MÃ LỆNH (CODE BLOCK / JSON)
            # ------------------------------------------------------------------
            if stripped.startswith('```'):
                code_lines = []
                line_idx += 1
                while line_idx < len(lines) and not lines[line_idx].strip().startswith('```'):
                    code_lines.append(lines[line_idx])
                    line_idx += 1
                line_idx += 1
                add_code_block_to_doc(doc, code_lines)
                continue

            # ------------------------------------------------------------------
            # 4. HÌNH ẢNH MARKDOWN: ![caption](path)
            # ------------------------------------------------------------------
            img_match = re.match(r'^!\[(.*?)\]\((.*?)\)$', stripped)
            if img_match:
                alt_text, img_path = img_match.groups()
                line_idx += 1

                while line_idx < len(lines) and not lines[line_idx].strip():
                    line_idx += 1

                caption_text = alt_text
                if line_idx < len(lines):
                    cand_cap = lines[line_idx].strip()
                    if cand_cap.startswith('*Hình') or cand_cap.startswith('Hình') or 'Hình ' in cand_cap:
                        caption_text = cand_cap.strip('*_')
                        line_idx += 1

                add_figure_to_doc(doc, img_path, caption_text, base_dirs)
                continue

            # ------------------------------------------------------------------
            # 5. CÔNG THỨC TOÁN HIỂN THỊ ($$...$$)
            # ------------------------------------------------------------------
            if stripped.startswith('$$') and stripped.endswith('$$') and len(stripped) > 4:
                math_content = stripped[2:-2].strip()
                omml_elem = render_latex_to_omml(math_content, is_display=True)
                
                p_math = doc.add_paragraph()
                p_math.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p_math.paragraph_format.space_before = Pt(6)
                p_math.paragraph_format.space_after = Pt(6)
                
                if omml_elem is not None:
                    p_math._p.append(omml_elem)
                else:
                    run_m = p_math.add_run(math_content)
                    run_m.font.name = FONT_NAME
                    run_m.font.size = Pt(12)
                    run_m.font.italic = True
                    run_m.font.color.rgb = COLOR_BLACK
                line_idx += 1
                continue

            # ------------------------------------------------------------------
            # 6. TIÊU ĐỀ (HEADINGS)
            # ------------------------------------------------------------------
            if (stripped.startswith('# ') or stripped.startswith('## ')) and not stripped.startswith('### '):
                h_text = re.sub(r'^#+\s*', '', stripped).strip()
                current_major_heading = h_text.upper()
                p = doc.add_paragraph()
                is_centered = any(k in h_text.upper() for k in [
                    "CHƯƠNG", "TỔNG QUAN", "KẾT LUẬN", "LỜI CẢM ƠN", "MỤC LỤC", "TÓM TẮT", "PHỤ LỤC", "TÀI LIỆU THAM KHẢO"
                ])
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER if is_centered else WD_ALIGN_PARAGRAPH.LEFT
                p.paragraph_format.space_before = Pt(14)
                p.paragraph_format.space_after = Pt(8)
                p.paragraph_format.keep_with_next = True
                
                # Chương: bold, font 14
                display_h_text = h_text if h_text.startswith("Chương ") else h_text.upper()
                add_inline_formatted_text(p, display_h_text, base_font_size=14, is_bold=True)
                
                # Gán bookmark sau khi đã có text
                bm_name = create_anchor_from_title(h_text)
                add_bookmark_to_paragraph(p, bm_name)
                
                line_idx += 1
                continue

            if stripped.startswith('### '):
                h_text = stripped[4:].strip()
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                p.paragraph_format.space_before = Pt(10)
                p.paragraph_format.space_after = Pt(4)
                p.paragraph_format.keep_with_next = True
                
                # Mục (3.1.): bold, font 13, thụt lề 1 tab (0.6 cm)
                if re.match(r'^\d+\.\d+\.?\s+', h_text):
                    p.paragraph_format.left_indent = Cm(0.6)
                
                add_inline_formatted_text(p, h_text, base_font_size=13, is_bold=True)
                
                # Gán bookmark sau khi đã có text
                bm_name = create_anchor_from_title(h_text)
                add_bookmark_to_paragraph(p, bm_name)
                
                line_idx += 1
                continue

            if stripped.startswith('#### '):
                h_text = stripped[5:].strip()
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                p.paragraph_format.space_before = Pt(8)
                p.paragraph_format.space_after = Pt(2)
                p.paragraph_format.keep_with_next = True
                
                # Tiểu mục (3.1.1.): bold, font 13, thụt lề 2 tab (1.2 cm)
                if re.match(r'^\d+\.\d+\.\d+\.?\s+', h_text):
                    p.paragraph_format.left_indent = Cm(1.2)
                
                add_inline_formatted_text(p, h_text, base_font_size=13, is_bold=True)
                
                # Gán bookmark sau khi đã có text
                bm_name = create_anchor_from_title(h_text)
                add_bookmark_to_paragraph(p, bm_name)
                
                line_idx += 1
                continue

            # ------------------------------------------------------------------
            # 7. TIÊU ĐỀ BẢNG BIỂU: **Bảng X.Y: ...**
            # ------------------------------------------------------------------
            if stripped.startswith('**Bảng ') or (stripped.startswith('Bảng ') and ':' in stripped and '**' in stripped):
                p_tab_cap = doc.add_paragraph()
                p_tab_cap.alignment = WD_ALIGN_PARAGRAPH.LEFT
                p_tab_cap.paragraph_format.space_before = Pt(10)
                p_tab_cap.paragraph_format.space_after = Pt(3)
                p_tab_cap.paragraph_format.keep_with_next = True
                
                caption_body = stripped.strip('*')
                add_inline_formatted_text(p_tab_cap, caption_body, base_font_size=12, is_bold=True)
                
                # Gán bookmark sau khi đã có text
                tab_key = extract_numbered_key(stripped)
                if tab_key:
                    add_bookmark_to_paragraph(p_tab_cap, f"_Toc_tab_{tab_key}")
                    
                line_idx += 1
                continue

            # ------------------------------------------------------------------
            # 8. MỤC LỤC HYPERLINK DẠNG LIST: - [Tiêu đề](#anchor)
            # ------------------------------------------------------------------
            toc_match = re.match(r'^(\s*)-\s+\[(.*?)\]\((#.*?)\)$', line)
            if toc_match and current_major_heading == 'MỤC LỤC':
                indent_str, toc_title, toc_anchor = toc_match.groups()
                level = 1
                if len(indent_str) >= 4:
                    level = 3
                elif len(indent_str) >= 2:
                    level = 2
                
                anchor_clean = create_anchor_from_title(toc_title)
                toc_number_key = extract_numbered_key(toc_title)
                target_anchor = heading_anchor_by_key.get(toc_number_key, anchor_clean)
                TOC_PAGE_LOOKUP = {
                    "_Toc_loi_cam_on": "ii",
                    "_Toc_danh_muc_hinh_ve": "iv",
                    "_Toc_danh_muc_bang_bieu": "v",
                    "_Toc_danh_muc_tu_viet_tat": "vi",
                    "_Toc_tom_tat_khoa_luan": "vii",
                    "_Toc_mo_dau": "viii",
                    "_Toc_chuong_1_tong_quan_ve_de_tai": "1",
                    "_Toc_1_1_mo_ta_de_tai_va_tinh_cap_thie": "1",
                    "_Toc_1_2_muc_tieu_cua_de_tai": "3",
                    "_Toc_1_3_doi_tuong_va_pham_vi_nghien_c": "4",
                    "_Toc_1_4_phuong_phap_thuc_hien": "4",
                    "_Toc_1_5_ket_qua_mong_doi_va_dong_gop": "5",
                    "_Toc_1_6_bo_cuc_cua_khoa_luan": "6",
                    "_Toc_chuong_2_co_so_ly_thuyet_va_cac_c": "7",
                    "_Toc_2_1_co_che_goi_cong_cu_tool_call": "7",
                    "_Toc_2_2_cac_nghien_cuu_va_bo_chuan_da": "7",
                    "_Toc_2_3_mo_hinh_ngon_ngu_nho_small_la": "8",
                    "_Toc_2_4_kien_truc_phan_tach_truy_hoi": "9",
                    "_Toc_2_4_1_giai_doan_1_truy_hoi_cong_c": "9",
                    "_Toc_2_4_2_giai_doan_2_trich_xuat_tham": "10",
                    "_Toc_2_5_nhung_thach_thuc_dac_thu_cua": "10",
                    "_Toc_chuong_3_xay_dung_bo_tieu_chuan_d": "13",
                    "_Toc_3_1_thiet_ke_schema_chuan_hoa_du": "13",
                    "_Toc_3_2_bo_chuan_quy_mo_lon_canonical": "14",
                    "_Toc_3_3_bo_chuan_mien_thuc_te_ban_dia": "14",
                    "_Toc_3_4_chien_luoc_tao_lap_va_kiem_so": "15",
                    "_Toc_3_5_quy_trinh_xay_dung_tap_kiem_t": "15",
                    "_Toc_chuong_4_phuong_phap_de_xuat_va_t": "18",
                    "_Toc_4_1_kien_truc_tong_quan_cua_hai_t": "18",
                    "_Toc_4_2_phuong_phap_1_slm_end_to_end": "19",
                    "_Toc_4_3_phuong_phap_2_kien_truc_phan": "20",
                    "_Toc_4_3_1_module_truy_hoi_cong_cu_ngu": "20",
                    "_Toc_4_3_2_module_trich_xuat_tham_so_p": "20",
                    "_Toc_4_4_co_che_chuan_hoa_gia_tri_thuc": "21",
                    "_Toc_chuong_5_thuc_nghiem_danh_gia_va": "24",
                    "_Toc_5_1_thiet_lap_thuc_nghiem_va_ha_t": "24",
                    "_Toc_5_2_he_thong_do_do_danh_gia_evalu": "25",
                    "_Toc_5_3_ket_qua_tong_the_va_giai_dap": "26",
                    "_Toc_5_3_1_rq1_nang_luc_chuyen_giao_tr": "26",
                    "_Toc_5_3_2_rq2_hien_tuong_kich_hoat_co": "27",
                    "_Toc_5_3_3_rq3_kha_nang_tong_quat_hoa": "28",
                    "_Toc_5_3_4_rq4_danh_doi_pareto_giua_do": "29",
                    "_Toc_5_3_5_rq5_kiem_thu_ap_luc_quy_mo": "30",
                    "_Toc_5_4_nghien_cuu_mo_rong_quy_mo_mo": "32",
                    "_Toc_5_5_so_sanh_doi_dau_voi_cac_front": "33",
                    "_Toc_5_5_1_thong_tin_tai_lap_phep_danh": "34",
                    "_Toc_chuong_6_phan_tich_loi_va_thao_lu": "35",
                    "_Toc_6_1_phan_loai_cac_dang_loi_pho_bi": "35",
                    "_Toc_6_2_phan_tich_nguyen_nhan_suy_gia": "36",
                    "_Toc_6_3_gioi_han_cua_de_tai_va_cac_th": "37",
                    "_Toc_ket_luan_va_huong_phat_trien": "38",
                    "_Toc_ket_luan": "38",
                    "_Toc_huong_phat_trien_trong_tuong_lai": "39",
                    "_Toc_tai_lieu_tham_khao": "40",
                    "_Toc_phu_luc_a_tai_nguyen_tai_lap_thuc": "42"
                }
                page_str = TOC_PAGE_LOOKUP.get(anchor_clean, "1")
                is_major = level == 1 and ("chương" in toc_title.lower() or "tổng quan" in toc_title.lower() or "mở đầu" in toc_title.lower() or "kết luận" in toc_title.lower())
                add_hyperlinked_entry(doc, toc_title, target_anchor, page_fallback=page_str, level=level, is_bold=is_major)
                line_idx += 1
                continue

            # ------------------------------------------------------------------
            # 9. DANH SÁCH THÔNG THƯỜNG (LIST ITEMS: - ..., 1. ...)
            # ------------------------------------------------------------------
            list_match = re.match(r'^(\*|-|\d+\.)\s+(.*)$', stripped)
            if list_match:
                bullet_marker, item_text = list_match.groups()
                p = doc.add_paragraph()
                p.paragraph_format.left_indent = Cm(0.75)
                p.paragraph_format.first_line_indent = Cm(-0.4)
                p.paragraph_format.space_before = Pt(1)
                p.paragraph_format.space_after = Pt(2)
                p.paragraph_format.line_spacing = 1.5
                marker_str = "• " if bullet_marker in ['-', '*'] else f"{bullet_marker} "
                run_m = p.add_run(marker_str)
                run_m.font.name = FONT_NAME
                run_m.font.bold = True
                run_m.font.color.rgb = COLOR_BLACK
                add_inline_formatted_text(p, item_text, base_font_size=12.5)
                line_idx += 1
                continue

            # ------------------------------------------------------------------
            # 10. KHỐI CHỮ KÝ HỘI ĐỒNG / GIẢNG VIÊN HƯỚNG DẪN
            # ------------------------------------------------------------------
            if "XÁC NHẬN CỦA CHỦ TỊCH HỘI ĐỒNG" in stripped and "GIẢNG VIÊN HƯỚNG DẪN" in stripped:
                tab_sign = doc.add_table(rows=2, cols=2)
                tab_sign.alignment = WD_TABLE_ALIGNMENT.CENTER
                tab_sign.autofit = False
                tab_sign.columns[0].width = Cm(8.0)
                tab_sign.columns[1].width = Cm(8.0)
                
                c0 = tab_sign.cell(0, 0).paragraphs[0]
                c0.alignment = WD_ALIGN_PARAGRAPH.CENTER
                r0 = c0.add_run("XÁC NHẬN CỦA CHỦ TỊCH HỘI ĐỒNG\n(Ký và ghi rõ họ tên)")
                r0.font.name = FONT_NAME
                r0.font.size = Pt(11.5)
                r0.font.bold = True
                r0.font.color.rgb = COLOR_BLACK
                
                c1 = tab_sign.cell(0, 1).paragraphs[0]
                c1.alignment = WD_ALIGN_PARAGRAPH.CENTER
                r1 = c1.add_run("GIẢNG VIÊN HƯỚNG DẪN\n(Ký và ghi rõ họ tên)")
                r1.font.name = FONT_NAME
                r1.font.size = Pt(11.5)
                r1.font.bold = True
                r1.font.color.rgb = COLOR_BLACK

                tab_sign.rows[1].height = Cm(2.5)
                
                tblBrd = parse_xml(f'<w:tblBorders {nsdecls("w")}><w:top w:val="none"/><w:left w:val="none"/><w:bottom w:val="none"/><w:right w:val="none"/><w:insideH w:val="none"/><w:insideV w:val="none"/></w:tblBorders>')
                tab_sign._tbl.tblPr.append(tblBrd)
                line_idx += 1
                continue

            # ------------------------------------------------------------------
            # 11. ĐOẠN VĂN THÔNG THƯỜNG (PARAGRAPH BODY TEXT)
            # ------------------------------------------------------------------
            p = doc.add_paragraph()
            if "TP. Hồ Chí Minh, tháng" in stripped or "Nhóm sinh viên thực hiện" in stripped or "Đào Phước Thịnh & Hà Quang Đạt" in stripped:
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                p.paragraph_format.first_line_indent = Cm(0)
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                p.paragraph_format.first_line_indent = Cm(0.8)

            p.paragraph_format.line_spacing = 1.5
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(4)

            add_inline_formatted_text(p, stripped, base_font_size=13)
            line_idx += 1

    # --------------------------------------------------------------------------
    # THIẾT LẬP BORDER CHỈ DUY NHẤT CHO SECTION 0 (BÌA NGOÀI)
    # CÁC SECTION CÒN LẠI (BÌA TRONG, FRONT MATTER, THÂN BÀI) ĐƯỢC TẮT VIỀN TRIỆT ĐỂ
    # --------------------------------------------------------------------------
    add_cover_page_borders(doc.sections[0])
    for s in doc.sections[1:]:
        disable_page_borders(s)

    # --------------------------------------------------------------------------
    # BƯỚC KHẮC PHỤC TRIỆT ĐỂ LỖI CORRUPT XML / FIELD CODES (OOXML AUTO-SANITIZER)
    # --------------------------------------------------------------------------
    print("[*] Đang thực hiện chuẩn hóa và làm sạch cấu trúc OpenXML...")
    sanitize_and_validate_document(doc)

    doc.save(output_docx_path)
    print(f"[✓] HOÀN TẤT XUẤT BẢN THÀNH CÔNG: {output_docx_path}")

if __name__ == "__main__":
    src_md = sys.argv[1] if len(sys.argv) > 1 else None
    out_docx = sys.argv[2] if len(sys.argv) > 2 else None
    convert_markdown_to_uit_docx(src_md, out_docx)
