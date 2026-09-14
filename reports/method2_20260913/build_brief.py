"""Create a short, plain-language Method 2 report tied to the experiment plan."""

from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, Table, TableStyle, Spacer, PageBreak, SimpleDocTemplate

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
E = json.loads((ROOT / "reports/method2_20260908/evidence.json").read_text(encoding="utf-8"))
for name, file in [("Arial", "arial.ttf"), ("Arial-Bold", "arialbd.ttf")]:
    pdfmetrics.registerFont(TTFont(name, str(Path("C:/Windows/Fonts") / file)))
pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold")
INK = colors.HexColor("#173A55")
W = A4[0] - 90
ST = {
    "title": ParagraphStyle("title", fontName="Arial-Bold", fontSize=19, leading=24, spaceAfter=10, textColor=INK),
    "sub": ParagraphStyle("sub", fontName="Arial-Bold", fontSize=12, leading=16, spaceBefore=8, spaceAfter=5, textColor=INK),
    "body": ParagraphStyle("body", fontName="Arial", fontSize=10.2, leading=14.3, spaceAfter=7),
    "small": ParagraphStyle("small", fontName="Arial", fontSize=8.5, leading=11.5, spaceAfter=6),
    "cell": ParagraphStyle("cell", fontName="Arial", fontSize=9.4, leading=12.6),
    "head": ParagraphStyle("head", fontName="Arial-Bold", fontSize=9.4, leading=12.6, textColor=colors.white),
}
FLOW = []
MD = ["# Method 2 báo cáo ngắn gọn", "", "Viết lại ngày 13/09/2026 • Kết quả thực nghiệm đã kiểm tra đến 08/09/2026", ""]


def page(title: str, plan: str) -> None:
    if FLOW:
        FLOW.append(PageBreak())
    FLOW.append(Paragraph(escape(title), ST["title"]))
    FLOW.append(Paragraph(escape(plan), ST["small"]))
    MD.extend(["## " + title, "", plan, ""])


def p(text: str, small: bool = False) -> None:
    FLOW.append(Paragraph(escape(text), ST["small" if small else "body"]))
    MD.extend([text, ""])


def h(text: str) -> None:
    FLOW.append(Paragraph(escape(text), ST["sub"]))
    MD.extend(["### " + text, ""])


def table(headers: list[str], rows: list[list[object]], weights: list[float]) -> None:
    data = [[Paragraph(escape(str(x)), ST["head"]) for x in headers]]
    data += [[Paragraph(escape(str(x)), ST["cell"]) for x in row] for row in rows]
    t = Table(data, colWidths=[W*x/sum(weights) for x in weights], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), INK),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F1F5F8")]),
        ("GRID", (0,0), (-1,-1), .4, colors.HexColor("#D9D9D9")),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), 6), ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 6), ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ]))
    FLOW.extend([t, Spacer(1,8)])
    MD.extend(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"]*len(headers)) + " |"])
    MD.extend("| " + " | ".join(str(x) for x in row) + " |" for row in rows)
    MD.append("")


def pct(x: float) -> str:
    return f"{x*100:.2f}%".replace(".", ",")


page("1 Mục tiêu và dữ liệu", "Theo experimental_plan.md: mục 1-3 và 6 • Bản đọc nhanh cho phần Method 2")
p("Kết luận: đã chạy xong các thí nghiệm chính của Method 2 và có số liệu để viết báo cáo. Tuy nhiên, mô hình còn yếu với công cụ chưa gặp; một số mục tiêu chất lượng và phần đánh giá của plan vẫn chưa hoàn thành.")
h("Hệ thống cần làm được việc gì?")
p("Với yêu cầu “Tìm khách sạn ở Huế dưới 1 triệu đồng”, hệ thống phải (1) chọn công cụ tìm khách sạn và (2) điền đúng địa điểm Huế, giá tối đa 1.000.000. Tool là công cụ; arguments là các thông tin cần điền. Nghiên cứu chấm việc chọn và điền này, không thực sự đặt phòng.")
p("Method 2 tách hai việc: Bi-Encoder tìm công cụ phù hợp; Cross-Encoder (CE) lấy thông tin từ câu để điền vào công cụ. Một yêu cầu có thể cần nhiều công cụ (multi-call), ví dụ vừa tìm khách sạn vừa tìm đường.")
h("Dữ liệu dùng để học khác dữ liệu dùng để thi")
table(["Phần dữ liệu", "Định nghĩa và mục đích"], [
    ["Train - tập học", "Mô hình học từ các câu yêu cầu có đáp án mẫu."],
    ["Validation - tập kiểm tra khi phát triển", "Dùng để chọn phiên bản mô hình và cách ra quyết định."],
    ["Test - tập thi cuối", "Chấm phiên bản đã chốt; không dùng kết quả này để chỉnh mô hình rồi chấm lại cùng đề."],
], [1.7,3.6])
p("Seen: công cụ đã xuất hiện khi huấn luyện. Unseen: công cụ được giữ ngoài huấn luyện; đến lúc thử mới đưa mô tả công cụ cho hệ thống. Strict unseen nghĩa là cũng không dùng công cụ giữ lại làm ví dụ gây nhầm khi train. Đã loại 20 công cụ dành cho validation/test unseen khỏi dữ liệu train và tìm ví dụ gây nhầm.")
h("Ba tập thi trong báo cáo")
table(["Tập", "Dùng để kiểm tra", "Số yêu cầu"], [
    ["Benchmark VI", "Tình huống tổng quát, từ Glaive và xLAM được Việt hóa", "10.555; tất cả cần gọi công cụ"],
    ["CustomTools seen", "Tình huống Việt Nam với công cụ đã học", "800 = 400 cần gọi + 400 không cần gọi"],
    ["CustomTools unseen", "Tình huống Việt Nam với công cụ chưa học", "800 = 400 cần gọi + 400 không cần gọi"],
], [1.35,2.5,1.7])
p("Query/sample đều chỉ một yêu cầu thử nghiệm. Positive là yêu cầu cần gọi công cụ; negative là yêu cầu không cần gọi. Tổng ba tập là 12.155 yêu cầu. Mục đích có negative là kiểm tra hệ thống biết khi nào nên dừng, thay vì luôn gọi một công cụ.")

page("2 Đã làm những công việc nào", "Theo plan: mục 6 và thứ tự triển khai ở mục 12")
table(["Công việc và file chạy", "Mục đích", "Đã thực hiện"], [
    ["Chốt dữ liệu", "Tránh mô hình học trước công cụ dành cho bài thi unseen", "Giữ bộ dữ liệu cố định; kiểm tra mã nhận diện file để nối đúng các bước."],
    ["Train Bi-Encoder - notebook 02, 03", "Học chọn công cụ, kể cả khi có công cụ dễ gây nhầm", "Hai lượt train; lượt đầu tìm ví dụ khó cho lượt sau."],
    ["Kiểm tra và sửa CE - notebook 01, 01b", "Học lấy đúng thông tin trong câu", "Sửa nhãn chỉ vị trí thông tin; train lại; chọn bản tốt hơn bằng validation."],
    ["Đánh giá cuối - notebook 04", "Đo hệ thống làm đúng bao nhiêu yêu cầu", "Chạy trên cả ba tập test."],
    ["Stress - notebook 05", "Xem có chậm và nhầm hơn khi nhiều công cụ", "Chạy từ 3 đến 1.000 công cụ."],
    ["Ablation - notebook 06", "Đo riêng lợi ích của bước chuẩn hóa giá trị", "Tắt bước này rồi chấm lại cùng mô hình, cùng test."],
], [1.7,2,2])
h("Chọn công cụ có tốt không?")
p("Recall@k hỏi: công cụ đúng có nằm trong k gợi ý đầu không? Ở đây dùng full Recall@5: một yêu cầu chỉ được tính đúng khi đủ tất cả công cụ cần gọi nằm trong 5 gợi ý đầu. Trên validation, tìm trong toàn bộ 4.464 công cụ đạt 83,96%. Chỉ số này đo khả năng tìm được ứng viên, chưa đảm bảo chọn đúng cuối cùng hoặc điền đúng thông tin.")
h("Lấy thông tin có đạt yêu cầu không?")
p("Gate là mức chất lượng đặt trước để quyết định đã đạt mục tiêu hay chưa. Bảng dưới chấm CE trên validation CustomTools, không phải điểm test cuối.")
table(["Chỉ số và ý nghĩa", "Kết quả", "Mục tiêu"], [
    ["Span EM: lấy đúng đoạn chữ cần điền", "93,64%", "≥80% - đạt"],
    ["Enum accuracy: chọn đúng một giá trị trong danh sách cho phép", "74,49%", "≥90% - chưa đạt"],
    ["Argument EM: điền đúng toàn bộ thông tin của một lần gọi công cụ", "61,96%", "≥70% - chưa đạt"],
], [3.25,1,1.5])
p("Sau sửa nhãn và train lại, Argument EM tăng từ 45,65% lên 61,96% (285/460 lần gọi đúng). Có cải thiện, nhưng chưa đạt gate. “Chạy notebook thành công” chỉ có nghĩa chương trình chạy xong; không có nghĩa mô hình đã đủ tốt.")

page("3 Kết quả cuối và cách hiểu các chỉ số", "Theo plan: mục 6.3, 8 và bảng so sánh/tổng quát hóa ở mục 11")
p("ArgA là tỷ lệ yêu cầu được xử lý đúng trọn vẹn: đúng công cụ, đủ số lần gọi và đúng tất cả thông tin. Chỉ tính trên yêu cầu cần gọi công cụ. Ví dụ 271/400 yêu cầu đúng thì ArgA=67,75%; sai một thông tin vẫn tính cả yêu cầu đó là chưa đúng.")
p("Strict chấm nghiêm ngặt kiểu giá trị và cách biểu diễn. Normalized cho phép quy đổi các khác biệt biểu diễn được quy định trước, như số và khoảng trắng; không tự xem hai từ đồng nghĩa là đúng. Bảng dùng ArgA normalized, đồng nhất giữa các tập.")
on = E['tables']['evaluation']['rows']
table(["Tập test", "Chọn đúng công cụ¹", "Đúng trọn vẹn - ArgA", "Được cho sẵn công cụ đúng²"], [
    [name, pct(r['tool_set_accuracy_positive']), pct(r['normalized_arga']), pct(r['oracle_arga'])]
    for name,r in zip(['Benchmark VI','Custom seen','Custom unseen'],on)
], [1.3,1.3,1.4,1.6])
p("¹ Tool accuracy ở đây yêu cầu chọn đủ, không thừa công cụ; chưa chấm thông tin điền vào. ² Oracle bỏ qua bước chọn công cụ và cho CE đúng công cụ cần dùng, để đo riêng khả năng lấy thông tin.", True)
p("Cách đọc: với công cụ đã học, khoảng 68/100 yêu cầu đúng trọn vẹn; với công cụ chưa học chỉ khoảng 23/100. Ngay khi cho sẵn công cụ đúng, unseen cũng chỉ đạt 36,50%: phần lấy thông tin vẫn là điểm yếu, không chỉ phần chọn công cụ.")
h("Có biết không gọi công cụ khi không cần không?")
p("Negative recall là tỷ lệ yêu cầu không cần công cụ được trả về đúng là không gọi. Kết quả: seen 75% (300/400), unseen 71,75% (287/400). Tức vẫn có 100 và 113 yêu cầu bị gọi công cụ không cần thiết.")
h("Tốc độ và các chỉ số phụ dùng để làm gì?")
table(["Chỉ số", "Định nghĩa và cách hiểu kết quả"], [
    ["Latency P50 / P95", "Thời gian xử lý: 50% / 95% yêu cầu có thời gian không vượt mức này. P50 khoảng 55-58 ms, P95 khoảng 77-93 ms; thấp hơn là nhanh hơn."],
    ["Argument F1", "Cân bằng giữa điền đúng và không bỏ sót các thông tin; cao hơn là tốt hơn. Benchmark/seen/unseen: 61,94% / 91,28% / 67,67%. Khác ArgA vì có thể được điểm một phần."],
    ["JSON / schema validity", "JSON: máy đọc được cấu trúc. Schema: cấu trúc và kiểu dữ liệu đúng yêu cầu công cụ. JSON đạt 100%, nhưng schema chỉ 81,54% / 66,67% / 68,75%; đọc được chưa chắc dùng đúng."],
], [1.5,4.1])
p("Giới hạn: test này chọn trong danh sách công cụ của từng yêu cầu, không phải luôn tìm cả 4.464 công cụ. Chưa đo chi phí USD và chưa thể kết luận rẻ/nhanh hơn các phương pháp chưa được chạy.", True)

page("4 Kiểm tra bổ sung và phần còn lại", "Theo plan: mục 8.4, 9, 10 và điều kiện hoàn tất ở mục 13")
h("Ablation - bước chuẩn hóa có giúp ích không?")
p("Normalizer chuyển thông tin trích được sang giá trị cần dùng, ví dụ “1 triệu” thành 1000000. Ablation nghĩa là tắt riêng một thành phần để đo tác dụng của nó. Notebook 04 bật, notebook 06 tắt normalizer; giữ cùng mô hình và cách chấm normalized.")
table(["ArgA normalized", "Bật chuẩn hóa", "Tắt chuẩn hóa"], [[name,pct(a['normalized_arga']),pct(b['normalized_arga'])] for name,a,b in zip(['Benchmark VI','Custom seen','Custom unseen'],on,E['tables']['ablation']['rows'])],[2.2,1.4,1.4])
p("Kết luận: chuẩn hóa giúp tăng độ đúng trên cả ba tập. Đây là bước xử lý trước khi xuất kết quả, khác với chuẩn hóa của bộ chấm ở trang 3.")
h("Stress test - nhiều công cụ hơn có khó hơn không?")
p("Giữ cùng 200 yêu cầu (100 cần gọi, 100 không cần gọi), thử 6 mức số công cụ: 3, 10, 50, 100, 500, 1.000. Thêm ngẫu nhiên công cụ không phải đáp án để gây nhiễu (random distractors): tổng 1.200 lần thử.")
table(["Kết quả hai đầu dải thử", "3 công cụ", "1.000 công cụ"], [["ArgA - đúng trọn vẹn","81%","54%"],["Thời gian P50","58,16 ms","91,88 ms"]],[2.5,1.3,1.3])
p("Nhiều công cụ làm hệ thống nhầm và chậm hơn. Phần same-domain - thêm công cụ cùng lĩnh vực, khó phân biệt hơn - chưa làm đủ vì thiếu nhóm công cụ đủ lớn. Random chưa thay thế được phần này.")
h("Đã đáp ứng plan đến đâu và cần làm gì tiếp?")
table(["Phần việc", "Kết luận hoặc bước tiếp theo"], [
    ["Scope Method 2", "Đã có train, test, random stress và ablation. Không cần chạy lại notebook 01-06 để báo cáo các số đo hiện tại."],
    ["Chất lượng và phân tích lỗi", "CE còn trượt gate. Cần người đọc kiểm tra 189 trường hợp lỗi đã lấy mẫu, giải thích sai công cụ/sai giá trị/thiếu thông tin; không chỉ dựa vào nhãn tự động."],
    ["Các phép đo còn thiếu", "Bổ sung hoặc ghi rõ thiếu same-domain stress, chi phí và Span F1 (đo mức trùng một phần của đoạn chữ lấy ra)."],
    ["Câu hỏi của toàn đề tài", "Chưa đủ đối chứng để kết luận VI hơn EN, song ngữ tốt hơn, hay CustomTools giúp tăng bao nhiêu. Chưa có so sánh hoàn chỉnh Method 1/OpenAI/Gemini."],
], [1.5,4.1])
p("Có thể dùng kết quả để viết phần thực nghiệm Method 2, với các hạn chế nêu trên. Nguồn: docs/experimental_plan.md; reports/method2_20260908/evidence.json và báo cáo chi tiết cùng thư mục. Bản này chỉ giải thích lại, không train mới hay thay đổi kết quả.",True)


def footer(canvas: object, doc: object) -> None:
    canvas.setFont("Arial",8)
    canvas.setFillColor(INK)
    canvas.drawString(45,24,"Method 2 • Bản giải thích ngắn • 13/09/2026")
    canvas.drawRightString(A4[0]-45,24,str(doc.page))


doc = SimpleDocTemplate(str(HERE/'bao_cao_ngan_gon.pdf'),pagesize=A4,leftMargin=45,rightMargin=45,topMargin=38,bottomMargin=40,title="Method 2 báo cáo ngắn gọn",author="Thinh")
doc.build(FLOW,onFirstPage=footer,onLaterPages=footer)
(HERE/'bao_cao_ngan_gon.md').write_text('\n'.join(MD),encoding='utf-8')
print(HERE)
