import sys

MERMAID_FIG_21 = r"""```mermaid
graph TD
    classDef io fill:#e1f5fe,stroke:#0288d1,stroke-width:2px,color:#01579b;
    classDef model fill:#e8f5e9,stroke:#388e3c,stroke-width:2px,color:#1b5e20;
    classDef inter fill:#fff8e1,stroke:#ffa000,stroke-width:2px,color:#e65100;
    classDef head fill:#ede7f6,stroke:#512da8,stroke-width:2px,color:#311b92;

    Query["Câu truy vấn của người dùng (User Query)"]:::io
    ToolPool["Kho danh mục công cụ (Tool Pool: N APIs)"]:::io

    subgraph Stage1 ["Giai đoạn 1: Lựa chọn công cụ (Tool Retrieval)"]
        BiEncoder["Mô hình Bi-Encoder (BGE-M3 + LoRA)"]:::model
        CosineSim["Xếp hạng tương đồng Cosine & Lọc ngưỡng tau, delta"]:::inter
    end

    subgraph Stage2 ["Giai đoạn 2: Trích xuất tham số theo lược đồ (Cross-Encoder Extraction)"]
        CrossEncoder["Mô hình Cross-Encoder (XLM-RoBERTa Backbone)"]:::model
        subgraph HierarchicalHeads ["Các đầu phân loại phân cấp (Hierarchical Heads)"]
            HasValue["has_value Head (Phân loại nhị phân Có/Không)"]:::head
            SpanHead["span_head (Dự đoán vị trí Start / End)"]:::head
            EnumHead["enum_head (Phân loại danh mục tĩnh)"]:::head
            BoolHead["bool_head (Phân loại giá trị True / False)"]:::head
        end
        Normalizer["Module Chuẩn hóa Giá trị (Vietnamese Value Normalizer)"]:::inter
    end

    Output["Lệnh gọi hàm hoàn chỉnh (JSON Function Call)"]:::io

    Query --> BiEncoder
    ToolPool --> BiEncoder
    BiEncoder --> CosineSim
    CosineSim -->|"Top-K công cụ ứng viên"| CrossEncoder
    Query -->|"Ngữ cảnh truy vấn (Context)"| CrossEncoder

    CrossEncoder --> HasValue
    CrossEncoder --> SpanHead
    CrossEncoder --> EnumHead
    CrossEncoder --> BoolHead

    HasValue -->|"Tham số có mặt"| Normalizer
    SpanHead --> Normalizer
    EnumHead --> Normalizer
    BoolHead --> Normalizer

    Normalizer --> Output
```

*Hình 2.1: Sơ đồ nguyên lý hoạt động của kiến trúc phân tách hai giai đoạn (Bi-Encoder Retrieval kết hợp Cross-Encoder Extraction).*"""

MERMAID_FIG_31 = r"""```mermaid
graph TD
    classDef source fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,color:#0d47a1;
    classDef process fill:#fff3e0,stroke:#ef6c00,stroke-width:2px,color:#e65100;
    classDef dataset fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#1b5e20;

    subgraph Sources ["1. Nguồn dữ liệu thô (Raw Data Sources)"]
        Glaive["Glaive Function Calling v2<br/>(60,734 mẫu đơn/đa lượt)"]:::source
        xLAM["Salesforce xLAM-60k<br/>(60,000 mẫu cấu trúc phức tạp)"]:::source
        CustomRaw["Tập chuyên biệt nghiệp vụ Việt Nam<br/>(8,000 mẫu 10 nhóm lĩnh vực)"]:::source
    end

    subgraph Pipeline ["2. Tiền xử lý & Chuyển ngữ có kiểm soát"]
        Filter["Lọc hội thoại Single-turn<br/>& Khử trùng lặp cú pháp (Dedup)"]:::process
        Translate["Chuyển ngữ với Alibaba Qwen-MT<br/>(Bảo toàn snake_case API & JSON Schema)"]:::process
        QACheck["Kiểm định chất lượng với QA Rules<br/>(Xác thực cú pháp & tính nguyên vẹn)"]:::process
        StratifiedSplit["Phân tầng dữ liệu Train / Val / Test<br/>(Tỷ lệ chuẩn hóa 80 / 10 / 10)"]:::process
    end

    subgraph Targets ["3. Các bộ chuẩn đánh giá hoàn thiện"]
        CoreBenchmark["Canonical Core Benchmark (Song ngữ EN–VI đối sánh 1:1)<br/>(61,615 Train / 7,701 Val / 7,712 Test)"]:::dataset
        CustomSeen["CustomTools-VI (Seen Tools: 20 công cụ)<br/>(5,600 Train / 400 Val / 800 Test)"]:::dataset
        CustomUnseen["CustomTools-VI (Unseen Tools: 20 công cụ Zero-Shot)<br/>(0 Train / 400 Val / 800 Test)"]:::dataset
        StressSet["Tập kiểm thử áp lực Stress Test<br/>(200 anchors × N = 3 đến 1000 distractors)"]:::dataset
    end

    Glaive --> Filter
    xLAM --> Filter
    Filter --> Translate
    Translate --> QACheck
    QACheck --> StratifiedSplit
    CustomRaw --> StratifiedSplit

    StratifiedSplit --> CoreBenchmark
    StratifiedSplit --> CustomSeen
    StratifiedSplit --> CustomUnseen
    CoreBenchmark --> StressSet
```

*Hình 3.1: Quy trình xử lý dữ liệu và xây dựng các bộ tiêu chuẩn đánh giá Benchmark.*"""

MERMAID_FIG_42 = r"""```mermaid
graph LR
    classDef io fill:#e1f5fe,stroke:#0288d1,stroke-width:2px,color:#01579b;
    classDef step fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#4a148c;

    Raw["Chuỗi văn bản thô trích xuất từ span_head"]:::io
    S1["Bước 1: Phân đoạn từ vựng & Ánh xạ từ lóng<br/>('hai củ rưỡi' -> '2.5 triệu', 'năm xị' -> '500k')"]:::step
    S2["Bước 2: Phân tích cú pháp số chữ tiếng Việt<br/>(Giải mã đệ quy hàng đơn vị, chục, trăm, triệu)"]:::step
    S3["Bước 3: Chuẩn hóa Regex có cấu trúc<br/>(Định dạng biển số xe, ngày tháng ISO-8601)"]:::step
    S4["Bước 4: Áp đặt kiểu dữ liệu theo Schema<br/>(Ép kiểu nguyên thủy integer, float, bool, string)"]:::step
    Clean["Giá trị tham số chuẩn hóa (JSON Compliant)"]:::io

    Raw --> S1
    S1 --> S2
    S2 --> S3
    S3 --> S4
    S4 --> Clean
```

*Hình 4.2: Quy trình bốn giai đoạn của module Chuẩn hóa Giá trị tiếng Việt (Vietnamese Value Normalizer).*"""

with open("/home/thinh/project/UIT/tool_calling_with_retrieval_extraction/thesis/khoa_luan_tot_nghiep.md", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update DANH MỤC HÌNH VẼ
old_figures_table = r"""## DANH MỤC HÌNH VẼ

| Ký hiệu hình | Tên hình vẽ | Trang |
| :---: | :--- | :---: |
| **Hình 4.1** | Sơ đồ kiến trúc tổng quan đối chiếu giữa Phương pháp 1 (SLM End-to-End) và Phương pháp 2 (Bi-Encoder + Cross-Encoder) | 28 |
| **Hình 5.1** | So sánh đối đầu toàn diện hiệu năng trên benchmark CustomTools-VI (Seen vs. Unseen) giữa SLM 2B, 4B, Method 2 và các Frontier APIs | 42 |
| **Hình 5.2** | Kết quả Stress Test đối đầu trực diện: Suy giảm độ chính xác, vùng sụp đổ CUDA OOM ($N \ge 500$) và đường cong độ trễ suy luận P50 (ms) log-scale | 49 |
| **Hình 5.3** | Nghiên cứu mở rộng quy mô mô hình (2B vs. 4B): Nâng trần độ chính xác Core Benchmark và triệt tiêu lỗi cú pháp JSON | 54 |"""

new_figures_table = r"""## DANH MỤC HÌNH VẼ

| Ký hiệu hình | Tên hình vẽ | Trang |
| :---: | :--- | :---: |
| **Hình 2.1** | Sơ đồ nguyên lý hoạt động của kiến trúc phân tách hai giai đoạn (Bi-Encoder Retrieval kết hợp Cross-Encoder Extraction) | 16 |
| **Hình 3.1** | Quy trình xử lý dữ liệu và xây dựng các bộ tiêu chuẩn đánh giá Benchmark | 24 |
| **Hình 4.1** | Sơ đồ kiến trúc tổng quan đối chiếu giữa Phương pháp 1 (SLM End-to-End) và Phương pháp 2 (Bi-Encoder + Cross-Encoder) | 28 |
| **Hình 4.2** | Quy trình bốn giai đoạn của module Chuẩn hóa Giá trị tiếng Việt (Vietnamese Value Normalizer) | 38 |
| **Hình 5.1** | So sánh đối đầu toàn diện hiệu năng trên benchmark CustomTools-VI (Seen vs. Unseen) giữa SLM 2B, 4B, Method 2 và các Frontier APIs | 42 |
| **Hình 5.2** | Kết quả Stress Test đối đầu trực diện: Suy giảm độ chính xác, vùng sụp đổ CUDA OOM ($N \ge 500$) và đường cong độ trễ suy luận P50 (ms) log-scale | 49 |
| **Hình 5.3** | Nghiên cứu mở rộng quy mô mô hình (2B vs. 4B): Nâng trần độ chính xác Core Benchmark và triệt tiêu lỗi cú pháp JSON | 54 |"""

if old_figures_table in content:
    content = content.replace(old_figures_table, new_figures_table, 1)
    print("Updated DANH MỤC HÌNH VẼ successfully")
else:
    print("Warning: old_figures_table not found exactly!")

# 2. Replace ASCII art in Section 2.4 with MERMAID_FIG_21
ascii_start = "```\n                                      [ Danh mục Công cụ N APIs ]"
ascii_end = "                                     └─────────────────────────────┘\n```"

p_start = content.find(ascii_start)
p_end = content.find(ascii_end)
if p_start != -1 and p_end != -1:
    p_end_full = p_end + len(ascii_end)
    content = content[:p_start] + MERMAID_FIG_21 + content[p_end_full:]
    print("Replaced ASCII art in Section 2.4 with Mermaid Figure 2.1 successfully")
else:
    print(f"Error finding ASCII art: p_start={p_start}, p_end={p_end}")

# 3. Insert MERMAID_FIG_31 into Chapter 3 right before Table 3.1
sec35_marker = "### 3.5. Quy trình xây dựng tập kiểm thử áp lực (Stress Test Benchmark)"
p_sec35 = content.find(sec35_marker)
table31_marker = "**Bảng 3.1: Thống kê định lượng các bộ dữ liệu trong Benchmark Tool Calling Tiếng Việt**"
p_table31 = content.find(table31_marker)

if p_table31 != -1:
    insert_text = "\n\n" + MERMAID_FIG_31 + "\n\n"
    content = content[:p_table31] + insert_text + content[p_table31:]
    print("Inserted Mermaid Figure 3.1 before Table 3.1 successfully")
else:
    print("Error finding Table 3.1 marker!")

# 4. Insert MERMAID_FIG_42 into Chapter 4 Section 4.4 right before the delimiter
sec44_delimiter = "trước khi được chuyển tiếp tới môi trường thực thi.\n\n---"
p_sec44_del = content.find(sec44_delimiter)
if p_sec44_del != -1:
    idx_insert = p_sec44_del + len("trước khi được chuyển tiếp tới môi trường thực thi.")
    insert_text = "\n\nSơ đồ tuần tự các bước xử lý của module được trực quan hóa tại Hình 4.2.\n\n" + MERMAID_FIG_42
    content = content[:idx_insert] + insert_text + content[idx_insert:]
    print("Inserted Mermaid Figure 4.2 in Section 4.4 successfully")
else:
    print("Error finding Section 4.4 delimiter!")

with open("/home/thinh/project/UIT/tool_calling_with_retrieval_extraction/thesis/khoa_luan_tot_nghiep.md", "w", encoding="utf-8") as f:
    f.write(content)

print("All diagrams inserted and file saved successfully!")
