# Upload lên Kaggle

Dựng lại thư mục staging bất cứ lúc nào:

```bash
python scripts/method2/build_kaggle_upload.py --hf-cache
```

Script tự kiểm tra `decontamination.json` có mặt và SHA-256 mọi artefact khớp
`manifest.json`. Không đạt thì exit khác 0 — đừng upload khi đó.

## Ba dataset

Tách riêng vì vòng đời khác nhau: code đổi mỗi commit, data đổi khi rebuild
pairs, cache model gần như không đổi. Tách ra thì sửa code không phải upload lại
3 GB.

| Dataset | Nội dung | Dung lượng |
|---|---|---|
| `toolcalling-vi-src` | `src/`, `configs/{method2,eval}/`, `notebooks/` | 0.6 MB |
| `toolcalling-vi-data` | `method2/`, `custom_vi/v1/`, `benchmark_vi/test.jsonl` | 222 MB |
| `toolcalling-vi-hf-cache` | `hub/models--BAAI--bge-m3/`, `hub/models--xlm-roberta-base/` | 3.2 GB |

## Hai chỗ dễ sai

**`decontamination.json` nằm trong `.gitignore`** nên không đi theo `git clone`.
Thiếu nó thì Run 0 fail — đúng thiết kế, nhưng dễ quên khi đóng gói bằng tay.
Script kiểm tra tường minh file này.

**Layout cache HuggingFace.** Đúng là `hub/models--BAAI--bge-m3/…`, không phải
`BAAI/bge-m3/…`. `HF_HOME` phải trỏ vào thư mục **chứa** `hub/`; thiếu tầng đó
thì biến bị bỏ qua trong im lặng và model vẫn tải lại từ Hub mỗi session.
Cell 0 **dò theo marker file** thay vì hardcode đường dẫn: Kaggle mount thành
`/kaggle/input/datasets/<user>/<ds>/<ds>/…`, và số tầng đổi theo cách upload.
Đã kiểm tra logic dò trên 4 layout (phẳng, lồng 1 tầng, namespace, namespace +
lồng) — đều tìm ra. Không tìm thấy thì notebook dừng ngay kèm marker đã thử.

Thứ tự trong notebook quan trọng: **Cell 0 set `HF_HOME` trước mọi import
`transformers`**. Import trước rồi mới set thì thư viện đã chốt cache mặc định.

## Upload

Qua giao diện web: New Dataset → kéo thả từng thư mục con của `kaggle_upload/`.

Qua CLI:

```bash
pip install kaggle          # cần ~/.kaggle/kaggle.json
cd kaggle_upload/toolcalling-vi-src
kaggle datasets init -p .
# sửa dataset-metadata.json: đặt "title" và "id" = "<username>/toolcalling-vi-src"
kaggle datasets create -p . --dir-mode zip
```

Lặp lại cho hai thư mục còn lại. Lần sau cập nhật thì dùng:

```bash
kaggle datasets version -p . -m "rebuild pairs sau khi decontaminate"
```

`--dir-mode zip` giữ nguyên cấu trúc thư mục con; thiếu cờ này Kaggle sẽ trải
phẳng và notebook không tìm thấy `method2/…`.

## Sau khi upload

Trong Notebook Editor: sidebar phải → **Input** → **Add Input** → tìm và Add
lần lượt `toolcalling-vi-src`, `toolcalling-vi-data`, `toolcalling-vi-hf-cache`.
Bật GPU T4, rồi chạy theo thứ tự:

1. `method2_kaggle_run0_preflight.ipynb` — không train, chỉ kiểm tra cổng.
   Cell 0 dò dataset, Cell 2 copy sang `/kaggle/working`, Cell 3 kiểm tra bản
   copy, rồi mới tới preflight. Ba bước tách riêng để hỏng ở đâu biết ngay.
2. `method2_kaggle_biencoder.ipynb` — dừng sau phần smoke + resume, review
   `smoke_run01/train_report.json` trước khi chạy full.

## Nếu tải model lỗi

`ConnectionError: … xet-read-token … 404` là do backend `xet` của
huggingface_hub. Script đã tự đặt `HF_HUB_DISABLE_XET=1`; nếu tải tay thì
export biến đó trước.
