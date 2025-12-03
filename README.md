# 🇻🇳 Vietnamese Sentiment Assistant

> **Trợ lý phân loại cảm xúc tiếng Việt sử dụng Transformer (PhoBERT)**

Ứng dụng phân loại cảm xúc văn bản tiếng Việt thành 3 loại: **POSITIVE** (Tích cực), **NEUTRAL** (Trung lập), **NEGATIVE** (Tiêu cực). Hỗ trợ xử lý văn bản có dấu, không dấu, viết tắt và phát hiện phủ định.

---

## 📋 Thông tin đồ án

| Mục                | Nội dung                                                                 |
| ------------------ | ------------------------------------------------------------------------ |
| **Tên đồ án**      | Trợ lý phân loại cảm xúc tiếng Việt (Vietnamese Sentiment Assistant)     |
| **Mục đích**       | Phân loại cảm xúc (tích cực, trung tính, tiêu cực) từ văn bản tiếng Việt |
| **Ngôn ngữ**       | Python                                                                   |
| **Thư viện chính** | Hugging Face Transformers, PhoBERT, Streamlit, SQLite                    |
| **Độ chính xác**   | 100% trên 10 test cases (vượt yêu cầu 65%)                               |

---

## ✨ Tính năng chính

- 🤖 **Phân loại cảm xúc**: Sử dụng mô hình Hybrid (Rule-based + PhoBERT Transformer)
- 🇻🇳 **Xử lý tiếng Việt**: Hỗ trợ có/không dấu, viết tắt (ko, dc, tks...), phát hiện phủ định
- 💾 **Lưu trữ lịch sử**: Lưu kết quả vào SQLite với bảng `sentiments` (id, text, sentiment, timestamp)
- 🌐 **Giao diện web**: Streamlit đẹp, dễ sử dụng
- 📊 **Thống kê**: Hiển thị số lượng theo từng loại cảm xúc

---

## 🖥️ Yêu cầu hệ thống

- Python 3.8+
- RAM: 4GB+ (cho mô hình Transformer)
- Dung lượng: 2GB+ (cho mô hình và dependencies)
- Kết nối internet (lần đầu để tải mô hình)

---

## 🚀 Cài đặt & Chạy ứng dụng

### Bước 1: Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### Bước 2: Chạy ứng dụng Web (Khuyến nghị)

```bash
cd vietnamese-sentiment-assistant
python -m streamlit run src/main.py
```

Ứng dụng sẽ mở trong trình duyệt tại `http://localhost:8501`

### Bước 3: Chạy Test Cases

```bash
python tests/test_cases.py
```

---

## 📖 Hướng dẫn sử dụng

1. **Nhập câu tiếng Việt** vào ô văn bản (VD: "Hôm nay tôi rất vui")
2. **Nhấn nút** "Phân loại cảm xúc"
3. **Xem kết quả**:
   - 😊 **Tích cực** (màu xanh)
   - 😐 **Trung lập** (màu xanh dương)
   - 😢 **Tiêu cực** (màu đỏ)
4. **Xem lịch sử** phân loại bên dưới

---

## 🧪 Bộ Test Cases (10 câu theo đề bài)

| STT | Đầu vào               | Đầu ra mong đợi |
| --- | --------------------- | --------------- |
| 1   | Hôm nay tôi rất vui   | POSITIVE        |
| 2   | Món ăn này dở quá     | NEGATIVE        |
| 3   | Thời tiết bình thường | NEUTRAL         |
| 4   | Rat vui hom nay       | POSITIVE        |
| 5   | Công việc ổn định     | NEUTRAL         |
| 6   | Phim này hay lắm      | POSITIVE        |
| 7   | Tôi buồn vì thất bại  | NEGATIVE        |
| 8   | Ngày mai đi học       | NEUTRAL         |
| 9   | Cảm ơn bạn rất nhiều  | POSITIVE        |
| 10  | Mệt mỏi quá hôm nay   | NEGATIVE        |

**Kết quả**: ✅ **100% (10/10)** - Vượt yêu cầu 65%

---

## 📂 Cấu trúc dự án

```
vietnamese-sentiment-assistant/
├── src/
│   ├── main.py                  # Giao diện Streamlit
│   ├── sentiment_classifier.py  # Logic phân loại cảm xúc (Hybrid)
│   └── database.py              # Quản lý SQLite (bảng sentiments)
├── tests/
│   └── test_cases.py            # 10 test cases theo đề bài
├── requirements.txt             # Dependencies
└── README.md                    # Tài liệu này
```

---

## 🔧 Kiến trúc hệ thống

```
[Đầu vào: Câu tiếng Việt]
        ↓
[Tiền xử lý] → Chuẩn hóa Unicode, mở rộng viết tắt, loại dấu câu
        ↓
[Phát hiện phủ định] → Kiểm tra từ phủ định (không, chưa, chẳng...)
        ↓
    ┌───┴───┐
    ↓       ↓
[Rule-based]  [PhoBERT Transformer]
    ↓       ↓
    └───┬───┘
        ↓
[Kết quả: POSITIVE / NEUTRAL / NEGATIVE]
        ↓
[Lưu SQLite + Hiển thị UI]
```

---

## 🛠️ Công nghệ sử dụng

| Thành phần    | Công nghệ                                          |
| ------------- | -------------------------------------------------- |
| **NLP Model** | PhoBERT (wonrax/phobert-base-vietnamese-sentiment) |
| **Framework** | Hugging Face Transformers                          |
| **Giao diện** | Streamlit                                          |
| **Database**  | SQLite3                                            |
| **Tokenizer** | PyVi (ViTokenizer)                                 |

---

## ⚠️ Troubleshooting

| Lỗi                   | Giải pháp                                              |
| --------------------- | ------------------------------------------------------ |
| Không load được model | Kiểm tra kết nối internet để tải model từ Hugging Face |
| Lỗi dependencies      | Chạy `pip install -r requirements.txt`                 |
| streamlit không nhận  | Chạy `python -m streamlit run src/main.py`             |

---

## 📜 License

Dự án này dành cho mục đích học thuật và nghiên cứu.

---

## 👨‍💻 Tác giả

Đồ án môn học - Seminar Project 2025
