# Vietnamese Sentiment Assistant

Dự án xây dựng trợ lý phân loại cảm xúc tiếng Việt sử dụng Transformer (PhoBERT). Ứng dụng hỗ trợ phân loại cảm xúc thành 3 loại: POSITIVE, NEUTRAL, NEGATIVE, với khả năng xử lý văn bản tiếng Việt có dấu, không dấu, viết tắt và lỗi chính tả.

## Tính năng chính

- **Phân loại cảm xúc**: Sử dụng mô hình hybrid (rule-based + PhoBERT) để phân loại chính xác.
- **Xử lý tiếng Việt**: Hỗ trợ có/không dấu, viết tắt, từ khóa cảm xúc.
- **Lưu trữ lịch sử**: Lưu kết quả phân loại vào cơ sở dữ liệu SQLite cục bộ.
- **Giao diện web**: Dễ sử dụng với Streamlit.
- **Đóng gói executable**: Chạy độc lập mà không cần cài đặt Python.

## Yêu cầu hệ thống

- Python 3.8+
- RAM: 4GB+ (cho mô hình Transformer)
- Dung lượng: 2GB+ (cho mô hình và dependencies)

## Cài đặt

1. **Cài đặt Python 3.8+** từ [python.org](https://www.python.org/).
2. **Cài đặt dependencies**:
   ```
   pip install -r requirements.txt
   ```
3. **Tải mô hình**: Mô hình PhoBERT sẽ tự động tải khi chạy lần đầu.

## Chạy ứng dụng

### Cách 1: Web App (Khuyến nghị)

Chạy lệnh sau trong thư mục gốc của dự án:

```
streamlit run src/main.py
```

Ứng dụng sẽ mở trong trình duyệt tại `http://localhost:8501`.

### Cách 2: Script Python

Chạy trực tiếp script:

```
python src/main.py
```

### Cách 3: Executable (.exe) - Chạy độc lập

#### Tạo file .exe

1. Cài đặt PyInstaller:

   ```
   pip install pyinstaller
   ```

2. Tạo executable:
   ```
   pyinstaller --onefile --noconsole --add-data "src;src" src/main.py
   ```
   - File `main.exe` sẽ được tạo trong thư mục `dist/`.

#### Chạy .exe

- Double-click vào `main.exe` để chạy ứng dụng.
- Ứng dụng sẽ tự động mở trình duyệt hoặc chạy trong cửa sổ console.

## Sử dụng

1. Nhập câu tiếng Việt vào ô văn bản (ví dụ: "Hôm nay tôi rất vui").
2. Nhấn nút "Phân loại cảm xúc".
3. Xem kết quả phân loại và độ tin cậy.
4. Lịch sử phân loại được hiển thị bên dưới.

## Test

Chạy test để kiểm tra độ chính xác:

```
python tests/test_cases.py
```

Yêu cầu: Độ chính xác >=65% (hiện tại đạt 100%).

## Cấu trúc dự án

```
vietnamese-sentiment-assistant/
├── src/
│   ├── main.py              # Giao diện Streamlit
│   ├── sentiment_classifier.py  # Logic phân loại cảm xúc
│   └── database.py          # Quản lý cơ sở dữ liệu
├── tests/
│   └── test_cases.py        # Test cases
├── requirements.txt         # Dependencies
└── README.md                # Tài liệu này
```

## Lưu ý kỹ thuật

- **Mô hình**: Sử dụng PhoBERT (wonrax/phobert-base-vietnamese-sentiment) cho độ chính xác cao với tiếng Việt.
- **Xử lý văn bản**: Bao gồm chuẩn hóa Unicode, mở rộng viết tắt, loại bỏ dấu câu cho rule-based.
- **Lưu trữ**: SQLite cục bộ, không cần server database.
- **Performance**: Mô hình load một lần khi khởi động, phân loại nhanh (<1s/câu).

## Troubleshooting

- **Lỗi load mô hình**: Kiểm tra kết nối internet để tải mô hình từ Hugging Face.
- **Lỗi dependencies**: Đảm bảo Python 3.8+ và chạy `pip install -r requirements.txt`.
- **Không chạy được .exe**: Đảm bảo antivirus không chặn file, hoặc thử chạy với quyền admin.

## License

Dự án này dành cho mục đích học thuật và nghiên cứu.
