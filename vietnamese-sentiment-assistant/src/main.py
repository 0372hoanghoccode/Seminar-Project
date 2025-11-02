import streamlit as st
from sentiment_classifier import SentimentClassifier
from database import SentimentDatabase
import time

# Khởi tạo
@st.cache_resource
def load_classifier():
    return SentimentClassifier()

@st.cache_resource
def load_database():
    return SentimentDatabase()

classifier = load_classifier()
db = load_database()

# UI
st.title("🇻🇳 Vietnamese Sentiment Assistant")
st.markdown("Phân loại cảm xúc từ văn bản tiếng Việt sử dụng Transformer (PhoBERT)")

# Input
text_input = st.text_area(
    "Nhập câu tiếng Việt:",
    placeholder="Ví dụ: Hôm nay tôi rất vui!",
    height=100
)

# Button
if st.button("Phân loại cảm xúc", type="primary"):
    if text_input.strip():
        with st.spinner("Đang phân loại..."):
            try:
                result = classifier.classify(text_input.strip())

                # Hiển thị kết quả
                sentiment = result['sentiment']

                if sentiment == 'POSITIVE':
                    st.success(f"😊 **Tích cực**")
                    color = "green"
                elif sentiment == 'NEGATIVE':
                    st.error(f"😢 **Tiêu cực**")
                    color = "red"
                else:
                    st.info(f"😐 **Trung lập**")
                    color = "blue"

                # Lưu vào database
                db.save_sentiment(result['text'], sentiment)
                st.rerun()

                st.markdown("---")

            except ValueError as e:
                st.error(str(e))
    else:
        st.warning("Vui lòng nhập văn bản!")

# Model info
with st.expander("ℹ️ Thông tin mô hình"):
    info = classifier.get_model_info()
    st.write(f"**Tên mô hình:** {info['model_name']}")
    st.write(f"**Phương pháp:** {info['method']}")
    st.write("**Tính năng:**")
    for feature in info['features']:
        st.write(f"- {feature}")

# Lịch sử
st.markdown("---")
st.subheader("📚 Lịch sử phân loại")

# Stats
stats = db.get_stats()
if stats:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Tích cực", stats.get('POSITIVE', 0))
    with col2:
        st.metric("Tiêu cực", stats.get('NEGATIVE', 0))
    with col3:
        st.metric("Trung lập", stats.get('NEUTRAL', 0))

# History table
history = db.get_history(20)
if history:
    import pandas as pd
    df = pd.DataFrame(history)
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%d/%m/%Y %H:%M')
    df = df.rename(columns={
        'text': 'Văn bản',
        'sentiment': 'Cảm xúc',
        'timestamp': 'Thời gian'
    })
    st.dataframe(df, use_container_width=True)

    if st.button("🗑️ Xóa lịch sử"):
        if db.clear_history():
            st.success("Đã xóa lịch sử!")
            st.rerun()
        else:
            st.error("Lỗi xóa lịch sử!")
else:
    st.info("Chưa có lịch sử phân loại nào.")

# Footer
st.markdown("---")
st.markdown("*Dự án Seminar - Phân loại cảm xúc tiếng Việt*")
