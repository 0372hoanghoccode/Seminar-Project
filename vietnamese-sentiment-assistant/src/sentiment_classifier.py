from transformers import pipeline
import warnings
import re
import unicodedata
from pyvi import ViTokenizer
warnings.filterwarnings('ignore')

class SentimentClassifier:
    def __init__(self):
        """
        Hybrid classifier: Preprocessing + Rule-based + Transformer Pipeline
        Xử lý: có dấu, không dấu, viết tắt, typo, PHỦ ĐỊNH
        """
        # Load Transformer pipeline (PhoBERT sentiment)
        try:
            print("Đang load pipeline Transformer (PhoBERT)...")
            self.pipeline = pipeline("sentiment-analysis", model="wonrax/phobert-base-vietnamese-sentiment", tokenizer="wonrax/phobert-base-vietnamese-sentiment")
            self.model_name = 'Hybrid: Rule-based + PhoBERT Pipeline + Negation Handling'
            print("✓ Pipeline loaded thành công!")
        except Exception as e:
            print(f"⚠ Không load được pipeline: {e}")
            self.pipeline = None
            self.model_name = 'Rule-based only'

        # Từ phủ định tiếng Việt
        self.negation_words = {
            'không', 'chẳng', 'chả', 'khong', 'chang', 'cha',
            'ko', 'k', 'hok', 'hong', 'hông',
            'chưa', 'chua', 'chưa bao giờ',
            'đâu có', 'đâu', 'đéo', 'éo'  # Từ lóng
        }

        # Từ điển viết tắt tiếng Việt
        self.abbreviations = {
            'k': 'không', 'ko': 'không', 'hok': 'không', 'khong': 'không',
            'dc': 'được', 'đc': 'được', 'duoc': 'được',
            'mn': 'mọi người', 'mng': 'mọi người',
            'tks': 'cảm ơn', 'thanks': 'cảm ơn', 'thank': 'cảm ơn',
            'oke': 'ok', 'okie': 'ok', 'oki': 'ok',
            'vs': 'với', 'ms': 'mới', 'nc': 'nói chuyện',
            'hpy': 'vui', 'happy': 'vui', 'vui vui': 'vui',
            'sad': 'buồn', 'buon': 'buồn',
            'met': 'mệt', 'met moi': 'mệt mỏi', 'metmoi': 'mệt mỏi'
        }

        # Từ điển cảm xúc (có dấu + không dấu)
        self.positive_words = {
            'vui', 'vuỉ', 'hanh phuc', 'hạnh phúc', 'tuyet', 'tuyệt',
            'hay', 'tot', 'tốt', 'dep', 'đẹp', 'yeu', 'yêu', 'thich', 'thích',
            'cam on', 'cảm ơn', 'cam ơn', 'thank', 'thanks', 'tks',
            'ok', 'oke', 'tuyet voi', 'tuyệt vời', 'rat vui', 'rất vui',
            'vui ve', 'vui vẻ', 'hai long', 'hài lòng', 'thanh cong', 'thành công'
        }

        self.negative_words = {
            'buon', 'buồn', 'te', 'tệ', 'do', 'dở', 'toi', 'tồi',
            'that bai', 'thất bại', 'kem', 'chan', 'chán', 'ghet', 'ghét',
            'xau', 'xấu', 'dau', 'đau', 'kho chiu', 'khó chịu',
            'met moi', 'mệt mỏi', 'met', 'mệt', 'kinh khung', 'kinh khủng',
            'tuc', 'tức', 'gian', 'giận', 'that vong', 'thất vọng',
            # Thêm các từ bị thiếu
            'rot', 'rớt', 'truot', 'trượt', 'trut', 'trot',  # Rớt (thi, môn)
            'te hai', 'tệ hại', 'do te', 'dở tệ', 'toi te', 'tồi tệ',
            'phuc vu te', 'phục vụ tệ', 'chat luong kem', 'chất lượng kém',
            'khong tot', 'không tốt', 'khong hai long', 'không hài lòng'
        }

        self.neutral_words = {
            'binh thuong', 'bình thường', 'on dinh', 'ổn định',
            'trung binh', 'trung bình', 'thong thuong', 'thông thường',
            'ngay mai', 'hom nay', 'hôm nay', 'di hoc', 'đi học',
            'di lam', 'đi làm', 'cong viec', 'công việc'
        }

    def remove_accents(self, text):
        """Loại bỏ dấu tiếng Việt để xử lý không dấu"""
        nfd = unicodedata.normalize('NFD', text)
        result = ''.join([c for c in nfd if not unicodedata.combining(c)])
        result = result.replace('đ', 'd').replace('Đ', 'D')
        return result

    def expand_abbreviations(self, text):
        """Mở rộng viết tắt"""
        words = text.split()
        expanded = []
        for word in words:
            word_lower = word.lower()
            if word_lower in self.abbreviations:
                expanded.append(self.abbreviations[word_lower])
            else:
                expanded.append(word)
        return ' '.join(expanded)

    def detect_negation(self, text):
        """
        Phát hiện phủ định trong câu

        Returns:
            bool: True nếu có phủ định trước từ cảm xúc
        """
        text_lower = text.lower()
        words = text_lower.split()

        # Tìm vị trí từ phủ định
        negation_positions = []
        for i, word in enumerate(words):
            if word in self.negation_words:
                negation_positions.append(i)

        if not negation_positions:
            return False

        # Kiểm tra xem có từ cảm xúc sau phủ định không
        # Phủ định có tác dụng trong khoảng 3-4 từ sau nó
        for neg_pos in negation_positions:
            # Check 4 từ sau từ phủ định
            window = words[neg_pos:min(neg_pos + 5, len(words))]

            # Check có từ cảm xúc trong window không
            for word in window:
                word_no_accent = self.remove_accents(word)

                # Check positive words
                for pos_word in self.positive_words:
                    pos_word_no_accent = self.remove_accents(pos_word)
                    if pos_word in word or pos_word_no_accent in word_no_accent:
                        return True  # Có phủ định trước từ tích cực

                # Check negative words
                for neg_word in self.negative_words:
                    neg_word_no_accent = self.remove_accents(neg_word)
                    if neg_word in word or neg_word_no_accent in word_no_accent:
                        # Phủ định + từ tiêu cực = tích cực/trung tính
                        return True

        return False

    def preprocess_text(self, text):
        """Tiền xử lý văn bản toàn diện"""
        text = unicodedata.normalize('NFC', text)
        text = text.lower().strip()
        text = self.expand_abbreviations(text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = ' '.join(text.split())
        return text

    def preprocess_for_transformer(self, text):
        """Chỉ mở rộng viết tắt, giữ nguyên dấu câu cho Transformer"""
        text = unicodedata.normalize('NFC', text)
        text = self.expand_abbreviations(text.lower())
        return text

    def rule_based_classify(self, text):
        """
        Phân loại dựa trên từ điển với xử lý mơ hồ (ambiguity)
        Xử lý cả có dấu và không dấu
        """
        text_lower = text.lower()
        text_no_accent = self.remove_accents(text_lower)

        # Đếm điểm từ cả 2 version (có dấu + không dấu)
        pos_score = 0
        neg_score = 0
        neu_score = 0

        for word in self.positive_words:
            word_no_accent = self.remove_accents(word)
            if word in text_lower or word_no_accent in text_no_accent:
                pos_score += 1

        for word in self.negative_words:
            word_no_accent = self.remove_accents(word)
            if word in text_lower or word_no_accent in text_no_accent:
                neg_score += 1

        for word in self.neutral_words:
            word_no_accent = self.remove_accents(word)
            if word in text_lower or word_no_accent in text_no_accent:
                neu_score += 1

        # === XỬ LÝ MƠ HỒ (CRITICAL FIX) ===
        # Nếu có CẢ từ tích cực VÀ tiêu cực (VD: "vui, rớt môn")
        # → Nhường cho Transformer xử lý (hiểu ngữ cảnh tốt hơn)
        if pos_score > 0 and neg_score > 0:
            return None  # Không quyết định, để Transformer xử lý

        # === QUYẾT ĐỊNH ĐƠN GIẢN ===
        # Ưu tiên từ tiêu cực (vì thường ảnh hưởng mạnh hơn)
        if neg_score > 0:
            return 'NEGATIVE'

        if pos_score > 0:
            return 'POSITIVE'

        if neu_score > 0:
            return 'NEUTRAL'

        return None

    def transformer_classify(self, text):
        """Phân loại bằng PhoBERT pipeline (word-segmented)"""
        if not self.pipeline:
            return None

        try:
            segmented = ViTokenizer.tokenize(text)
            result = self.pipeline(segmented)
            sentiment = result[0]['label']

            # Map abbreviated labels to full labels
            label_mapping = {
                'POS': 'POSITIVE',
                'NEG': 'NEGATIVE',
                'NEU': 'NEUTRAL'
            }
            sentiment = label_mapping.get(sentiment, sentiment)

            return sentiment
        except Exception as e:
            print(f"Error in transformer classify: {e}")
            return None

    def classify(self, text):
        """
        Phân loại cảm xúc từ văn bản tiếng Việt

        Quy trình:
        1. Kiểm tra phủ định
        2. Nếu có phủ định → Dùng TRANSFORMER (ưu tiên)
        3. Nếu không phủ định → Thử Rule-based trước
        4. Fallback: Transformer
        5. Default: NEUTRAL

        Args:
            text (str): Câu tiếng Việt (có/không dấu, viết tắt)

        Returns:
            dict: {"text": str, "sentiment": str}

        Raises:
            ValueError: Nếu câu không hợp lệ
        """
        if not text or len(text.strip()) <= 3:
            raise ValueError("Câu không hợp lệ, thử lại (quá ngắn hoặc rỗng).")

        # Preprocessing
        processed_text_rule = self.preprocess_text(text)
        processed_text_transformer = self.preprocess_for_transformer(text)

        # Bước 1: Kiểm tra phủ định
        has_negation = self.detect_negation(processed_text_rule)

        if has_negation:
            # Ưu tiên Transformer nếu có phủ định (xử lý ngữ cảnh tốt hơn)
            transformer_sentiment = self.transformer_classify(processed_text_transformer)
            if transformer_sentiment:
                return {"text": text, "sentiment": transformer_sentiment}

            # Fallback: Rule-based nhưng cẩn thận với phủ định
            # Nếu detect phủ định mà không có model → trả NEUTRAL an toàn
            return {"text": text, "sentiment": "NEUTRAL"}

        else:
            # Không có phủ định → Rule-based (ưu tiên)
            rule_sentiment = self.rule_based_classify(processed_text_rule)
            if rule_sentiment:
                return {"text": text, "sentiment": rule_sentiment}

            # Fallback: Transformer
            transformer_sentiment = self.transformer_classify(processed_text_transformer)
            if transformer_sentiment:
                return {"text": text, "sentiment": transformer_sentiment}

            # Default
            return {"text": text, "sentiment": "NEUTRAL"}

    def get_model_info(self):
        """Trả về thông tin model"""
        return {
            "model_name": self.model_name,
            "method": "Negation-aware Hybrid (Rule-based + PhoBERT)",
            "features": [
                "Xử lý có/không dấu",
                "Mở rộng viết tắt",
                "Phát hiện phủ định",
                "Ưu tiên Transformer cho phủ định",
                "Fallback strategy"
            ]
        }
