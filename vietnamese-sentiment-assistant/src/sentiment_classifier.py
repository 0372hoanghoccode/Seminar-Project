from transformers import RobertaForSequenceClassification, AutoTokenizer
import torch
import warnings
import re
import unicodedata
from pyvi import ViTokenizer
warnings.filterwarnings('ignore')

class SentimentClassifier:
    def __init__(self):
        """
        Hybrid classifier: Preprocessing + Rule-based + Transformer
        Xử lý: có dấu, không dấu, viết tắt, typo
        """
        # Load Transformer model (PhoBERT sentiment)
        try:
            print("Đang load model Transformer (PhoBERT)...")
            self.model = RobertaForSequenceClassification.from_pretrained("wonrax/phobert-base-vietnamese-sentiment")
            self.tokenizer = AutoTokenizer.from_pretrained("wonrax/phobert-base-vietnamese-sentiment", use_fast=False)
            self.model_name = 'Hybrid: Preprocessing + Rule-based + PhoBERT'
            print("✓ Model loaded thành công!")
        except Exception as e:
            print(f"⚠ Không load được Transformer: {e}")
            self.model = None
            self.tokenizer = None
            self.model_name = 'Rule-based only'

        # Từ điển viết tắt tiếng Việt
        self.abbreviations = {
            # Thông dụng
            'k': 'không', 'ko': 'không', 'hok': 'không', 'khong': 'không',
            'dc': 'được', 'đc': 'được', 'duoc': 'được',
            'mn': 'mọi người', 'mng': 'mọi người',
            'tks': 'cảm ơn', 'thanks': 'cảm ơn', 'thank': 'cảm ơn',
            'oke': 'ok', 'okie': 'ok', 'oki': 'ok',
            'vs': 'với', 'ms': 'mới', 'nc': 'nói chuyện',
            # Cảm xúc
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
            'tuc', 'tức', 'gian', 'giận', 'that vong', 'thất vọng'
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
        # Xử lý các ký tự đặc biệt tiếng Việt
        result = result.replace('đ', 'd').replace('Đ', 'D')
        return result

    def expand_abbreviations(self, text):
        """Mở rộng viết tắt"""
        words = text.split()
        expanded = []
        for word in words:
            word_lower = word.lower()
            # Kiểm tra từ điển viết tắt
            if word_lower in self.abbreviations:
                expanded.append(self.abbreviations[word_lower])
            else:
                expanded.append(word)
        return ' '.join(expanded)

    def preprocess_text(self, text):
        """
        Tiền xử lý văn bản toàn diện
        """
        # 1. Chuẩn hóa Unicode
        text = unicodedata.normalize('NFC', text)

        # 2. Lowercase
        text = text.lower().strip()

        # 3. Mở rộng viết tắt
        text = self.expand_abbreviations(text)

        # 4. Xóa ký tự đặc biệt (giữ lại chữ, số, khoảng trắng)
        text = re.sub(r'[^\w\s]', ' ', text)

        # 5. Xóa khoảng trắng thừa
        text = ' '.join(text.split())

        return text

    def preprocess_for_transformer(self, text):
        """Chỉ mở rộng viết tắt, giữ nguyên dấu câu cho Transformer"""
        text = unicodedata.normalize('NFC', text)
        text = self.expand_abbreviations(text.lower())
        return text

    def rule_based_classify(self, text):
        """
        Phân loại dựa trên từ điển
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

        # Quyết định
        if pos_score > 0 and pos_score >= neg_score:
            return 'POSITIVE'
        elif neg_score > 0 and neg_score > pos_score:
            return 'NEGATIVE'
        elif neu_score > 0:
            return 'NEUTRAL'

        return None

    def transformer_classify(self, text):
        """Phân loại bằng PhoBERT (word-segmented)"""
        if not self.model or not self.tokenizer:
            return None

        try:
            # Word segmentation (required for PhoBERT)
            segmented = ViTokenizer.tokenize(text)

            # Tokenize
            input_ids = torch.tensor([self.tokenizer.encode(segmented)])

            # Predict
            with torch.no_grad():
                out = self.model(input_ids)
                logits = out.logits.softmax(dim=-1).tolist()[0]

            # Map to sentiments: [NEG, POS, NEU]
            labels = ['NEGATIVE', 'POSITIVE', 'NEUTRAL']
            sentiment = labels[logits.index(max(logits))]
            return sentiment
        except Exception as e:
            print(f"Error in transformer classify: {e}")
            return None

    def classify(self, text):
        """
        Phân loại cảm xúc từ văn bản tiếng Việt

        Quy trình:
        1. Preprocessing cho Rule-based (xóa dấu câu,...)
        2. Rule-based (ưu tiên)
        3. Preprocessing cho Transformer (chỉ mở rộng viết tắt)
        4. Transformer (fallback với text đã xử lý viết tắt)
        5. NEUTRAL (default)

        Args:
            text (str): Câu tiếng Việt (có/không dấu, viết tắt)

        Returns:
            dict: {"text": str, "sentiment": str}

        Raises:
            ValueError: Nếu câu không hợp lệ
        """
        if not text or len(text.strip()) <= 3:
            raise ValueError("Câu không hợp lệ, thử lại (quá ngắn hoặc rỗng).")

        # Bước 1: Preprocessing cho Rule-based (xóa dấu câu,...)
        processed_text_rule = self.preprocess_text(text)

        # Bước 2: Rule-based (ưu tiên)
        rule_sentiment = self.rule_based_classify(processed_text_rule)
        if rule_sentiment:
            return {"text": text, "sentiment": rule_sentiment}

        # Bước 3: Preprocessing cho Transformer (chỉ mở rộng viết tắt)
        processed_text_transformer = self.preprocess_for_transformer(text)

        # Bước 4: Transformer (fallback với text đã xử lý viết tắt)
        transformer_sentiment = self.transformer_classify(processed_text_transformer)
        if transformer_sentiment:
            return {"text": text, "sentiment": transformer_sentiment}

        # Bước 5: Default
        return {"text": text, "sentiment": "NEUTRAL"}

    def get_model_info(self):
        """Trả về thông tin model"""
        return {
            "model_name": self.model_name,
            "method": "Preprocessing + Rule-based + Transformer",
            "features": [
                "Xử lý có/không dấu",
                "Mở rộng viết tắt",
                "Fuzzy matching",
                "Fallback Transformer"
            ]
        }
