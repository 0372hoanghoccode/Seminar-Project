import sqlite3
import os
from datetime import datetime

class SentimentDatabase:
    def __init__(self, db_path='sentiments.db'):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        """Khởi tạo database và bảng"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Tạo bảng sentiment_history
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sentiment_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                sentiment TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        conn.commit()
        conn.close()

    def save_sentiment(self, text, sentiment):
        """Lưu kết quả phân loại vào database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO sentiment_history (text, sentiment, timestamp)
                VALUES (?, ?, ?)
            ''', (text, sentiment, datetime.now()))

            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Lỗi lưu database: {e}")
            return False

    def get_history(self, limit=50):
        """Lấy lịch sử phân loại gần đây"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT text, sentiment, timestamp
                FROM sentiment_history
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))

            rows = cursor.fetchall()
            conn.close()

            # Chuyển đổi thành list of dict
            history = []
            for row in rows:
                history.append({
                    'text': row[0],
                    'sentiment': row[1],
                    'timestamp': row[2]
                })

            return history
        except Exception as e:
            print(f"Lỗi đọc database: {e}")
            return []

    def clear_history(self):
        """Xóa toàn bộ lịch sử"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('DELETE FROM sentiment_history')
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Lỗi xóa database: {e}")
            return False

    def get_stats(self):
        """Thống kê cảm xúc"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT sentiment, COUNT(*) as count
                FROM sentiment_history
                GROUP BY sentiment
                ORDER BY count DESC
            ''')

            rows = cursor.fetchall()
            conn.close()

            stats = {}
            for row in rows:
                stats[row[0]] = row[1]

            return stats
        except Exception as e:
            print(f"Lỗi thống kê: {e}")
            return {}
