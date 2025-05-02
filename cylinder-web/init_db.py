import os
import sqlite3
import datetime

# Define the SQLite database file
DB_FILE = 'cylinder_detector.db'

def init_sqlite_db():
    """Initialize SQLite database with required tables"""
    try:
        # Connect to SQLite database (create it if it doesn't exist)
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Create detections table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id TEXT PRIMARY KEY,
            count INTEGER,
            image_path TEXT,
            original_path TEXT,
            timestamp TEXT,
            models_used TEXT
        )
        ''')
        
        # Create users table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT UNIQUE,
            email TEXT UNIQUE,
            password_hash TEXT,
            created_at TEXT
        )
        ''')
        
        # Create cylinder_records table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS cylinder_records (
            id TEXT PRIMARY KEY,
            detection_id TEXT,
            count INTEGER,
            location TEXT,
            batch_number TEXT,
            operator TEXT,
            notes TEXT,
            timestamp TEXT
        )
        ''')
        
        # Create production_stats table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS production_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            total_count INTEGER,
            average_per_batch REAL,
            min_count INTEGER,
            max_count INTEGER,
            created_at TEXT
        )
        ''')
        
        # Add a sample detection if none exists
        cursor.execute("SELECT COUNT(*) FROM detections")
        count = cursor.fetchone()[0]
        
        if count == 0:
            # Add a sample detection
            sample_id = "sample1"
            current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            cursor.execute(
                "INSERT INTO detections (id, count, image_path, original_path, timestamp, models_used) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    sample_id,
                    1,
                    "static/uploads/sample_result.jpg",
                    "static/uploads/sample.jpg",
                    current_time,
                    '["yolov8n.pt"]'
                )
            )
        
        conn.commit()
        conn.close()
        print("SQLite database initialized successfully!")
        return True
    except Exception as e:
        print(f"Error initializing SQLite database: {e}")
        return False

if __name__ == "__main__":
    init_sqlite_db()
    print(f"Database file created at: {os.path.abspath(DB_FILE)}") 