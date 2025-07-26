import os
import psycopg2
from psycopg2 import pool
import json
import uuid
from datetime import datetime

class DatabaseManager:
    def __init__(self):
        self.database_url = os.environ.get('DATABASE_URL')
        if self.database_url:
            # Use Replit PostgreSQL with connection pooling
            pooled_url = self.database_url.replace('.us-east-2', '-pooler.us-east-2')
            self.connection_pool = pool.SimpleConnectionPool(1, 10, pooled_url)
        else:
            self.connection_pool = None
            print("No DATABASE_URL found, using fallback JSON storage")

    def init_tables(self):
        """Initialize database tables"""
        if not self.connection_pool:
            return False

        try:
            conn = self.connection_pool.getconn()
            cur = conn.cursor()

            # Users table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    user_id VARCHAR(100) UNIQUE NOT NULL,
                    trial_count INTEGER DEFAULT 0,
                    trial_start_date TIMESTAMP,
                    premium BOOLEAN DEFAULT FALSE,
                    premium_expiry TIMESTAMP,
                    total_uses INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Drawings table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS drawings (
                    id SERIAL PRIMARY KEY,
                    user_id VARCHAR(100) NOT NULL,
                    title VARCHAR(200),
                    drawing_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
            """)

            # AI Training Data table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ai_training_data (
                    id SERIAL PRIMARY KEY,
                    user_id VARCHAR(100),
                    prompt TEXT,
                    ai_response TEXT,
                    user_rating INTEGER,
                    correction TEXT,
                    session_id VARCHAR(100),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Collaborative Sessions table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS collaborative_sessions (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(50) UNIQUE NOT NULL,
                    host_user_id VARCHAR(100),
                    participants TEXT[], 
                    canvas_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.commit()
            return True

        except Exception as e:
            print(f"Database initialization error: {e}")
            return False
        finally:
            if conn:
                cur.close()
                self.connection_pool.putconn(conn)

    def get_user_data(self, user_id):
        """Get user data from database"""
        if not self.connection_pool:
            return self._fallback_get_user_data(user_id)

        try:
            conn = self.connection_pool.getconn()
            cur = conn.cursor()

            cur.execute("SELECT * FROM users WHERE user_id = %s", (user_id,))
            result = cur.fetchone()

            if result:
                return {
                    "trial_count": result[2],
                    "trial_start_date": result[3].isoformat() if result[3] else None,
                    "premium": result[4],
                    "premium_expiry": result[5].isoformat() if result[5] else None,
                    "total_uses": result[6]
                }
            else:
                # Create new user
                self.create_user(user_id)
                return {
                    "trial_count": 0,
                    "trial_start_date": None,
                    "premium": False,
                    "premium_expiry": None,
                    "total_uses": 0
                }

        except Exception as e:
            print(f"Database query error: {e}")
            return self._fallback_get_user_data(user_id)
        finally:
            if conn:
                cur.close()
                self.connection_pool.putconn(conn)

    def save_user_data(self, user_id, user_data):
        """Save user data to database"""
        if not self.connection_pool:
            return self._fallback_save_user_data(user_id, user_data)

        try:
            conn = self.connection_pool.getconn()
            cur = conn.cursor()

            cur.execute("""
                INSERT INTO users (user_id, trial_count, trial_start_date, premium, premium_expiry, total_uses)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id) DO UPDATE SET
                    trial_count = EXCLUDED.trial_count,
                    trial_start_date = EXCLUDED.trial_start_date,
                    premium = EXCLUDED.premium,
                    premium_expiry = EXCLUDED.premium_expiry,
                    total_uses = EXCLUDED.total_uses
            """, (
                user_id,
                user_data.get("trial_count", 0),
                user_data.get("trial_start_date"),
                user_data.get("premium", False),
                user_data.get("premium_expiry"),
                user_data.get("total_uses", 0)
            ))

            conn.commit()
            return True

        except Exception as e:
            print(f"Database save error: {e}")
            return self._fallback_save_user_data(user_id, user_data)
        finally:
            if conn:
                cur.close()
                self.connection_pool.putconn(conn)

    def save_ai_training_data(self, user_id, prompt, response, rating, correction=None):
        """Save AI training data"""
        if not self.connection_pool:
            return False

        try:
            conn = self.connection_pool.getconn()
            cur = conn.cursor()

            cur.execute("""
                INSERT INTO ai_training_data (user_id, prompt, ai_response, user_rating, correction, session_id)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (user_id, prompt, response, rating, correction, str(uuid.uuid4())))

            conn.commit()
            return True

        except Exception as e:
            print(f"AI training data save error: {e}")
            return False
        finally:
            if conn:
                cur.close()
                self.connection_pool.putconn(conn)

    def _fallback_get_user_data(self, user_id):
        """Fallback to JSON file storage"""
        try:
            with open("user_data.json", 'r') as f:
                data = json.load(f)
                return data.get(user_id, {
                    "trial_count": 0,
                    "trial_start_date": None,
                    "premium": False,
                    "premium_expiry": None,
                    "total_uses": 0
                })
        except FileNotFoundError:
            return {
                "trial_count": 0,
                "trial_start_date": None,
                "premium": False,
                "premium_expiry": None,
                "total_uses": 0
            }

    def _fallback_save_user_data(self, user_id, user_data):
        """Fallback to JSON file storage"""
        try:
            try:
                with open("user_data.json", 'r') as f:
                    data = json.load(f)
            except FileNotFoundError:
                data = {}

            data[user_id] = user_data

            with open("user_data.json", 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            print(f"Fallback save error: {e}")
            return False

    def create_user(self, user_id):
        """Create new user"""
        if not self.connection_pool:
            return False

        try:
            conn = self.connection_pool.getconn()
            cur = conn.cursor()

            cur.execute("""
                INSERT INTO users (user_id, trial_count, premium, total_uses)
                VALUES (%s, 0, FALSE, 0)
                ON CONFLICT (user_id) DO NOTHING
            """, (user_id,))

            conn.commit()
            return True

        except Exception as e:
            print(f"User creation error: {e}")
            return False
        finally:
            if conn:
                cur.close()
                self.connection_pool.putconn(conn)

# Initialize database manager
db_manager = DatabaseManager()