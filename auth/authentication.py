import os
import bcrypt
from sqlchemy import create_engine
POSTGRES_URL = os.getenv("POSTGRES_URL")

create_table = """CREATE TABLE IF NOT EXISTS users (
    user_id VARCHAR(100) PRIMARY KEY,
    username VARCHAR(100) NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

db = create_engine(POSTGRES_URL)
def autheticate():

def create_profile(username: str, password: str)-> dict:

    pass_hash = bycrpt.hashpw(password.encode(), bycrpt.gensalt())
    
    
        

    
    

if __name__ == "__main__":
    autheticate()