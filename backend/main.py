from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Union
from datetime import datetime
from pydantic import BaseModel
import os
import logging
import sqlite3
from pathlib import Path
from model_service import model_service

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="NeuroBeton API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Настройка базы данных
DB_PATH = Path("users.db")
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL
        )
    ''')
    
    # Создаем таблицу для истории предсказаний
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            filename TEXT NOT NULL,
            strength_prediction REAL NOT NULL,
            has_cracks BOOLEAN NOT NULL,
            crack_probability REAL NOT NULL,
            concrete_type TEXT NOT NULL,
            type_confidence REAL NOT NULL,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (username) REFERENCES users(username)
        )
    ''')
    conn.commit()
    conn.close()

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# Инициализация базы данных при запуске
init_db()

class User(BaseModel):
    username: str
    password: str

class PredictionResponse(BaseModel):
    strength: float
    cracks: Dict[str, float]
    concrete_type: Dict[str, Union[str, float]]

@app.post("/register")
async def register(user: User):
    # Validate input
    if not user.username or not user.password:
        raise HTTPException(status_code=400, detail="Username and password are required")
    
    if len(user.username) < 3:
        raise HTTPException(status_code=400, detail="Username must be at least 3 characters long")
    
    if len(user.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters long")
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                 (user.username, user.password))
        conn.commit()
        return {"message": "User registered successfully"}
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Username already exists")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")
    finally:
        conn.close()

@app.post("/login")
async def login(user: User):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username = ? AND password = ?",
             (user.username, user.password))
    result = c.fetchone()
    conn.close()
    
    if result:
        return {"message": "Login successful", "username": user.username}
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials")

@app.post("/predict", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...), username: str = None):
    if not username:
        raise HTTPException(status_code=401, detail="Username is required")
        
    # Проверяем существование пользователя
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username = ?", (username,))
    if not c.fetchone():
        conn.close()
        raise HTTPException(status_code=401, detail="User not found")
    conn.close()
    
    # Сохраняем файл
    timestamp = datetime.now().isoformat()
    filename = f"{timestamp}_{file.filename}"
    file_path = UPLOAD_DIR / filename
    
    try:
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
    except Exception as e:
        logger.error(f"Error saving file: {e}")
        raise HTTPException(status_code=500, detail="Error saving file")
    
    try:
        # Получаем предсказания от всех моделей
        strength = float(model_service.predict_strength(file_path))  # Ensure it's a float
        cracks = model_service.predict_cracks(file_path)
        concrete_type = model_service.predict_concrete_type(file_path)
        
        # Сохраняем результаты в базу данных
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''
            INSERT INTO predictions 
            (username, filename, strength_prediction, has_cracks, crack_probability, 
             concrete_type, type_confidence, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            username, filename, strength, cracks['has_cracks'], cracks['crack_probability'],
            concrete_type['concrete_type'], concrete_type['confidence'], timestamp
        ))
        conn.commit()
        conn.close()
        
        return {
            "strength": strength,
            "cracks": {
                "has_cracks": float(cracks['has_cracks']),
                "probability": cracks['crack_probability']
            },
            "concrete_type": {
                "type": concrete_type['concrete_type'],
                "confidence": concrete_type['confidence']
            }
        }
        
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail="Error making prediction")
    finally:
        # Удаляем временный файл
        try:
            os.remove(file_path)
        except:
            pass

@app.get("/history/{username}")
async def get_prediction_history(username: str):
    logger.info(f"Fetching history for user: {username}")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        SELECT filename, strength_prediction, has_cracks, crack_probability,
               concrete_type, type_confidence, timestamp
        FROM predictions
        WHERE username = ?
        ORDER BY timestamp DESC
    ''', (username,))
    results = c.fetchall()
    logger.info(f"Found {len(results)} records in database")
    conn.close()
    
    history = []
    for r in results:
        try:
            logger.info(f"Processing record: {r}")
            # Try to convert strength_prediction to float, handling both string and binary data
            try:
                strength = float(r[1])
                logger.info(f"Successfully converted strength to float: {strength}")
            except (ValueError, TypeError) as e:
                logger.error(f"Error converting strength: {e}, value type: {type(r[1])}, value: {r[1]}")
                # If it's binary data, try to decode it first
                if isinstance(r[1], bytes):
                    try:
                        strength = float(r[1].decode('utf-8'))
                        logger.info(f"Successfully decoded bytes and converted to float: {strength}")
                    except Exception as e:
                        logger.error(f"Error decoding bytes: {e}")
                        strength = 0.0
                else:
                    strength = 0.0  # Default value if conversion fails
            
            history.append({
                "filename": str(r[0]),
                "strength": strength,
                "cracks": {
                    "has_cracks": bool(r[2]),
                    "probability": float(r[3])
                },
                "concrete_type": {
                    "type": str(r[4]),
                    "confidence": float(r[5])
                },
                "timestamp": str(r[6])
            })
            logger.info(f"Successfully processed record")
        except Exception as e:
            logger.error(f"Error processing history record: {e}")
            continue
    
    logger.info(f"Returning {len(history)} processed records")
    return history
