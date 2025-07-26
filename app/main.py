import json
import numpy as np
import faiss
from datetime import datetime, timedelta
from typing import Optional, List

from fastapi import FastAPI, Request, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr
from sqlalchemy import (
    Column, Integer, String, ForeignKey, DateTime, Text, create_engine
)
from sqlalchemy.orm import sessionmaker, relationship, declarative_base, Session
import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline, BartTokenizerFast

# ---------------------------
# Configuration & constants
# ---------------------------

SECRET_KEY = "a_very_secret_key_for_jwt_token_generation"  # replace with env var in prod
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # token valid for 7 days

DATABASE_URL = "sqlite:///./legal_rag.db"

# ---------------------------
# Database setup - SQLAlchemy
# ---------------------------

Base = declarative_base()
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ---------------------------
# Models
# ---------------------------

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    chats = relationship("ChatSession", back_populates="owner")

class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    messages = relationship("ChatMessage", back_populates="chat_session", cascade="all, delete-orphan")
    owner = relationship("User", back_populates="chats")

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(Integer, primary_key=True, index=True)
    chat_session_id = Column(Integer, ForeignKey("chat_sessions.id"))
    sender = Column(String)  # 'user' or 'bot'
    content = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    chat_session = relationship("ChatSession", back_populates="messages")

# ---------------------------
# Pydantic Schemas
# ---------------------------

class UserCreate(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    user_id: Optional[int] = None

class UserOut(BaseModel):
    id: int
    email: EmailStr

    class Config:
        orm_mode = True

class ChatMessageOut(BaseModel):
    id: int
    sender: str
    content: str
    timestamp: datetime

    class Config:
        orm_mode = True

class ChatSessionOut(BaseModel):
    id: int
    created_at: datetime
    messages: List[ChatMessageOut]

    class Config:
        orm_mode = True

class AskRequest(BaseModel):
    query: str
    chat_session_id: Optional[int] = None  # If None, use/create latest session

# ---------------------------
# Auth utilities
# ---------------------------

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def decode_access_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: int = payload.get("sub")
        if user_id is None:
            return None
        return user_id
    except JWTError:
        return None

def get_user(db: Session, user_id: int):
    return db.query(User).filter(User.id == user_id).first()

def authenticate_user(db: Session, email: str, password: str):
    user = db.query(User).filter(User.email == email).first()
    if user is None:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(lambda: SessionLocal())):
    user_id = decode_access_token(token)
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = get_user(db, user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user

# ---------------------------
# FastAPI app & CORS
# ---------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for dev, restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Load embedding model, FAISS, summarizer (only once)
# ---------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Using device: {device}")

embedder = SentenceTransformer("sentence-transformers/multi-qa-MiniLM-L6-cos-v1", device=device)
index = faiss.read_index("app/faiss_index.index")  # adjust if path different

with open("app/central_chunks.jsonl", "r", encoding="utf-8") as f:
    chunks = [json.loads(line.strip())["text"] for line in f]

summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6",
    device=0 if device == "cuda" else -1,
)

bart_tokenizer = BartTokenizerFast.from_pretrained("sshleifer/distilbart-cnn-12-6")

# ---------------------------
# FAISS search and summarization utils
# ---------------------------

def search_faiss(query: str, k: int = 5):
    query_embedding = embedder.encode([query])
    scores, indices = index.search(np.array(query_embedding).astype("float32"), k)
    top_chunks = []
    for idx in indices[0]:
        if 0 <= idx < len(chunks):
            top_chunks.append(chunks[idx])
    return top_chunks

def split_into_token_chunks(text: str, max_tokens: int = 900):
    sentences = text.split(". ")
    token_chunks = []
    current_chunk = []
    current_token_count = 0

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        tokens = bart_tokenizer.encode(sent, add_special_tokens=False)
        if current_token_count + len(tokens) > max_tokens:
            token_chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_token_count = 0
        current_chunk.append(sent)
        current_token_count += len(tokens)

    if current_chunk:
        token_chunks.append(" ".join(current_chunk))

    return token_chunks

def summarize_chunks(raw_chunks: list, max_input_tokens=900, max_summary_tokens=200):
    combined_text = " ".join(raw_chunks)
    token_chunks = split_into_token_chunks(combined_text, max_tokens=max_input_tokens)
    summaries = []
    for i, chunk in enumerate(token_chunks):
        try:
            summary_text = summarizer(
                chunk,
                max_length=max_summary_tokens,
                min_length=50,
                do_sample=False,
            )[0]["summary_text"]
            summaries.append(summary_text)
        except Exception as e:
            print(f"⚠️ Error summarizing chunk {i}: {e}")
            continue
    return "\n".join(summaries)

# ---------------------------
# Dependency: get DB session
# ---------------------------

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ---------------------------
# API Endpoints
# ---------------------------

@app.on_event("startup")
def startup():
    # Create tables if not exist
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created")

@app.post("/register", status_code=201, response_model=UserOut)
def register(user_create: UserCreate, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == user_create.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(
        email=user_create.email,
        hashed_password=get_password_hash(user_create.password)
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

@app.post("/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=400,
            detail="Incorrect username or password"
        )
    access_token = create_access_token(data={"sub": str(user.id)})
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/me", response_model=UserOut)
def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user

@app.post("/new_chat", response_model=ChatSessionOut)
def create_new_chat(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    new_chat = ChatSession(user_id=current_user.id)
    db.add(new_chat)
    db.commit()
    db.refresh(new_chat)
    return new_chat

@app.get("/history", response_model=List[ChatSessionOut])
def get_chat_history(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    chats = db.query(ChatSession).filter(ChatSession.user_id == current_user.id).order_by(ChatSession.created_at.desc()).all()
    return chats

@app.post("/ask")
def ask(request: AskRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Find or create chat session
    if request.chat_session_id is not None:
        chat_session = db.query(ChatSession).filter(
            ChatSession.id == request.chat_session_id,
            ChatSession.user_id == current_user.id
        ).first()
        if chat_session is None:
            raise HTTPException(status_code=404, detail="Chat session not found")
    else:
        # Use latest session or create new
        chat_session = db.query(ChatSession).filter(ChatSession.user_id == current_user.id).order_by(ChatSession.created_at.desc()).first()
        if chat_session is None:
            chat_session = ChatSession(user_id=current_user.id)
            db.add(chat_session)
            db.commit()
            db.refresh(chat_session)

    # Save user message to DB
    user_msg = ChatMessage(
        chat_session_id=chat_session.id,
        sender="user",
        content=query,
    )
    db.add(user_msg)
    db.commit()

    # Retrieve relevant legal chunks using FAISS
    top_chunks = search_faiss(query, k=5)
    if not top_chunks:
        summary_text = "No relevant information found for your question."
    else:
        summary_text = summarize_chunks(top_chunks)

    # Save bot reply to DB
    bot_msg = ChatMessage(
        chat_session_id=chat_session.id,
        sender="bot",
        content=summary_text,
    )
    db.add(bot_msg)
    db.commit()

    return {
        "summary": summary_text,
        "chat_session_id": chat_session.id,
        "chunks": top_chunks,
    }
