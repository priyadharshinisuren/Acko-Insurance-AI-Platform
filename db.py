# db.py
import os
import uuid
from datetime import datetime, timedelta
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, Enum, func, ForeignKey, Text
)

from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.dialects.postgresql import UUID

# --- Database Setup ---
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:root@localhost:5432/chatbot")
engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# --- ORM Models ---
class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True)
    question = Column(String, nullable=False)
    answer = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

class Quotation(Base):
    __tablename__ = "quotations"
    id = Column(Integer, primary_key=True)
    vehicle_type = Column(String, nullable=False)
    vehicle_make = Column(String, nullable=False)
    vehicle_model = Column(String, nullable=False)
    manufacturing_year = Column(Integer, nullable=False)
    fuel_type = Column(String, nullable=False)
    city = Column(String, nullable=False)
    vehicle_age = Column(Integer, nullable=False)
    idv = Column(Float, nullable=False)
    ncb_percent = Column(Integer, nullable=False)
    predicted_premium = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)



class ChatLog(Base):
    __tablename__ = "chat_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False)  # FK removed
    intent = Column(Enum("policy_qa", "quotation", "claims", name="intent_type"), nullable=False)
    question = Column(Text, nullable=False)
    retrieved_source = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

# --- Create Tables ---
Base.metadata.create_all(engine)


# --- Logging Functions ---
def log_conversation(question: str, answer: str):
    with SessionLocal() as session:
        conv = Conversation(question=question, answer=answer)
        session.add(conv)
        session.commit()

def save_quote(input_data, prediction: float):
    with SessionLocal() as session:
        quote = Quotation(
            vehicle_type=input_data.get("vehicle_type"),
            vehicle_make=input_data.get("vehicle_make"),
            vehicle_model=input_data.get("vehicle_model"),
            manufacturing_year=int(input_data.get("manufacturing_year")),
            fuel_type=input_data.get("fuel_type"),
            city=input_data.get("city"),
            vehicle_age=datetime.now().year - int(input_data.get("manufacturing_year")),
            idv=float(input_data.get("idv")),
            ncb_percent=int(input_data.get("ncb_percent")),
            predicted_premium=float(prediction),
        )
        session.add(quote)
        session.commit()


def log_chat_interaction(user_id: uuid.UUID, intent: str, question: str, retrieved_source: str = None):
    """Insert a RAG agent interaction log into chat_logs."""
    with SessionLocal() as session:
        log = ChatLog(
            user_id=user_id,
            intent=intent,
            question=question,
            retrieved_source=retrieved_source
        )
        session.add(log)
        session.commit()
        return log.id


# --- Analytics Functions ---
def get_city_distribution(session):
    """Return a dictionary of city -> count of quotations."""
    results = session.query(Quotation.city, func.count(Quotation.id)).group_by(Quotation.city).all()
    return {city: count for city, count in results}


def get_quotation_stats(session):
    total = session.query(Quotation).count()
    avg_premium = session.query(func.avg(Quotation.predicted_premium)).scalar() or 0
    return total, avg_premium

def get_chatlog_stats(session):
    results = session.query(ChatLog.intent, func.count(ChatLog.id)).group_by(ChatLog.intent).all()
    return {intent: count for intent, count in results}
