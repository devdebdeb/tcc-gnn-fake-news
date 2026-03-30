import os
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

_db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "truth_analytics.db")
DATABASE_URL = f"sqlite:///{_db_path}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class AnaliseHistory(Base):
    __tablename__ = "historico_analises"

    id = Column(Integer, primary_key=True, index=True)
    url_bsky = Column(String)
    texto_resumo = Column(String)
    tamanho_grafo = Column(Integer)
    pred_upfd = Column(String)
    cert_upfd = Column(Float)
    pred_bsky = Column(String)
    cert_bsky = Column(Float)
    heuristica_final = Column(Float)
    data_consulta = Column(DateTime, default=datetime.datetime.utcnow)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
