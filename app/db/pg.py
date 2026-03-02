# app/db/pg.py
import os
import psycopg
from dotenv import load_dotenv

_conn_str = None

def _get_database_url():
    global _conn_str
    if _conn_str:
        return _conn_str

    load_dotenv()  # 确保加载 .env

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("Missing DATABASE_URL")

    _conn_str = database_url
    return _conn_str


def get_conn():
    return psycopg.connect(_get_database_url())