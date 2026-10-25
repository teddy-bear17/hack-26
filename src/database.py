import sqlite3
import pandas as pd
import os
import hashlib
import random

DB_NAME = "data/app.db"

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def init_db():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS companies
                 (id TEXT PRIMARY KEY, name TEXT, inn TEXT, parent_id TEXT)''')
                 
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY, password_hash TEXT, role TEXT, company_id TEXT)''')
    
    c.execute("INSERT OR IGNORE INTO companies VALUES ('C001', 'ООО «Грузоперевозки-063»', '0098524470', NULL)")
    c.execute("INSERT OR IGNORE INTO companies VALUES ('C002', 'ИП Предприниматель-156', '003844671674', 'C001')")
    c.execute("INSERT OR IGNORE INTO users VALUES (?, ?, ?, ?)", ('admin', hash_password('123'), 'Director', 'C001'))
    
    conn.commit()
    conn.close()

def get_companies():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM companies", conn)
    conn.close()
    return df

def register_user(username, password, role):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users VALUES (?, ?, ?, NULL)", (username, hash_password(password), role))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def authenticate_user(username, password):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT password_hash, role, company_id FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    conn.close()
    
    if result and result[0] == hash_password(password):
        return {"role": result[1], "company_id": result[2]}
    return None

def create_company_and_link(username, comp_name, comp_inn):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    new_id = f"C{random.randint(100, 999)}"
    try:
        c.execute("INSERT INTO companies VALUES (?, ?, ?, NULL)", (new_id, comp_name, comp_inn))
        c.execute("UPDATE users SET company_id = ? WHERE username = ?", (new_id, username))
        conn.commit()
        return new_id
    except sqlite3.IntegrityError:
        return None
    finally:
        conn.close()

def link_user_to_company(username, company_id):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT id FROM companies WHERE id = ?", (company_id,))
    if not c.fetchone():
        conn.close()
        return False
    c.execute("UPDATE users SET company_id = ? WHERE username = ?", (company_id, username))
    conn.commit()
    conn.close()
    return True

def import_companies_from_csv(df, username):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    count = 0
    for _, row in df.iterrows():
        comp_id = str(row['company_id']).strip()
        comp_name = str(row['company_name']).strip()
        comp_inn = str(row['company_inn']).strip().zfill(10) 
        try:
            c.execute("INSERT INTO companies VALUES (?, ?, ?, NULL)", (comp_id, comp_name, comp_inn))
            count += 1
        except sqlite3.IntegrityError:
            pass
    if count > 0:
        c.execute("SELECT id FROM companies LIMIT 1")
        first_id = c.fetchone()[0]
        c.execute("UPDATE users SET company_id = ? WHERE username = ?", (first_id, username))
    conn.commit()
    conn.close()
    return count