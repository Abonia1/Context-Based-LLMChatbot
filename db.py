import sqlite3
import json

def create_db() -> None:
  conn = sqlite3.connect('prompt_logs.db')
  cursor = conn.cursor()

  cursor.execute('''CREATE TABLE IF NOT EXISTS prompt_logs (
                      id INTEGER PRIMARY KEY AUTOINCREMENT,
                      prompt TEXT,
                      answer TEXT,
                      citations TEXT
                  )''')
  conn.close()

def log_to_database(prompt: str, answer: str, citations: list) -> None:
    conn = sqlite3.connect('prompt_logs.db')
    c = conn.cursor()

    str_citation = json.dumps(citations)
    c.execute('INSERT INTO prompt_logs (prompt, answer, citations) VALUES (?, ?, ?)', (prompt, answer, str_citation))

    conn.commit()
    conn.close()

def retrieve_logs():
    """
    Retrieve all prompt-answer pairs from the database, starting with the latest entry
    """
    conn = sqlite3.connect('prompt_logs.db')
    cursor = conn.cursor()
    cursor.execute('''SELECT * FROM prompt_logs ORDER BY id DESC LIMIT 50''')
    for row in cursor.fetchall():
        yield row
      
    conn.close()