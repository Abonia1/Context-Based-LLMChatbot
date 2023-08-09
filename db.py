import sqlite3
import json

def create_db() -> None:
  conn = sqlite3.connect('prompt_logs.db')
  cursor = conn.cursor()

  cursor.execute('''CREATE TABLE IF NOT EXISTS prompt_logs (
                      id INTEGER PRIMARY KEY AUTOINCREMENT,
                      prompt TEXT,
                      answer TEXT,
                      citations TEXT,
                      file_path TEXT
                  )''')
  conn.close()

def log_to_database(prompt: str, answer: str, citations: list, file_path: str) -> None:
    conn = sqlite3.connect('prompt_logs.db')
    c = conn.cursor()

    str_citation = json.dumps(citations)
    c.execute('INSERT INTO prompt_logs (prompt, answer, citations, file_path) VALUES (?, ?, ?, ?)', (prompt, answer, str_citation, file_path))

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