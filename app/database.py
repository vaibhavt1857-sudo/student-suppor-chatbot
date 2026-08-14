import sqlite3

def get_connection():
    conn = sqlite3.connect("data/faqs.db")
    return conn

def get_faq_answer(query):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT answer FROM faqs WHERE question LIKE ?", ('%' + query + '%',))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else "Sorry, I don’t know that yet."
