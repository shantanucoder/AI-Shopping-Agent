import json
import os
import sqlite3
from typing import Optional

import streamlit as st
from google import genai
from google.genai import types

from reviews_api import get_product_rating, get_ratings_for_products
from setup_db import create_database

# Setup DB
DB_PATH = os.path.join(os.path.dirname(__file__), "store.db")
if not os.path.exists(DB_PATH):
    create_database()

# Database Tools
def search_products(query: str = "", max_price: Optional[float] = None, is_organic: Optional[bool] = None) -> str:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    sql = "SELECT id, name, category, price, description, is_organic FROM products WHERE 1=1"
    params = []
    if query and query.strip():
        sql += " AND (name LIKE ? OR description LIKE ? OR category LIKE ?)"
        like = f"%{query.strip()}%"
        params.extend([like, like, like])
    if max_price is not None:
        sql += " AND price <= ?"
        params.append(max_price)
    if is_organic is not None:
        sql += " AND is_organic = ?"
        params.append(1 if is_organic else 0)
    cursor.execute(sql, params)
    rows = cursor.fetchall()
    conn.close()
    return json.dumps([
        {"id": r[0], "name": r[1], "category": r[2], "price": r[3], "description": r[4], "is_organic": bool(r[5])}
        for r in rows
    ])

def get_rating(product_id: int) -> str:
    return json.dumps(get_product_rating(product_id))

def checkout(product_id: int) -> str:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name, price FROM products WHERE id = ?", (product_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return f"Error: product {product_id} not found."
    name, price = row
    cursor.execute("INSERT INTO orders (product_id, product_name, price) VALUES (?, ?, ?)", (product_id, name, price))
    order_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return f"Order #{order_id} confirmed! '{name}' ordered for ${price:.2f}."

tools_map = {"search_products": search_products, "get_rating": get_rating, "checkout": checkout}

# Page Setup
st.set_page_config(page_title="AI Shopping Assistant", page_icon="🛒")
st.title("🛒 AI Shopping Assistant")

api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY")

if not api_key:
    st.error("API Key missing! Add GEMINI_API_KEY in Streamlit Secrets.")
    st.stop()

client = genai.Client(api_key=api_key)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("How can I help you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        contents = [{"role": "user", "parts": [{"text": prompt}]}]
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                tools=[search_products, get_rating, checkout],
                temperature=0,
                system_instruction="You are a helpful shopping assistant.",
            ),
        )

        while response.function_calls:
            for call in response.function_calls:
                tool_name = call.name
                tool_args = call.args
                if tool_name in tools_map:
                    output = tools_map[tool_name](**tool_args)
                    contents.append({"role": "model", "parts": [{"function_call": call}]})
                    contents.append({
                        "role": "user",
                        "parts": [{"function_response": {"name": tool_name, "response": {"result": output}}}],
                    })
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents,
                config=types.GenerateContentConfig(tools=[search_products, get_rating, checkout], temperature=0),
            )

        final_text = response.text or "Done!"
        st.markdown(final_text)

    st.session_state.messages.append({"role": "assistant", "content": final_text})