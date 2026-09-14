import json
import os
import sqlite3
from typing import Optional

import streamlit as st
from google import genai
from google.genai import types

from reviews_api import get_product_rating, get_ratings_for_products
from setup_db import create_database

# ---------------------------------------------------------------------------
# Database Setup
# ---------------------------------------------------------------------------
DB_PATH = os.path.join(os.path.dirname(__file__), "store.db")

# Ensure the SQLite DB exists upon deployment
if not os.path.exists(DB_PATH):
    create_database()

# ---------------------------------------------------------------------------
# Database Helper Tools
# ---------------------------------------------------------------------------

def search_products(query: str = "", max_price: Optional[float] = None, is_organic: Optional[bool] = None) -> str:
    """Search products by query keyword, maximum price, or organic status."""
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

    products = [
        {
            "id": row[0],
            "name": row[1],
            "category": row[2],
            "price": row[3],
            "description": row[4],
            "is_organic": bool(row[5]),
        }
        for row in rows
    ]
    return json.dumps(products)

def get_rating(product_id: int) -> str:
    """Get rating and review count for a specific product ID."""
    result = get_product_rating(product_id)
    return json.dumps(result)

def checkout(product_id: int) -> str:
    """Checkout and place an order for a product by product ID."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name, price FROM products WHERE id = ?", (product_id,))
    row = cursor.fetchone()

    if not row:
        conn.close()
        return f"Error: product with ID {product_id} not found."

    name, price = row
    cursor.execute(
        "INSERT INTO orders (product_id, product_name, price) VALUES (?, ?, ?)",
        (product_id, name, price),
    )
    order_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return f"Order #{order_id} confirmed! '{name}' ordered for ${price:.2f}."

tools_map = {
    "search_products": search_products,
    "get_rating": get_rating,
    "checkout": checkout,
}

# ---------------------------------------------------------------------------
# Streamlit UI Setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="AI Shopping Assistant", page_icon="🛒", layout="wide")
st.title("🛒 AI Shopping Assistant")
st.caption("Tell me what you want — I'll search, rate, and order items from our store.")

# Retrieve GEMINI_API_KEY from environment variables or Streamlit Secrets
api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")

if not api_key:
    st.error(
        "⚠️ `GEMINI_API_KEY` is missing.\n\n"
        "Please add `GEMINI_API_KEY` in **Streamlit Cloud -> Settings -> Secrets**."
    )
    st.stop()

client = genai.Client(api_key=api_key)

# Maintain Chat History in Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Process New User Input
if prompt := st.chat_input("e.g., I want organic honey under 15 dollars"):
    # Render user prompt
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching store database..."):
            contents = [{"role": "user", "parts": [{"text": prompt}]}]
            
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=contents,
                config=types.GenerateContentConfig(
                    tools=[search_products, get_rating, checkout],
                    temperature=0,
                    system_instruction=(
                        "You are an AI shopping assistant. Use search_products to find items based on criteria. "
                        "Use get_rating to retrieve customer review scores if requested. "
                        "Only call checkout when the user explicitly confirms they want to place an order."
                    ),
                ),
            )

            # Handle Function/Tool Call Loop
            while response.function_calls:
                for call in response.function_calls:
                    tool_name = call.name
                    tool_args = call.args

                    if tool_name in tools_map:
                        tool_output = tools_map[tool_name](**tool_args)

                        contents.append({"role": "model", "parts": [{"function_call": call}]})
                        contents.append({
                            "role": "user",
                            "parts": [{
                                "function_response": {
                                    "name": tool_name,
                                    "response": {"result": tool_output},
                                }
                            }],
                        })

                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=contents,
                    config=types.GenerateContentConfig(
                        tools=[search_products, get_rating, checkout],
                        temperature=0,
                    ),
                )

            final_text = response.text or "I completed the action."
            st.markdown(final_text)

    st.session_state.messages.append({"role": "assistant", "content": final_text})
