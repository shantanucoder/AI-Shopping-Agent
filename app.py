import json
import os
import sqlite3
from typing import Optional
import gradio as gr
from google import genai
from google.genai import types

from reviews_api import get_product_rating, get_ratings_for_products
from setup_db import create_database

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
# Agent Chat Handler
# ---------------------------------------------------------------------------

def chat_function(message, history):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return (
            "⚠️ Environment variable `GEMINI_API_KEY` is not set.\n"
            "If using Hugging Face Spaces: Go to Settings > Variables and secrets > Add GEMINI_API_KEY.\n"
            "If running locally: Set GEMINI_API_KEY in your environment or .env file."
        )

    # Initialize client dynamically per call
    client = genai.Client(api_key=api_key)

    # Construct chat context turn
    contents = [{"role": "user", "parts": [{"text": message}]}]
    
    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=contents,
        config=types.GenerateContentConfig(
            tools=[search_products, get_rating, checkout],
            temperature=0,
            system_instruction=(
                "You are an AI shopping assistant. Use search_products to find items based on criteria. "
                "Use get_rating to retrieve customer review scores if requested. "
                "Only call checkout when the user explicitly confirms they want to place an order."
            )
        )
    )

    # Execute tool loop for sequential function calls
    while response.function_calls:
        for call in response.function_calls:
            tool_name = call.name
            tool_args = call.args
            
            if tool_name in tools_map:
                tool_output = tools_map[tool_name](**tool_args)
                
                # Append assistant function request and user execution response
                contents.append({"role": "model", "parts": [{"function_call": call}]})
                contents.append({
                    "role": "user",
                    "parts": [{"function_response": {"name": tool_name, "response": {"result": tool_output}}}]
                })

        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=contents,
            config=types.GenerateContentConfig(
                tools=[search_products, get_rating, checkout],
                temperature=0,
            )
        )

    return response.text

# ---------------------------------------------------------------------------
# Gradio Application Interface
# ---------------------------------------------------------------------------

demo = gr.ChatInterface(
    fn=chat_function,
    title="🛒 AI Shopping Assistant",
    description="Ask for products (e.g., 'I want organic honey under 15 dollar').",
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
