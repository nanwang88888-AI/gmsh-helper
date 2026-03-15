#!/usr/bin/env python3
"""GMSH Command Helper Bot — Flask web application."""

import json
import os
import re
import sys
from pathlib import Path

# Must be set before pypdf / cryptography imports
os.environ.setdefault("CRYPTOGRAPHY_OPENSSL_NO_LEGACY", "1")

from dotenv import load_dotenv
from flask import Flask, Response, render_template, request, send_file, stream_with_context
from openai import OpenAI
from pypdf import PdfReader

load_dotenv()

app = Flask(__name__)

PDF_PATH = Path(__file__).parent / "gmsh.pdf"
CACHE_FILE = Path(__file__).parent / ".gmsh_bot_cache"

INSTRUCTIONS = """You are an expert GMSH (Gmsh mesh generator) assistant.
You have access to the official GMSH user guide via file search.

When answering questions:
1. Search the guide and cite the relevant section (e.g., "Section 5.1.2").
2. Provide the exact command/option syntax.
3. Give a short practical example when it helps.
4. If multiple approaches exist, briefly mention the alternatives.
5. If the answer is not in the guide, say so clearly.

Be concise and precise — the user is an engineer who uses GMSH commands regularly."""

# --- OpenAI setup -----------------------------------------------------------

def load_cache() -> dict:
    if CACHE_FILE.exists():
        data = {}
        for line in CACHE_FILE.read_text().splitlines():
            k, _, v = line.partition("=")
            data[k.strip()] = v.strip()
        return data
    return {}


def save_cache(data: dict) -> None:
    CACHE_FILE.write_text("\n".join(f"{k}={v}" for k, v in data.items()))


def get_or_create_assistant(client: OpenAI) -> str:
    cache = load_cache()
    vs_id = cache.get("vector_store_id")
    asst_id = cache.get("assistant_id")

    if vs_id and asst_id:
        try:
            client.beta.vector_stores.retrieve(vs_id)
            client.beta.assistants.retrieve(asst_id)
            return asst_id
        except Exception:
            pass

    print("Creating vector store and uploading GMSH user guide...")
    vs = client.beta.vector_stores.create(name="GMSH User Guide")
    with open(PDF_PATH, "rb") as f:
        client.beta.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vs.id,
            files=[(PDF_PATH.name, f)],
        )

    asst = client.beta.assistants.create(
        name="GMSH Helper",
        instructions=INSTRUCTIONS,
        model="gpt-4o",
        tools=[{"type": "file_search"}],
        tool_resources={"file_search": {"vector_store_ids": [vs.id]}},
    )

    cache["vector_store_id"] = vs.id
    cache["assistant_id"] = asst.id
    save_cache(cache)
    print(f"Assistant ready (id: {asst.id})")
    return asst.id


# Initialize on startup
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("Error: OPENAI_API_KEY environment variable not set.", file=sys.stderr)
    sys.exit(1)

client = OpenAI(api_key=api_key)
ASSISTANT_ID = get_or_create_assistant(client)

# Build page-text index once at startup for fast quote → page lookup
print("Indexing PDF pages...", end=" ", flush=True)
_reader = PdfReader(PDF_PATH)
PDF_PAGES: list[str] = [
    (page.extract_text() or "").lower() for page in _reader.pages
]
print("done")

def find_page(quote: str) -> int | None:
    """Return 1-based page number of the first page containing the quote, or None."""
    needle = re.sub(r"\s+", " ", quote[:120]).strip().lower()
    # Try progressively shorter prefixes so noisy quotes still match
    for length in (len(needle), 80, 50, 30):
        fragment = needle[:length]
        if not fragment:
            break
        for i, text in enumerate(PDF_PAGES):
            if fragment in text:
                return i + 1
    return None

# --- Routes -----------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/pdf")
def serve_pdf():
    """Serve the GMSH PDF so the browser can open it at a specific page."""
    return send_file(PDF_PATH, mimetype="application/pdf")


@app.route("/new_thread", methods=["POST"])
def new_thread():
    """Create a new conversation thread and return its ID."""
    thread = client.beta.threads.create()
    return {"thread_id": thread.id}


@app.route("/chat", methods=["POST"])
def chat():
    """Stream a response for the given question in the given thread."""
    data = request.get_json()
    thread_id: str = data.get("thread_id", "")
    question: str = data.get("question", "").strip()

    if not thread_id or not question:
        return {"error": "Missing thread_id or question"}, 400

    client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=question,
    )

    def generate():
        with client.beta.threads.runs.stream(
            thread_id=thread_id,
            assistant_id=ASSISTANT_ID,
        ) as stream:
            for delta in stream.text_deltas:
                yield f"data: {json.dumps({'text': delta})}\n\n"

        yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    app.run(debug=False, port=5000)
