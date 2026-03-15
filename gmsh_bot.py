#!/usr/bin/env python3
"""GMSH Command Helper Bot — answers questions using the GMSH user guide PDF.
Uses OpenAI Assistants API with file_search for accurate retrieval from the 458-page guide.
"""

import os
import sys
import time
from pathlib import Path
from openai import OpenAI

PDF_PATH = Path(__file__).parent / "gmsh.pdf"
CACHE_FILE = Path(__file__).parent / ".gmsh_bot_cache"  # stores assistant/vector-store IDs

INSTRUCTIONS = """You are an expert GMSH (Gmsh mesh generator) assistant.
You have access to the official GMSH user guide via file search.

When answering questions:
1. Search the guide and cite the relevant section (e.g., "Section 5.1.2").
2. Provide the exact command/option syntax.
3. Give a short practical example when it helps.
4. If multiple approaches exist, briefly mention the alternatives.
5. If the answer is not in the guide, say so clearly.

Be concise and precise — the user is an engineer who uses GMSH commands regularly."""


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


def setup(client: OpenAI, cache: dict) -> tuple[str, str]:
    """Create (or reuse) the vector store + assistant. Returns (assistant_id, vector_store_id)."""

    # Reuse existing resources if available
    vs_id = cache.get("vector_store_id")
    asst_id = cache.get("assistant_id")
    if vs_id and asst_id:
        try:
            client.beta.vector_stores.retrieve(vs_id)
            client.beta.assistants.retrieve(asst_id)
            print(f"Reusing existing assistant (id: {asst_id})")
            return asst_id, vs_id
        except Exception:
            pass  # IDs are stale — recreate below

    # Upload PDF to a new vector store
    print("Creating vector store and uploading GMSH user guide...")
    vs = client.beta.vector_stores.create(name="GMSH User Guide")
    with open(PDF_PATH, "rb") as f:
        client.beta.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vs.id,
            files=[(PDF_PATH.name, f)],
        )
    print(f"  Vector store ready (id: {vs.id})")

    # Create assistant with file_search
    asst = client.beta.assistants.create(
        name="GMSH Helper",
        instructions=INSTRUCTIONS,
        model="gpt-4o",
        tools=[{"type": "file_search"}],
        tool_resources={"file_search": {"vector_store_ids": [vs.id]}},
    )
    print(f"  Assistant created (id: {asst.id})")

    cache["vector_store_id"] = vs.id
    cache["assistant_id"] = asst.id
    save_cache(cache)
    return asst.id, vs.id


def ask(client: OpenAI, assistant_id: str, thread_id: str, question: str) -> str:
    """Add a question to the thread, run it, and stream the response."""
    client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=question,
    )

    print("\nAssistant: ", end="", flush=True)
    full_text = ""
    with client.beta.threads.runs.stream(
        thread_id=thread_id,
        assistant_id=assistant_id,
    ) as stream:
        for delta in stream.text_deltas:
            print(delta, end="", flush=True)
            full_text += delta
    print("\n")
    return full_text


def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)

    if not PDF_PATH.exists():
        print(f"Error: PDF not found at {PDF_PATH}")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    cache = load_cache()

    assistant_id, _ = setup(client, cache)

    # Each session uses a fresh thread (preserves conversation context within a session)
    thread = client.beta.threads.create()

    print("\nGMSH Command Helper Bot")
    print("=" * 40)
    print("Ask any question about GMSH commands.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        try:
            ask(client, assistant_id, thread.id, question)
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
