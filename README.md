# ChatTogether with Neo4j Memory and Lightweight RAG

A multi-character conversation playground built with Streamlit, Neo4j, and Google Gemini. The app lets several NPCs talk to each other (and optionally the player) while:
- Persisting conversations and per-role recent context in Neo4j
- Retrieving relevant long‑term memories via vector search
- Allowing the model to write back new memories when important facts emerge

This README documents the app implemented in `3.0_multi_character_chatbot_ChatTogether_WithRag.py`.

---

## Highlights
- Multi‑party chat: any number of NPCs plus the protagonist (you)
- Weighted speaker selection: each participant has a “desire to speak” slider
- Per‑role RecentBuffer: short‑term context tracked per character
- Long‑term Memory nodes with vector embeddings (text-embedding-004)
- Lightweight RAG: last turns → embeddings → top‑K memories → injected into prompts
- Memory write‑back: the model can emit `[[MEMORY]] ... [[/MEMORY]]` to store new facts
- Robust graph schema with relationships linking conversations, characters, and memories
- Streamlit UI with single‑step, multi‑step, reroll speaker, clear history, and cleanup tools

---

## How it works

### Graph schema (Neo4j)
Nodes:
- `(:Character {name, persona, worldview, desire})`
- `(:RecentBuffer {id, lines_json})` — most recent turns per role, capped by size
- `(:Conversation {id, title, history_json, created_at, updated_at})`
- `(:Memory {content, embedding})` — normalized textual memory with optional vector

Relationships:
- `(Character)-[:HAS_RECENT_BUFFER]->(RecentBuffer)`
- `(Character)-[:PARTICIPATED_IN {role:"npc"|"protagonist"}]->(Conversation)`
- `(Character)-[:AWARE_OF]->(Memory)` — which memories a role “knows”
- `(Conversation)-[:MENTIONED_MEMORY]->(Memory)` — memory referenced in a conversation

Vector index (optional but recommended):
- App attempts to create `VECTOR INDEX mem_vec` on `Memory.embedding` once the embedding dimension is known. If unavailable, memory retrieval gracefully degrades to “off”.

### Retrieval pipeline (Lightweight RAG)
1. Take up to the last 3 conversation turns (recent history)
2. Format each turn into a query string and embed with `text-embedding-004`
3. Query Neo4j’s vector index for the top‑K memories visible to the current speaker
4. Prepend the top memories into the prompt as a concise bullet list “Things you already know”

### Memory write‑back
- The model may output blocks like: `[[MEMORY]] ... [[/MEMORY]]`
- The app extracts these blocks, normalizes them to the form `Speaker once said: ...`, embeds and upserts them as `(:Memory)`
- It then links `AWARE_OF` from all participants (including the protagonist) and `MENTIONED_MEMORY` from the current `:Conversation`

### Speaker selection
- Each participant has a desire weight (0–1)
- The app randomly selects the next speaker proportionally to desire, avoiding immediate repetition when possible

### Protagonist mode (player)
- If the selected speaker is the protagonist, the model generates a suggested line (with memory assistance)
- You can edit/accept it or reroll a new suggestion; your accepted line is persisted

---

## Requirements
- Python 3.9+
- Neo4j 5.x (with vector index capability recommended)
- Google Cloud project with access to Gemini models and `text-embedding-004`
- Credentials via Application Default Credentials or `GOOGLE_APPLICATION_CREDENTIALS`

Python packages (install as needed):
- `streamlit`, `neo4j`, `python-dotenv`, `google-genai` (google’s GenAI SDK), plus any dependencies present in `requirements.txt`

Note: The app uses `from google import genai` with `genai.Client(vertexai=True, ...)`. Ensure the `google-genai` package is installed and authenticated.

---

## Configuration (.env)
Create a `.env` file in the project root with your settings:

```
# Neo4j
NEO4J_URI=neo4j://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
# Optional database name (leave empty for default)
NEO4J_DATABASE=

# Google GenAI / Gemini
GEMINI_PROJECT_ID=your_gcp_project_id
GEMINI_LOCATION=us-central1
# Model used for generation
GEMINI_MODEL_STD=gemini-2.0-flash-lite
# Embedding model used for memory
EMBED_MODEL=text-embedding-004

# One of the following must provide credentials:
# 1) Path to a service account key file
GOOGLE_APPLICATION_CREDENTIALS=path\to\service-account.json
# or 2) Use gcloud ADC: gcloud auth application-default login
```

---

## Run the app
Make sure your Neo4j server is up and `.env` is set correctly.

Windows (cmd.exe):
```
streamlit run d:\NewProjects\LowCostChattingBot\npc-graph-chat\3.0_multi_character_chatbot_ChatTogether_WithRag.py
```

Cross‑platform (adjust path to your workspace):
```
streamlit run ./3.0_multi_character_chatbot_ChatTogether_WithRag.py
```

When the app opens:
1. Add or refresh characters
2. Select NPC participants and optionally a conversation title
3. Configure the protagonist persona/worldview and a topic hint
4. Lock participants to start the conversation
5. Use the control buttons: single step, multi‑step (skip/allow protagonist), reroll speaker, or clear
6. If it’s your turn (protagonist), review/edit the suggested line and send

---

## UI cheatsheet
- Refresh character list: reloads NPCs stored in Neo4j
- Desire sliders: control “how likely” each NPC is to speak next
- Lock participants: fixes the scene, creates a Conversation node on first write
- Generate next line: advances one turn (with memory retrieval)
- Multi‑step (skip protagonist): drives the NPCs for 3 steps
- Multi‑step (allow protagonist): runs up to 3 steps but pauses if it’s your turn
- Reroll speaker: pick a different speaker for the next line
- Clear conversation: empties current conversation history
- Tools → Cleanup: deletes empty conversations from the graph

---

## Data model guarantees
- A single `:Conversation` node stores the entire turn list in `history_json`
- Each role keeps a capped `:RecentBuffer` for faster short‑term context
- Memories are deduplicated by `Memory.content` and optionally embedded
- Retrieval gracefully degrades if vector indexes are unavailable (no crashes)

---

## Troubleshooting
- “missing ScriptRunContext”: run the app via `streamlit run ...`, not `python script.py`
- “Model not found” or permission errors: verify `GEMINI_MODEL_STD`, project/location, and credentials
- No memories retrieved: ensure embeddings are being created; vector index may not exist — the app will continue without RAG
- Neo4j routing errors: the app automatically falls back from `neo4j://` to `bolt://` and informs you in the UI

---

## Safety & privacy
- Keep your service‑account JSON secure and out of version control
- Consider separate databases or labels for dev vs. prod

---

## File
- `3.0_multi_character_chatbot_ChatTogether_WithRag.py` — the Streamlit app implementing everything described above

Enjoy building story‑rich, memory‑aware multi‑character conversations! 🎭
