# benbot

**benbot** is a Flask-based RAG (Retrieval‑Augmented Generation) web application for chatting with your organization’s documents. It supports role-based access, multi-collection document indexing (Level 1–4), conversation history, citations, and an admin panel for managing users/documents/categories.

## Key features

- **Chat with document grounding**: Ask questions, get answers with citations.
- **Conversation management**: Create, rename, delete, export conversations.
- **Role-based access control**: `user`, `b-manager`, `a-manager`, `admin` (access levels map to `level-1`..`level-4` collections).
- **Document indexing + retrieval**: Upload & index documents into ChromaDB; retrieve relevant chunks at query time.
- **Categories**: Create/manage categories and assign documents; filter documents by **collection** and **category** (Admin panel).
- **Modern UI**: Responsive sidebar with collapse/expand; mobile overlay behavior; PWA manifest.

## Project structure (high level)

```
.
├── app.py                      # Main Flask app (routes, models, auth)
├── requirements.txt
├── client/src/components/       # Indexer/Retriever/RAG graph
├── templates/                   # login/dashboard/chat/admin pages
├── static/                      # CSS, icons, manifest, service worker
├── uploads/                     # Uploaded docs (auto-created)
├── chroma_vectorstore/          # ChromaDB data (auto-created)
└── instance/                    # SQLite DB may be stored here by Flask
```

## Requirements

- **Python**: 3.8+ (recommended 3.10+)
- **Tesseract OCR** (optional but recommended for OCR workflows)
- **API key for at least one model provider**:
  - Gemini: `GOOGLE_API_KEY`
  - OpenAI: `OPENAI_API_KEY`
  - Anthropic: `ANTHROPIC_API_KEY`

## Setup

### 1) Create a virtual environment and install dependencies

```bash
python -m venv venv
venv\Scripts\activate   # Windows PowerShell
# source venv/bin/activate  # macOS/Linux

pip install -r requirements.txt
```

### 2) Configure `.env`

Create a `.env` file in the project root:

```env
# At least one of these is required
GOOGLE_API_KEY=...
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...

# Optional
CDN_DOMAIN=
USE_CDN=false
SESSION_COOKIE_SECURE=false
```

### 3) Run benbot

```bash
python app.py
```

Open: `http://localhost:5000`

## Default admin account

When first run, the app initializes the DB and creates:

- **Username**: `admin`
- **Password**: `admin123`

## How to use

### Chat

- Go to **Chat**
- Choose a collection (**Level 1–4**)
- Ask questions; benbot retrieves relevant context and answers
- Manage conversations from the **Conversations** panel:
  - **Rename** (rename icon / double click)
  - **Delete**
  - **Export**

### Upload documents (B-Manager+)

- From **Chat** → **Upload Documents**
- Select:
  - **Collection level** (`level-1`..`level-4`)
  - **Category** (optional)
- Upload & index

### Admin panel

- Manage **Users**, **Documents**, **Conversations**, **Chat history**
- Documents tab includes:
  - Search
  - Filter by **Collection**
  - Filter by **Category** (including **Uncategorized**)

## Notes

- **SQLite DB file**: the configured URI is `sqlite:///rag_chatbot.db` (file may appear in the project root or under `instance/` depending on Flask/working directory).
- **ChromaDB data**: stored under `chroma_vectorstore/` (per collection level).
- **Collections**: role → allowed collections are enforced in backend routes.

## API endpoints (high level)

| Endpoint                         |     Method |             Auth | Description                                       |
| -------------------------------- | ---------: | ---------------: | ------------------------------------------------- |
| `/login`                         |   GET/POST |               No | Login page + auth                                 |
| `/logout`                        |        GET |              Yes | Logout                                            |
| `/dashboard`                     |        GET |              Yes | Dashboard                                         |
| `/chat`                          |        GET |              Yes | Chat UI                                           |
| `/admin`                         |        GET |            Admin | Admin UI                                          |
| `/api/chat`                      |       POST |              Yes | Send a chat message (creates/uses a conversation) |
| `/api/chat-history`              |        GET |              Yes | Load messages for a conversation                  |
| `/api/conversations`             |   GET/POST |              Yes | List/create conversations                         |
| `/api/conversations/<id>`        | PUT/DELETE |              Yes | Rename/delete a conversation                      |
| `/api/conversations/<id>/export` |        GET |              Yes | Export conversation (markdown/txt)                |
| `/api/settings/ai`               |    GET/PUT |              Yes | Get/update AI settings                            |
| `/api/categories`                |   GET/POST | Yes / B-Manager+ | List/create categories                            |
| `/api/categories/<id>`           | PUT/DELETE |       B-Manager+ | Update/delete categories                          |
| `/api/documents`                 |        GET |              Yes | List documents with optional filters              |

## Troubleshooting

### Tesseract not found (Windows)

- Install from the UB Mannheim build and ensure it’s on PATH: `https://github.com/UB-Mannheim/tesseract/wiki`
- Verify:

```bash
where tesseract
```

### ChromaDB / indexing issues

- Delete `chroma_vectorstore/` and restart.
- You will need to re-upload/re-index documents.

### Import/dependency issues

- Ensure venv is activated and dependencies installed:

```bash
pip install -r requirements.txt
```

## Security notes (recommended for production)

- Change `SECRET_KEY`
- Use HTTPS
- Use strong passwords / disable default admin password
- Prefer Postgres over SQLite
