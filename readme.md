# RAG Chatbot - Professional Full-Stack Application

A production-ready RAG (Retrieval-Augmented Generation) chatbot with role-based access control, document indexing, and intelligent search capabilities.

## Features

### Core Functionality
- **AI-Powered Document Search**: Upload PDFs and ask questions with intelligent context retrieval
- **Multi-Modal Processing**: Extracts text, images, captions, and OCR from PDFs
- **Vector Database**: ChromaDB for efficient semantic search
- **LangGraph Integration**: Advanced conversational AI with tool usage

### Security & Access Control
- **Role-Based Authentication**: 4 access levels (User, B-Manager, A-Manager, Admin)
- **Secure Password Hashing**: Werkzeug security for password management
- **Session Management**: Flask sessions for secure user authentication
- **Document-Level Access Control**: Restrict documents based on user roles

### User Interface
- **Professional Design**: Modern, responsive UI with clean aesthetics
- **Real-time Chat**: Interactive conversational interface
- **Admin Dashboard**: Complete user and document management
- **Upload Interface**: Drag-and-drop PDF uploads with progress tracking

## Project Structure

```
project/
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── .env                           # Environment variables (create this)
├── client/
│   └── src/
│       └── components/
│           ├── indexer.py         # Document indexing logic
│           ├── retriever.py       # Vector store retrieval
│           └── graph.py           # LangGraph RAG agent
├── templates/
│   ├── login.html                 # Login page
│   ├── dashboard.html             # User dashboard
│   ├── chat.html                  # Chat interface
│   └── admin.html                 # Admin panel
├── static/
│   └── css/
│       └── style.css              # All styles
├── uploads/                       # PDF uploads (auto-created)
├── chroma_vectorstore/            # Vector database (auto-created)
└── instance/
    └── rag_chatbot.db            # SQLite database (auto-created)
```

## Installation

### Prerequisites
- Python 3.8 or higher
- Tesseract OCR installed on your system
- Google Gemini API key

### Step 1: Install Tesseract OCR

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

**Windows:**
Download and install from: https://github.com/UB-Mannheim/tesseract/wiki

### Step 2: Clone and Setup

```bash
# Create project directory
mkdir rag-chatbot
cd rag-chatbot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_gemini_api_key_here
```

**Get your Google Gemini API key:**
1. Visit https://makersuite.google.com/app/apikey
2. Create a new API key
3. Copy and paste into `.env` file

### Step 4: Create Directory Structure

```bash
# Create necessary directories
mkdir -p client/src/components
mkdir -p templates
mkdir -p static/css
mkdir uploads
mkdir chroma_vectorstore
```

### Step 5: Place Your Files

Copy the provided files into their respective locations:
- `app.py` → project root
- `indexer.py`, `retriever.py`, `graph.py` → `client/src/components/`
- HTML files → `templates/`
- `style.css` → `static/css/`
- `requirements.txt` → project root

### Step 6: Initialize Database

```bash
python app.py
```

This will:
- Create the SQLite database
- Set up all tables
- Create default admin account (username: `admin`, password: `admin123`)

### Step 7: Create Additional Demo Users (Optional)

Run Python shell:
```bash
python
```

```python
from app import app, db, User
from werkzeug.security import generate_password_hash

with app.app_context():
    # B-Manager
    manager_b = User(
        username='manager-b',
        email='manager-b@example.com',
        password=generate_password_hash('pass123'),
        role='b-manager'
    )
    
    # A-Manager
    manager_a = User(
        username='manager-a',
        email='manager-a@example.com',
        password=generate_password_hash('pass123'),
        role='a-manager'
    )
    
    # Regular User
    user = User(
        username='user',
        email='user@example.com',
        password=generate_password_hash('user123'),
        role='user'
    )
    
    db.session.add_all([manager_b, manager_a, user])
    db.session.commit()
    print("Demo users created!")
```

## Running the Application

```bash
python app.py
```

Access the application at: **http://localhost:5000**

## Default Login Credentials

| Username   | Password | Role       |
|------------|----------|------------|
| admin      | admin123 | Admin      |
| manager-a  | pass123  | A-Manager  |
| manager-b  | pass123  | B-Manager  |
| user       | user123  | User       |

## Usage Guide

### 1. Login
- Navigate to http://localhost:5000
- Use one of the demo accounts to login

### 2. Upload Documents (B-Manager+)
- Click "Chat" in the sidebar
- Click "Upload Documents" 
- Select a PDF file
- Enter a collection name (e.g., "technical_docs")
- Set access level
- Click "Upload & Index"

### 3. Chat with Documents
- Select a collection from the dropdown
- Type your question in the chat input
- Press Enter or click Send
- AI will retrieve relevant information and respond

### 4. Admin Functions (Admin Only)
- Navigate to "Admin Panel"
- View all users and documents
- Create new users with specific roles
- Edit user roles and status
- Delete users

## Access Level Hierarchy

1. **User** (Level 1)
   - View basic documents
   - Chat with AI
   - Access personal history

2. **B-Manager** (Level 2)
   - All User permissions
   - Upload documents
   - Access B-level documents

3. **A-Manager** (Level 3)
   - All B-Manager permissions
   - Access A-level documents

4. **Admin** (Level 4)
   - Full system access
   - User management
   - System configuration

## Technology Stack

### Backend
- **Flask**: Web framework
- **SQLAlchemy**: ORM for database
- **LangChain**: LLM orchestration
- **LangGraph**: AI agent workflows
- **ChromaDB**: Vector database
- **Google Gemini**: LLM for chat and vision

### Frontend
- **HTML5**: Structure
- **CSS3**: Styling with modern design
- **Vanilla JavaScript**: Interactivity

### Processing
- **PyMuPDF**: PDF parsing
- **Tesseract**: OCR
- **Pillow**: Image processing
- **Google Gemini Vision**: Image captioning

## API Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/login` | POST | No | User authentication |
| `/logout` | GET | Yes | End session |
| `/dashboard` | GET | Yes | User dashboard |
| `/chat` | GET | Yes | Chat interface |
| `/admin` | GET | Admin | Admin panel |
| `/api/chat` | POST | Yes | Send message |
| `/api/upload` | POST | B-Manager+ | Upload document |
| `/api/users` | GET/POST | Admin | Manage users |
| `/api/users/<id>` | PUT/DELETE | Admin | Edit/delete user |
| `/api/chat-history` | GET | Yes | Get chat history |

## Troubleshooting

### "No module named 'client'"
Make sure you have the correct directory structure:
```
project/
└── client/
    └── src/
        └── components/
```

### Tesseract not found
Install Tesseract and ensure it's in your PATH:
```bash
which tesseract  # Linux/Mac
where tesseract  # Windows
```

### ChromaDB errors
Delete the `chroma_vectorstore` folder and restart:
```bash
rm -rf chroma_vectorstore
python app.py
```

### Import errors
Ensure virtual environment is activated and all packages installed:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

## Security Recommendations

For production deployment:

1. **Change Secret Key**:
   ```python
   app.config['SECRET_KEY'] = 'your-random-secret-key-here'
   ```

2. **Use PostgreSQL** instead of SQLite:
   ```python
   app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://user:pass@localhost/dbname'
   ```

3. **Enable HTTPS** with SSL certificates

4. **Set Strong Passwords** for all accounts

5. **Add Rate Limiting** to prevent abuse:
   ```bash
   pip install Flask-Limiter
   ```

6. **Use Environment Variables** for all sensitive data

## Performance Optimization

- Adjust chunk size in `indexer.py` for larger documents
- Increase `search_kwargs["k"]` for more relevant results
- Use GPU for faster embeddings (if available)
- Implement Redis for session management
- Add caching for frequently asked questions

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License - feel free to use this project for commercial or personal use.

## Support

For issues or questions:
- Check the troubleshooting section
- Review the code comments
- Create an issue on GitHub

## Future Enhancements

- [ ] Multi-document chat sessions
- [ ] Export chat history to PDF
- [ ] Advanced analytics dashboard
- [ ] Real-time collaborative chat
- [ ] Mobile app integration
- [ ] Multi-language support
- [ ] Voice input/output
- [ ] Document version control

---

Built with ❤️ using Flask, LangChain, and Google Gemini
