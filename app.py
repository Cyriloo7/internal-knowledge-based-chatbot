from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import os
from datetime import datetime
from client.src.components.indexer import Indexer
from client.src.components.retriever import Retriever
from client.src.components.graph import RAG_Agent

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-this'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///rag_chatbot.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # user, b-manager, a-manager, admin
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)

class Document(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(200), nullable=False)
    collection_name = db.Column(db.String(100), nullable=False)
    uploaded_by = db.Column(db.Integer, db.ForeignKey('user.id'))
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)

class Conversation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    collection_name = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'collection_name': self.collection_name,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

class ChatHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    conversation_id = db.Column(db.Integer, db.ForeignKey('conversation.id'), nullable=True)
    message = db.Column(db.Text, nullable=False)
    response = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    collection_name = db.Column(db.String(100))

# Role hierarchy for access control
ROLE_HIERARCHY = {
    'user': 1,
    'b-manager': 2,
    'a-manager': 3,
    'admin': 4
}

ROLE_MAX_LEVEL = {
    "user": 1,
    "b-manager": 2,
    "a-manager": 3,
    "admin": 4
}

# Collection levels (ordered)
COLLECTION_LEVELS = [
    "level-1",
    "level-2",
    "level-3",
    "level-4"
]

def get_allowed_collections(role):
    max_level = ROLE_MAX_LEVEL.get(role, 0)
    return COLLECTION_LEVELS[:max_level]

def get_level_number(collection_name):
    return int(collection_name.split("-")[-1])

# Decorators
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user_id = session.get('user_id')

        if not user_id:
            session.clear()
            return redirect(url_for('login'))

        user = db.session.get(User, user_id)
        if not user:
            session.clear()
            return redirect(url_for('login'))

        return f(*args, **kwargs)
    return decorated_function

def role_required(min_role):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'user_id' not in session:
                return redirect(url_for('login'))
            user = User.query.get(session['user_id'])
            if ROLE_HIERARCHY.get(user.role, 0) < ROLE_HIERARCHY.get(min_role, 0):
                return jsonify({'error': 'Insufficient permissions'}), 403
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# Routes
@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.get_json()
        user = User.query.filter_by(username=data['username']).first()
        
        if user and check_password_hash(user.password, data['password']) and user.is_active:
            session['user_id'] = user.id
            session['username'] = user.username
            session['role'] = user.role
            return jsonify({
                'success': True,
                'role': user.role,
                'username': user.username
            })
        return jsonify({'success': False, 'message': 'Invalid credentials'}), 401
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    user = User.query.get(session['user_id'])
    return render_template('dashboard.html', user=user)

@app.route('/chat')
@login_required
def chat():
    user = User.query.get(session['user_id'])
    allowed_collections = get_allowed_collections(user.role)
    conversation_id = request.args.get('conversation_id', type=int)

    documents = Document.query.filter(
        Document.collection_name.in_(allowed_collections)
    ).all()
    
    conversations = Conversation.query.filter_by(user_id=user.id).order_by(
        Conversation.updated_at.desc()
    ).all()
    
    current_conversation = None
    if conversation_id:
        current_conversation = Conversation.query.filter_by(
            id=conversation_id,
            user_id=user.id
        ).first()
    
    if not current_conversation and conversations:
        current_conversation = conversations[0]
        conversation_id = current_conversation.id

    return render_template(
        'chat.html',
        user=user,
        documents=documents,
        collections=allowed_collections,
        conversations=conversations,
        current_conversation_id=conversation_id
    )

@app.route('/upload')
@login_required
def upload_documents():
    user = User.query.get(session['user_id'])
    allowed_collections = get_allowed_collections(user.role)

    documents = Document.query.filter(
        Document.collection_name.in_(allowed_collections)
    ).all()

    return render_template(
        'chat.html',
        user=user,
        documents=documents,
        collections=allowed_collections
    )

@app.route('/admin')
@role_required('admin')
def admin_panel():
    users = User.query.all()
    documents = Document.query.all()
    return render_template('admin.html', users=users, documents=documents)

@app.route('/api/chat', methods=['POST'])
@login_required
def chat_message():
    data = request.get_json()
    message = data.get('message')
    collection_name = data.get('collection')
    conversation_id = data.get('conversation_id')
    is_first_message = data.get('is_first_message', False)

    user = User.query.get(session['user_id'])

    if collection_name not in get_allowed_collections(user.role):
        return jsonify({'error': 'Access denied'}), 403

    # Get or create conversation
    conversation = None
    if conversation_id:
        conversation = Conversation.query.filter_by(
            id=conversation_id,
            user_id=user.id
        ).first()
    
    if not conversation:
        # Create title from first message (max 50 chars)
        title = message[:50] + "..." if len(message) > 50 else message
        conversation = Conversation(
            user_id=user.id,
            title=title,
            collection_name=collection_name
        )
        db.session.add(conversation)
        db.session.commit()

    retriever = Retriever(collection_name=collection_name)
    vectorstore = retriever.get_vector_store()
    
    # Use conversation-specific thread_id
    thread_id = f"conversation_{conversation.id}"
    agent = RAG_Agent(vector_store=vectorstore, thread_id=thread_id)

    from langchain_core.messages import HumanMessage, AIMessage
    
    # Load previous conversation history for context (last 10 exchanges)
    previous_messages = []
    chat_history = ChatHistory.query.filter_by(
        conversation_id=conversation.id
    ).order_by(ChatHistory.timestamp.desc()).limit(20).all()
    
    # Reverse to get chronological order
    chat_history.reverse()
    
    for history_item in chat_history:
        previous_messages.append(HumanMessage(content=history_item.message))
        previous_messages.append(AIMessage(content=history_item.response))
    
    # Add current message
    previous_messages.append(HumanMessage(content=message))
    
    result = agent.langgraph_graph().invoke(
        {"messages": previous_messages},
        config=agent.config
    )

    response = result['messages'][-1].text

    # Save chat history
    db.session.add(ChatHistory(
        user_id=user.id,
        conversation_id=conversation.id,
        message=message,
        response=response,
        collection_name=collection_name
    ))
    
    # Update conversation timestamp
    conversation.updated_at = datetime.utcnow()
    db.session.commit()

    return jsonify({
        'response': response,
        'conversation_id': conversation.id
    })

@app.route('/api/upload', methods=['POST'])
@role_required('b-manager')
def upload_document():
    try:
        file = request.files.get('file')
        collection_name = request.form.get('collection_name')

        if not file or file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Validate file extension
        allowed_extensions = {'.pdf', '.docx', '.doc', '.ppt', '.pptx', '.txt'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            return jsonify({'error': f'Unsupported file format. Allowed formats: {", ".join(allowed_extensions)}'}), 400

        if collection_name not in COLLECTION_LEVELS:
            return jsonify({'error': 'Invalid collection level'}), 400

        # Role-based upload permission
        user_role = session['role']
        if get_level_number(collection_name) > ROLE_MAX_LEVEL[user_role]:
            return jsonify({'error': 'Cannot upload to higher level'}), 403

        # Create level-specific upload directory
        base_upload_folder = 'uploads'
        level_folder = os.path.join(base_upload_folder, collection_name)
        os.makedirs(level_folder, exist_ok=True)

        filename = file.filename
        filepath = os.path.join(level_folder, filename)

        file.save(filepath)

        # Index document
        indexer = Indexer(filepath)
        indexer.index_document(level=collection_name)

        # Save metadata in DB
        doc = Document(
            filename=filename,
            collection_name=collection_name,
            uploaded_by=session['user_id']
        )
        db.session.add(doc)
        db.session.commit()

        return jsonify({'success': True})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/users', methods=['GET', 'POST'])
@role_required('admin')
def manage_users():
    if request.method == 'POST':
        data = request.get_json()
        
        if User.query.filter_by(username=data['username']).first():
            return jsonify({'error': 'Username already exists'}), 400
        
        if User.query.filter_by(email=data['email']).first():
            return jsonify({'error': 'Email already exists'}), 400
        
        user = User(
            username=data['username'],
            email=data['email'],
            password=generate_password_hash(data['password']),
            role=data['role']
        )
        db.session.add(user)
        db.session.commit()
        
        return jsonify({'success': True, 'user_id': user.id})
    
    users = User.query.all()
    return jsonify([{
        'id': u.id,
        'username': u.username,
        'email': u.email,
        'role': u.role,
        'is_active': u.is_active,
        'created_at': u.created_at.isoformat()
    } for u in users])

@app.route('/api/users/<int:user_id>', methods=['PUT', 'DELETE'])
@role_required('admin')
def manage_user(user_id):
    user = User.query.get_or_404(user_id)
    
    if request.method == 'PUT':
        data = request.get_json()
        if 'role' in data:
            user.role = data['role']
        if 'is_active' in data:
            user.is_active = data['is_active']
        db.session.commit()
        return jsonify({'success': True})
    
    if request.method == 'DELETE':
        db.session.delete(user)
        db.session.commit()
        return jsonify({'success': True})

@app.route('/api/chat-history')
@login_required
def get_chat_history():
    conversation_id = request.args.get('conversation_id', type=int)
    user_id = session['user_id']
    
    if conversation_id:
        # Verify conversation belongs to user
        conversation = Conversation.query.filter_by(
            id=conversation_id,
            user_id=user_id
        ).first()
        if not conversation:
            return jsonify({'error': 'Conversation not found'}), 404
        
        history = ChatHistory.query.filter_by(
            conversation_id=conversation_id
        ).order_by(ChatHistory.timestamp.asc()).limit(100).all()
    else:
        # Fallback to old behavior for backward compatibility
        collection_name = request.args.get('collection', None)
        query = ChatHistory.query.filter_by(user_id=user_id)
        if collection_name:
            query = query.filter_by(collection_name=collection_name)
        history = query.order_by(ChatHistory.timestamp.asc()).limit(50).all()
    
    return jsonify([{
        'message': h.message,
        'response': h.response,
        'timestamp': h.timestamp.isoformat(),
        'collection': h.collection_name
    } for h in history])

@app.route('/api/conversations', methods=['GET', 'POST'])
@login_required
def manage_conversations():
    user_id = session['user_id']
    
    if request.method == 'POST':
        # Create new conversation
        data = request.get_json()
        title = data.get('title', 'New Conversation')
        collection_name = data.get('collection_name', 'level-1')
        
        conversation = Conversation(
            user_id=user_id,
            title=title,
            collection_name=collection_name
        )
        db.session.add(conversation)
        db.session.commit()
        
        return jsonify(conversation.to_dict()), 201
    
    # GET: List all conversations
    conversations = Conversation.query.filter_by(user_id=user_id).order_by(
        Conversation.updated_at.desc()
    ).all()
    
    return jsonify([c.to_dict() for c in conversations])

@app.route('/api/conversations/<int:conversation_id>', methods=['PUT', 'DELETE'])
@login_required
def manage_conversation(conversation_id):
    user_id = session['user_id']
    conversation = Conversation.query.filter_by(
        id=conversation_id,
        user_id=user_id
    ).first_or_404()
    
    if request.method == 'PUT':
        # Update conversation title
        data = request.get_json()
        if 'title' in data:
            new_title = data['title'].strip()
            if new_title:
                conversation.title = new_title
                db.session.commit()
        return jsonify(conversation.to_dict())
    
    elif request.method == 'DELETE':
        # Delete conversation and its messages
        ChatHistory.query.filter_by(conversation_id=conversation_id).delete()
        db.session.delete(conversation)
        db.session.commit()
        return jsonify({'success': True})

# Initialize database and create default users
def init_db():
    with app.app_context():
        db.create_all()
        
        # Create default users if they don't exist
        default_users = [
            {'username': 'admin', 'email': 'admin@example.com', 'password': 'admin123', 'role': 'admin'}
        ]
        
        for user_data in default_users:
            if not User.query.filter_by(username=user_data['username']).first():
                user = User(
                    username=user_data['username'],
                    email=user_data['email'],
                    password=generate_password_hash(user_data['password']),
                    role=user_data['role']
                )
                db.session.add(user)
        
        db.session.commit()
        print("Database initialized with default users")

if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)