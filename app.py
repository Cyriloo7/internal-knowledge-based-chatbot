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

class ChatHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
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

    documents = Document.query.filter(
        Document.collection_name.in_(allowed_collections)
    ).all()

    return render_template(
        'chat.html',
        user=user,
        documents=documents,
        collections=allowed_collections
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

    user = User.query.get(session['user_id'])

    if collection_name not in get_allowed_collections(user.role):
        return jsonify({'error': 'Access denied'}), 403

    retriever = Retriever(collection_name=collection_name)
    vectorstore = retriever.get_vector_store()
    agent = RAG_Agent(vector_store=vectorstore)

    from langchain_core.messages import HumanMessage
    result = agent.langgraph_graph().invoke(
        {"messages": [HumanMessage(content=message)]},
        config=agent.config
    )

    response = result['messages'][-1].text

    db.session.add(ChatHistory(
        user_id=user.id,
        message=message,
        response=response,
        collection_name=collection_name
    ))
    db.session.commit()

    return jsonify({'response': response})


@app.route('/api/upload', methods=['POST'])
@role_required('b-manager')
def upload_document():
    try:
        file = request.files.get('file')
        collection_name = request.form.get('collection_name')

        if not file or file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if collection_name not in COLLECTION_LEVELS:
            return jsonify({'error': 'Invalid collection level'}), 400

        # Role-based upload permission
        user_role = session['role']
        if get_level_number(collection_name) > ROLE_MAX_LEVEL[user_role]:
            return jsonify({'error': 'Cannot upload to higher level'}), 403

        # ✅ Create level-specific upload directory
        base_upload_folder = 'uploads'
        level_folder = os.path.join(base_upload_folder, collection_name)
        os.makedirs(level_folder, exist_ok=True)

        # Optional: secure filename
        filename = file.filename
        filepath = os.path.join(level_folder, filename)

        file.save(filepath)

        # Index document (pass full path)
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
        
        # Check if user exists
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
    history = ChatHistory.query.filter_by(user_id=session['user_id']).order_by(
        ChatHistory.timestamp.desc()
    ).limit(50).all()
    
    return jsonify([{
        'message': h.message,
        'response': h.response,
        'timestamp': h.timestamp.isoformat(),
        'collection': h.collection_name
    } for h in history])

# Initialize database and create default admin
def init_db():
    with app.app_context():
        db.create_all()
        
        # Create default admin if not exists
        if not User.query.filter_by(username='admin').first():
            admin = User(
                username='admin',
                email='admin@example.com',
                password=generate_password_hash('admin123'),
                role='admin'
            )
            db.session.add(admin)
            db.session.commit()
            print("Default admin created - username: admin, password: admin123")

if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)