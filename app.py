from flask import Flask, render_template, request, jsonify, session, redirect, url_for, Response, stream_with_context
import json
import os
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import os
import shutil
from datetime import datetime, timezone
from client.src.components.indexer import Indexer
from client.src.components.retriever import Retriever
from client.src.components.graph import RAG_Agent

# Import optimization utilities
try:
    from utils.cache import cache, cache_response, make_cache_key
    from utils.db_optimizer import create_indexes, optimize_connection_pool
    from utils.vector_optimizer import optimized_vector_search
    from utils.background_jobs import process_document_indexing, get_job_status
    OPTIMIZATIONS_ENABLED = True
except ImportError:
    OPTIMIZATIONS_ENABLED = False
    print("Warning: Optimization utilities not available")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-this'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///rag_chatbot.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['DEBUG'] = True  # Enable debug mode for better error messages

# CDN Configuration
app.config['CDN_DOMAIN'] = os.getenv('CDN_DOMAIN', '')  # Set CDN domain in .env
app.config['USE_CDN'] = bool(os.getenv('USE_CDN', 'False').lower() == 'true')

# Load Balancing - Session configuration for stateless design
app.config['SESSION_COOKIE_SECURE'] = os.getenv('SESSION_COOKIE_SECURE', 'False').lower() == 'true'
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

db = SQLAlchemy(app)

# Use timezone-aware UTC timestamps (Python deprecates datetime.utcnow()).
def utcnow():
    return datetime.now(timezone.utc)

# Initialize cache if available
if OPTIMIZATIONS_ENABLED:
    try:
        cache.init_app(app)
        # Optimize database
        # db.engine requires an application context in Flask-SQLAlchemy
        with app.app_context():
            optimize_connection_pool(db)
    except Exception as e:
        print(f"Warning: Could not initialize optimizations: {e}")

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # user, b-manager, a-manager, admin
    created_at = db.Column(db.DateTime, default=utcnow)
    is_active = db.Column(db.Boolean, default=True)
    session_version = db.Column(db.Integer, default=0)  # Incremented to invalidate all sessions

class UserSettings(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(80), db.ForeignKey('user.username'), nullable=False, unique=True)
    ai_model = db.Column(db.String(50), default='gemini-3-pro-preview')
    temperature = db.Column(db.Float, default=1.0)
    max_context_messages = db.Column(db.Integer, default=20)  # Number of messages to keep in context
    context_strategy = db.Column(db.String(20), default='truncate')  # 'truncate' or 'summarize'
    updated_at = db.Column(db.DateTime, default=utcnow, onupdate=utcnow)
    
    # Relationship
    user = db.relationship('User', foreign_keys=[user_id], backref='settings')

class Category(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False, unique=True)
    description = db.Column(db.String(500))
    color = db.Column(db.String(7), default='#4F46E5')  # Hex color for UI
    created_at = db.Column(db.DateTime, default=utcnow)
    created_by = db.Column(db.String(80), db.ForeignKey('user.username'), nullable=False)
    
    # Relationship
    creator = db.relationship('User', foreign_keys=[created_by], backref='created_categories')
    documents = db.relationship('Document', backref='category', lazy=True)

class Document(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(200), nullable=False)
    collection_name = db.Column(db.String(100), nullable=False)
    uploaded_by = db.Column(db.String(80), db.ForeignKey('user.username'), nullable=False)
    uploaded_at = db.Column(db.DateTime, default=utcnow)
    category_id = db.Column(db.Integer, db.ForeignKey('category.id'), nullable=True)
    
    # Relationship to User
    uploader = db.relationship('User', foreign_keys=[uploaded_by], backref='uploaded_documents')

class Conversation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(80), db.ForeignKey('user.username'), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    collection_name = db.Column(db.String(100))
    bookmarked = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=utcnow)
    updated_at = db.Column(db.DateTime, default=utcnow, onupdate=utcnow)
    
    # Relationship to User
    user = db.relationship('User', foreign_keys=[user_id], backref='conversations')
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'title': self.title,
            'collection_name': self.collection_name,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

class ChatHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(80), db.ForeignKey('user.username'), nullable=False)
    conversation_id = db.Column(db.Integer, db.ForeignKey('conversation.id'), nullable=True)
    message = db.Column(db.Text, nullable=False)
    response = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=utcnow)
    collection_name = db.Column(db.String(100))
    
    # Relationship to User
    user = db.relationship('User', foreign_keys=[user_id], backref='chat_history')

# SavedQuery model removed - feature no longer used

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
        session_version = session.get('session_version')

        if not user_id:
            session.clear()
            # Check if this is an API request
            if request.path.startswith('/api/'):
                return jsonify({'error': 'Authentication required'}), 401
            return redirect(url_for('login'))

        user = db.session.get(User, user_id)
        if not user:
            session.clear()
            # Check if this is an API request
            if request.path.startswith('/api/'):
                return jsonify({'error': 'User not found'}), 401
            return redirect(url_for('login'))

        # Check if user is active and session version matches
        if not user.is_active or session_version != user.session_version:
            session.clear()
            # Check if this is an API request
            if request.path.startswith('/api/'):
                return jsonify({'error': 'Session invalidated. Please log in again.'}), 401
            return redirect(url_for('login'))

        return f(*args, **kwargs)
    return decorated_function

def role_required(min_role):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'user_id' not in session:
                return redirect(url_for('login'))
            user = db.session.get(User, session['user_id'])
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

@app.route('/static/manifest.json')
def manifest():
    return app.send_static_file('manifest.json')

@app.route('/static/sw.js')
def service_worker():
    response = app.send_static_file('sw.js')
    response.headers['Content-Type'] = 'application/javascript'
    response.headers['Service-Worker-Allowed'] = '/'
    return response

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.get_json()
        user = db.session.query(User).filter_by(username=data['username']).first()
        
        if user and check_password_hash(user.password, data['password']) and user.is_active:
            session['user_id'] = user.id
            session['username'] = user.username
            session['role'] = user.role
            session['session_version'] = user.session_version  # Store session version
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
    user = db.session.get(User, session.get('user_id'))
    
    # Get analytics data
    total_conversations = db.session.query(Conversation).filter_by(user_id=user.username).count()
    total_messages = db.session.query(ChatHistory).filter_by(user_id=user.username).count()
    recent_conversations = db.session.query(Conversation).filter_by(user_id=user.username).order_by(
        Conversation.updated_at.desc()
    ).limit(5).all()
    
    # Get activity stats (last 7 days)
    from datetime import timedelta
    seven_days_ago = utcnow() - timedelta(days=7)
    recent_messages = db.session.query(ChatHistory).filter(
        ChatHistory.user_id == user.username,
        ChatHistory.timestamp >= seven_days_ago
    ).count()
    
    # Get most used collections
    collection_stats = db.session.query(
        ChatHistory.collection_name,
        db.func.count(ChatHistory.id).label('count')
    ).filter(
        ChatHistory.user_id == user.username
    ).group_by(ChatHistory.collection_name).all()
    
    return render_template(
        'dashboard.html',
        user=user,
        total_conversations=total_conversations,
        total_messages=total_messages,
        recent_messages=recent_messages,
        recent_conversations=recent_conversations,
        collection_stats=collection_stats
    )

@app.route('/api/analytics')
@login_required
def get_analytics():
    user = db.session.get(User, session.get('user_id'))
    
    # Daily message count for last 30 days
    from datetime import timedelta
    thirty_days_ago = utcnow() - timedelta(days=30)
    
    daily_stats = db.session.query(
        db.func.date(ChatHistory.timestamp).label('date'),
        db.func.count(ChatHistory.id).label('count')
    ).filter(
        ChatHistory.user_id == user.username,
        ChatHistory.timestamp >= thirty_days_ago
    ).group_by(db.func.date(ChatHistory.timestamp)).all()
    
    # Collection usage
    collection_usage = db.session.query(
        ChatHistory.collection_name,
        db.func.count(ChatHistory.id).label('count')
    ).filter(
        ChatHistory.user_id == user.username
    ).group_by(ChatHistory.collection_name).all()
    
    return jsonify({
        'daily_stats': [{'date': str(stat.date), 'count': stat.count} for stat in daily_stats],
        'collection_usage': [{'collection': stat.collection_name, 'count': stat.count} for stat in collection_usage]
    })

@app.route('/chat')
@login_required
def chat():
    user = db.session.get(User, session.get('user_id'))
    allowed_collections = get_allowed_collections(user.role)
    conversation_id = request.args.get('conversation_id', type=int)

    documents = db.session.query(Document).filter(
        Document.collection_name.in_(allowed_collections)
    ).all()
    
    conversations = db.session.query(Conversation).filter_by(user_id=user.username).order_by(
        Conversation.updated_at.desc()
    ).all()
    
    current_conversation = None
    if conversation_id:
        current_conversation = db.session.query(Conversation).filter_by(
            id=conversation_id,
            user_id=user.username
        ).first()
        # If conversation_id was provided but not found, reset it to None
        if not current_conversation:
            # print(f"DEBUG: Conversation {conversation_id} not found for user {user.username}, resetting to None")
            conversation_id = None
    
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
    user = db.session.get(User, session.get('user_id'))
    allowed_collections = get_allowed_collections(user.role)

    documents = db.session.query(Document).filter(
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
    user = db.session.get(User, session.get('user_id'))
    users = db.session.query(User).all()
    documents = db.session.query(Document).all()
    conversations = db.session.query(Conversation).order_by(Conversation.updated_at.desc()).all()
    chat_history = db.session.query(ChatHistory).order_by(ChatHistory.timestamp.desc()).limit(100).all()
    categories = db.session.query(Category).order_by(Category.name).all()
    
    return render_template(
        'admin.html', 
        user=user,
        users=users, 
        documents=documents,
        conversations=conversations,
        chat_history=chat_history,
        categories=categories
    )

@app.route('/api/chat', methods=['POST'])
@login_required
def chat_message():
    # print(f"DEBUG: /api/chat endpoint called, method: {request.method}, path: {request.path}")
    try:
        data = request.get_json()
        # print(f"DEBUG: Received data: {data}")
        if not data:
            return jsonify({'error': 'Invalid request data'}), 400
        
        message = data.get('message', '').strip()
        if not message:
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        collection_name = data.get('collection')
        if not collection_name:
            return jsonify({'error': 'Collection name is required'}), 400
        
        conversation_id = data.get('conversation_id')
        is_first_message = data.get('is_first_message', False)
        stream = data.get('stream', False)
        
        # print(f"DEBUG: Message: {message[:50]}..., Collection: {collection_name}, Stream: {stream}")
        # print(f"DEBUG: Conversation ID received: {conversation_id}, type: {type(conversation_id)}, is_first_message: {is_first_message}")

        user = db.session.get(User, session.get('user_id'))
        if not user:
            return jsonify({'error': 'User not found'}), 404

        if collection_name not in get_allowed_collections(user.role):
            return jsonify({'error': 'Access denied'}), 403

        # Get or create conversation
        conversation = None
        # Handle conversation_id: it might be None, null, 0, or a valid ID
        if conversation_id is not None and conversation_id != '' and conversation_id != 'null' and conversation_id != 0:
            try:
                # Convert to int if it's a string
                if isinstance(conversation_id, str):
                    conversation_id = int(conversation_id) if conversation_id.isdigit() else None
                
                if conversation_id:
                    # print(f"DEBUG: Looking for conversation with ID: {conversation_id}, user: {user.username}")
                    conversation = db.session.query(Conversation).filter_by(
                        id=conversation_id,
                        user_id=user.username
                    ).first()
                    
                    if conversation:
                        # print(f"DEBUG: Found conversation: {conversation.id}, title: {conversation.title}")
                        pass
                    else:
                        # print(f"DEBUG: Conversation {conversation_id} not found for user {user.username}")
                        # Check if conversation exists for other users (for debugging)
                        # other_conv = db.session.query(Conversation).filter_by(id=conversation_id).first()
                        # if other_conv:
                        #     print(f"DEBUG: Conversation exists but belongs to user: {other_conv.user_id}")
                        # else:
                        #     print(f"DEBUG: Conversation {conversation_id} does not exist at all")
                        # Don't return error here - create a new conversation instead
                        conversation = None
            except (ValueError, TypeError) as e:
                # print(f"DEBUG: Invalid conversation_id format: {e}")
                conversation = None
        
        if not conversation:
            # Create title from first message (max 50 chars)
            title = message[:50] + "..." if len(message) > 50 else message
            # print(f"DEBUG: Creating new conversation with title: {title}")
            conversation = Conversation(
                user_id=user.username,  # Store username
                title=title,
                collection_name=collection_name
            )
            db.session.add(conversation)
            db.session.commit()
            # print(f"DEBUG: Created new conversation with ID: {conversation.id}")
        
        if not conversation:
            # Create title from first message (max 50 chars)
            title = message[:50] + "..." if len(message) > 50 else message
            conversation = Conversation(
                user_id=user.username,  # Store username
                title=title,
                collection_name=collection_name
            )
            db.session.add(conversation)
            db.session.commit()

        from langchain_core.messages import HumanMessage, AIMessage
        
        # Get user settings or create default
        user_settings = db.session.query(UserSettings).filter_by(user_id=user.username).first()
        if not user_settings:
            user_settings = UserSettings(
                user_id=user.username,
                ai_model='gemini-3-pro-preview',
                temperature=1.0,
                max_context_messages=20,
                context_strategy='truncate'
            )
            db.session.add(user_settings)
            db.session.commit()
        
        # Get model and temperature from request or user settings
        # Temperature is fixed at 0.5
        ai_model = data.get('ai_model') or user_settings.ai_model
        temperature = 0.5  # Fixed temperature value
        max_context_messages = int(data.get('max_context_messages') or user_settings.max_context_messages)
        context_strategy = data.get('context_strategy') or user_settings.context_strategy
        
        # Debug: Log the model being used
        # print(f"DEBUG: Selected model from request: {data.get('ai_model')}")
        # print(f"DEBUG: User settings model: {user_settings.ai_model}")
        # print(f"DEBUG: Final model being used: {ai_model}")
        
        # Check cache for similar queries (intelligent caching)
        if OPTIMIZATIONS_ENABLED:
            cache_key = f"chat_response_{make_cache_key(message, user.username, collection_name)}"
            cached_response = cache.get(cache_key)
            if cached_response:
                # print(f"DEBUG: Returning cached response for query")
                return jsonify(cached_response)
        
        # Validate API key availability for the selected model
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        if ai_model.startswith('gpt-') or ai_model.startswith('o1-') or ai_model.startswith('o3-'):
            if not os.getenv('OPENAI_API_KEY'):
                return jsonify({'error': 'OpenAI API key not configured. Please add OPENAI_API_KEY to your .env file.'}), 400
        elif ai_model.startswith('claude-') or ai_model.startswith('sonnet-') or ai_model.startswith('opus-') or ai_model.startswith('haiku-'):
            if not os.getenv('ANTHROPIC_API_KEY'):
                return jsonify({'error': 'Anthropic API key not configured. Please add ANTHROPIC_API_KEY to your .env file.'}), 400
        else:
            # Gemini models
            if not os.getenv('GOOGLE_API_KEY'):
                return jsonify({'error': 'Google API key not configured. Please add GOOGLE_API_KEY to your .env file.'}), 400
        
        try:
            retriever = Retriever(collection_name=collection_name)
            vectorstore = retriever.get_vector_store()
        except Exception as e:
            return jsonify({'error': f'Failed to initialize retriever: {str(e)}'}), 500
        
        # Use conversation-specific thread_id
        thread_id = f"conversation_{conversation.id}"
        
        # Create streaming callback if streaming is requested
        # Note: For now, we'll not use callbacks as LangGraph handles streaming differently
        streaming_handler = None
        
        try:
            agent = RAG_Agent(
                vector_store=vectorstore, 
                thread_id=thread_id, 
                streaming_callback=streaming_handler,
                model=ai_model,
                temperature=temperature
            )
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"Agent initialization error: {error_trace}")
            return jsonify({'error': f'Failed to initialize agent: {str(e)}'}), 500
        
        # Load previous conversation history for context
        previous_messages = []
        chat_history = db.session.query(ChatHistory).filter_by(
            conversation_id=conversation.id
        ).order_by(ChatHistory.timestamp.asc()).all()
        
        for history_item in chat_history:
            previous_messages.append(HumanMessage(content=history_item.message))
            previous_messages.append(AIMessage(content=history_item.response))
        
        # Context window management
        if len(previous_messages) > max_context_messages * 2:  # *2 because each exchange has 2 messages
            if context_strategy == 'truncate':
                # Keep only the most recent messages
                previous_messages = previous_messages[-(max_context_messages * 2):]
                # print(f"DEBUG: Truncated context to {max_context_messages * 2} messages")
            elif context_strategy == 'summarize':
                # Keep first few and last few, summarize the middle
                keep_first = 2  # Keep first exchange
                keep_last = max_context_messages - 1
                if len(previous_messages) > (keep_first + keep_last) * 2:
                    first_messages = previous_messages[:keep_first * 2]
                    last_messages = previous_messages[-(keep_last * 2):]
                    # Add a summary message (simplified for now)
                    summary_msg = HumanMessage(content=f"[Previous conversation context: {len(previous_messages) - (keep_first + keep_last) * 2} messages summarized]")
                    previous_messages = first_messages + [summary_msg] + last_messages
                    # print(f"DEBUG: Summarized context, kept {len(previous_messages)} messages")
        
        # Add current message
        previous_messages.append(HumanMessage(content=message))
        
        if stream:
            # Streaming response
            def generate_stream():
                full_response = ""
                citations = []
                retrieved_docs = []  # Store documents retrieved during tool calls
                try:
                    graph = agent.langgraph_graph()
                    accumulated_length = 0
                    
                    # Stream through graph execution
                    # LangGraph's astream returns an async generator
                    # We need to convert it to sync for Flask
                    import asyncio
                    
                    # Get the async generator
                    async_gen = graph.astream(
                        {"messages": previous_messages},
                        config=agent.config
                    )
                    
                    # Create or get event loop
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_closed():
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    
                    # Process chunks from async generator
                    while True:
                        try:
                            chunk = loop.run_until_complete(async_gen.__anext__())
                        except StopAsyncIteration:
                            break
                        except Exception as e:
                            import traceback
                            print(f"Error getting chunk: {e}\n{traceback.format_exc()}")
                            # Fallback: use invoke instead
                            try:
                                final_result = graph.invoke(
                                    {"messages": previous_messages},
                                    config=agent.config
                                )
                                if 'messages' in final_result and len(final_result['messages']) > 0:
                                    last_msg = final_result['messages'][-1]
                                    if hasattr(last_msg, 'content'):
                                        full_response = str(last_msg.content)
                                        yield f"data: {json.dumps({'chunk': full_response, 'done': False})}\n\n"
                            except Exception as fallback_error:
                                print(f"Fallback invoke also failed: {fallback_error}")
                            break
                        
                        # Process the chunk
                        
                        # Process each node's output in the chunk
                        for node_name, node_output in chunk.items():
                            if 'messages' in node_output and len(node_output['messages']) > 0:
                                # Check all messages in this output, not just the last one
                                for msg in node_output['messages']:
                                    # Extract documents from tool messages
                                    if hasattr(msg, 'name') and msg.name == 'document_search_tool':
                                        # Parse the tool output to extract document metadata
                                        import re
                                        content = str(msg.content) if hasattr(msg, 'content') else ''
                                        # Extract metadata blocks from the tool output
                                        # Format: "METADATA: {'source': ..., 'page': ..., 'type': ...}"
                                        metadata_blocks = re.findall(r'METADATA:\s*({[^}]+})', content)
                                        for metadata_str in metadata_blocks:
                                            try:
                                                import ast
                                                # Parse the metadata dictionary
                                                metadata = ast.literal_eval(metadata_str)
                                                retrieved_docs.append({
                                                    'metadata': metadata,
                                                    'source': metadata.get('source', 'Unknown')
                                                })
                                            except Exception as parse_error:
                                                print(f"Error parsing metadata: {parse_error}")
                                                pass
                                
                                last_message = node_output['messages'][-1]
                                
                                # Check if this is an AI message (not tool message)
                                # Try different ways to check message type
                                is_ai_message = False
                                try:
                                    if hasattr(last_message, 'get_type'):
                                        is_ai_message = last_message.get_type() == 'ai'
                                    elif hasattr(last_message, 'type'):
                                        is_ai_message = getattr(last_message, 'type', '') == 'ai'
                                    elif hasattr(last_message, '__class__'):
                                        # Check by class name as fallback
                                        class_name = last_message.__class__.__name__
                                        is_ai_message = 'AI' in class_name or 'AIMessage' in class_name
                                    else:
                                        # If we can't determine type, assume it's AI if it has content and no tool_calls
                                        is_ai_message = (hasattr(last_message, 'content') and 
                                                        not hasattr(last_message, 'tool_calls'))
                                except Exception as type_check_error:
                                    # If type check fails, default to checking content only
                                    is_ai_message = hasattr(last_message, 'content')
                                
                                if hasattr(last_message, 'content') and is_ai_message:
                                    try:
                                        content = str(last_message.content)
                                        
                                        # If we have new content, send it
                                        if len(content) > accumulated_length:
                                            new_chunk = content[accumulated_length:]
                                            accumulated_length = len(content)
                                            full_response = content
                                            
                                            # Send in smaller chunks for better streaming effect
                                            if new_chunk:
                                                # Split into words for smoother streaming
                                                words = new_chunk.split(' ')
                                                for i, word in enumerate(words):
                                                    chunk_to_send = word + (' ' if i < len(words) - 1 else '')
                                                    yield f"data: {json.dumps({'chunk': chunk_to_send, 'done': False})}\n\n"
                                    except Exception as e:
                                        import traceback
                                        print(f"Error processing message content: {e}\n{traceback.format_exc()}")
                    
                    # Ensure we have the final response
                    if not full_response:
                        # Get final result to extract response
                        final_result = graph.invoke(
                            {"messages": previous_messages},
                            config=agent.config
                        )
                        if 'messages' in final_result and len(final_result['messages']) > 0:
                            last_msg = final_result['messages'][-1]
                            if hasattr(last_msg, 'content'):
                                full_response = str(last_msg.content)
                                if full_response and accumulated_length == 0:
                                    # Send full response if we didn't stream incrementally
                                    yield f"data: {json.dumps({'chunk': full_response, 'done': False})}\n\n"
                    
                    # Extract citations from documents actually retrieved by the agent
                    # Use the documents we collected during streaming
                    try:
                        # If we collected documents during streaming, use those
                        if retrieved_docs:
                            # Build citations from retrieved documents
                            seen_sources = set()
                            for doc_info in retrieved_docs:
                                metadata = doc_info['metadata']
                                source = doc_info['source']
                                if source and source not in seen_sources:
                                    page = metadata.get('page_num') or metadata.get('page')
                                    # Handle page ranges if multiple pages are referenced
                                    if isinstance(page, (list, tuple)):
                                        if len(page) > 1:
                                            page = f"{min(page)}-{max(page)}"
                                        else:
                                            page = page[0] if page else None
                                    
                                    citations.append({
                                        'source': os.path.basename(source) if source else 'Unknown',
                                        'page': page,
                                        'type': metadata.get('type', 'text')
                                    })
                                    seen_sources.add(source)
                        else:
                            # Fallback to direct retrieval if we didn't collect documents
                            retriever = agent.retriever
                            search_results = retriever.invoke(message)
                            seen_sources = set()
                            for doc in search_results[:5]:
                                metadata = doc.metadata
                                source = metadata.get('source', 'Unknown')
                                if source and source not in seen_sources:
                                    page = metadata.get('page_num') or metadata.get('page')
                                    citations.append({
                                        'source': os.path.basename(source) if source else 'Unknown',
                                        'page': page,
                                        'type': metadata.get('type', 'text')
                                    })
                                    seen_sources.add(source)
                    except Exception as e:
                        import traceback
                        print(f"Error extracting citations: {e}\n{traceback.format_exc()}")
                        # Fallback to simple retrieval
                        try:
                            retriever = agent.retriever
                            search_results = retriever.invoke(message)
                            seen_sources = set()
                            for doc in search_results[:5]:
                                metadata = doc.metadata
                                source = metadata.get('source', 'Unknown')
                                if source and source not in seen_sources:
                                    citations.append({
                                        'source': os.path.basename(source) if source else 'Unknown',
                                        'page': metadata.get('page_num', metadata.get('page', None)),
                                        'type': metadata.get('type', 'text')
                                    })
                                    seen_sources.add(source)
                        except:
                            pass
                    
                    # Final message
                    yield f"data: {json.dumps({'chunk': '', 'done': True, 'full_response': full_response, 'citations': citations, 'conversation_id': conversation.id})}\n\n"
                    
                    # Save chat history
                    db.session.add(ChatHistory(
                        user_id=user.username,
                        conversation_id=conversation.id,
                        message=message,
                        response=full_response,
                        collection_name=collection_name
                    ))
                    conversation.updated_at = utcnow()
                    db.session.commit()
                    
                except Exception as e:
                    import traceback
                    error_msg = f"{str(e)}\n{traceback.format_exc()}"
                    print(f"Streaming error: {error_msg}")
                    import sys
                    sys.stderr.write(f"Streaming error: {error_msg}\n")
                    
                    # If we have a partial response, try to save it
                    if full_response:
                        try:
                            db.session.add(ChatHistory(
                                user_id=user.username,
                                conversation_id=conversation.id,
                                message=message,
                                response=full_response,
                                collection_name=collection_name
                            ))
                            conversation.updated_at = utcnow()
                            db.session.commit()
                        except:
                            db.session.rollback()
                    
                    # Send error to client
                    try:
                        yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"
                    except:
                        pass
            
            return Response(stream_with_context(generate_stream()), mimetype='text/event-stream')
        else:
            # Non-streaming response (original behavior)
            # print("DEBUG: Using non-streaming mode")
            try:
                # print("DEBUG: Invoking agent graph...")
                result = agent.langgraph_graph().invoke(
                    {"messages": previous_messages},
                    config=agent.config
                )
                # print(f"DEBUG: Agent result received: {type(result)}, keys: {result.keys() if isinstance(result, dict) else 'N/A'}")
            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                # print(f"DEBUG: Error invoking agent: {error_trace}")
                return jsonify({'error': f'Failed to generate response: {str(e)}'}), 500

            if not result or 'messages' not in result or len(result['messages']) == 0:
                # print("DEBUG: Empty result from agent")
                return jsonify({'error': 'Empty response from agent'}), 500

            # print(f"DEBUG: Extracting response from result, messages count: {len(result['messages'])}")
            last_message = result['messages'][-1]
            # print(f"DEBUG: Last message type: {type(last_message)}, has text: {hasattr(last_message, 'text')}, has content: {hasattr(last_message, 'content')}")
            
            # Try multiple ways to extract the response
            response = None
            if hasattr(last_message, 'text'):
                response = last_message.text
            elif hasattr(last_message, 'content'):
                content = last_message.content
                if isinstance(content, str):
                    response = content
                elif isinstance(content, list):
                    # Handle list content (e.g., from multimodal messages)
                    response = ' '.join(str(item) for item in content)
                else:
                    response = str(content)
            else:
                # Fallback: try to get any string representation
                response = str(last_message)
            
            if not response or (isinstance(response, str) and len(response.strip()) == 0):
                # print("DEBUG: WARNING - Response is empty or None!")
                # Try to get response from earlier messages
                for msg in reversed(result['messages']):
                    if hasattr(msg, 'content') and msg.content:
                        response = str(msg.content)
                        if response and len(response.strip()) > 0:
                            # print(f"DEBUG: Found response in earlier message")
                            break
            
            # print(f"DEBUG: Response extracted, length: {len(response) if response else 0}, preview: {str(response)[:100] if response else 'None'}...")
            
            # Ensure we have a valid response
            if not response or (isinstance(response, str) and len(response.strip()) == 0):
                # print("DEBUG: ERROR - No valid response found!")
                return jsonify({'error': 'No response generated from the agent'}), 500
            
            # Extract citations from documents actually retrieved by the agent
            # Parse the result to find tool messages with document metadata
            citations = []
            try:
                # Look through all messages in the result to find tool messages
                retrieved_docs = []
                for msg in result.get('messages', []):
                    # Check if this is a tool message from document_search_tool
                    if hasattr(msg, 'name') and msg.name == 'document_search_tool':
                        # Parse the tool output to extract document metadata
                        import re
                        content = str(msg.content) if hasattr(msg, 'content') else ''
                        # Extract metadata blocks from the tool output
                        # Format: "METADATA: {'source': ..., 'page': ..., 'type': ...}"
                        metadata_blocks = re.findall(r'METADATA:\s*({[^}]+})', content)
                        for metadata_str in metadata_blocks:
                            try:
                                import ast
                                # Parse the metadata dictionary
                                metadata = ast.literal_eval(metadata_str)
                                retrieved_docs.append({
                                    'metadata': metadata,
                                    'source': metadata.get('source', 'Unknown')
                                })
                            except Exception as parse_error:
                                print(f"Error parsing metadata: {parse_error}")
                                pass
                
                # If we couldn't extract from tool messages, fall back to direct retrieval
                if not retrieved_docs:
                    retriever = agent.retriever
                    search_results = retriever.invoke(message)
                    for doc in search_results[:5]:
                        retrieved_docs.append({
                            'metadata': doc.metadata,
                            'source': doc.metadata.get('source', 'Unknown')
                        })
                
                # Build citations from retrieved documents
                seen_sources = set()
                for doc_info in retrieved_docs:
                    metadata = doc_info['metadata']
                    source = doc_info['source']
                    if source and source not in seen_sources:
                        page = metadata.get('page_num') or metadata.get('page')
                        # Handle page ranges if multiple pages are referenced
                        if isinstance(page, (list, tuple)):
                            if len(page) > 1:
                                page = f"{min(page)}-{max(page)}"
                            else:
                                page = page[0] if page else None
                        
                        citations.append({
                            'source': os.path.basename(source) if source else 'Unknown',
                            'page': page,
                            'type': metadata.get('type', 'text')
                        })
                        seen_sources.add(source)
            except Exception as e:
                import traceback
                print(f"Error extracting citations: {e}\n{traceback.format_exc()}")
                # Fallback to simple retrieval
                try:
                    retriever = agent.retriever
                    search_results = retriever.invoke(message)
                    seen_sources = set()
                    for doc in search_results[:5]:
                        metadata = doc.metadata
                        source = metadata.get('source', 'Unknown')
                        if source and source not in seen_sources:
                            citations.append({
                                'source': os.path.basename(source) if source else 'Unknown',
                                'page': metadata.get('page_num', metadata.get('page', None)),
                                'type': metadata.get('type', 'text')
                            })
                            seen_sources.add(source)
                except:
                    pass

            # Save chat history with username
            db.session.add(ChatHistory(
                user_id=user.username,  # Store username
                conversation_id=conversation.id,
                message=message,
                response=response,
                collection_name=collection_name
            ))
            
            # Update conversation timestamp
            conversation.updated_at = utcnow()
            db.session.commit()

            response_data = {
                'response': response,
                'conversation_id': conversation.id,
                'citations': citations
            }
            # print(f"DEBUG: Returning response, response length: {len(response) if response else 0}, citations: {len(citations)}")
            return jsonify(response_data)
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Chat message error: {error_trace}")
        db.session.rollback()
        return jsonify({'error': f'Internal server error: {str(e)}', 'trace': error_trace if app.debug else None}), 500

@app.route('/api/upload', methods=['POST'])
@role_required('b-manager')
def upload_document():
    try:
        # Support both single file and multiple files (bulk upload)
        files = request.files.getlist('file')  # Get list of files
        if not files or len(files) == 0:
            return jsonify({'error': 'No files selected'}), 400
        
        collection_name = request.form.get('collection_name')
        category_id = request.form.get('category_id')
        if category_id:
            try:
                category_id = int(category_id)
                # Validate category exists
                category = db.session.get(Category, category_id)
                if not category:
                    category_id = None
            except (ValueError, TypeError):
                category_id = None

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

        allowed_extensions = {'.pdf', '.docx', '.doc', '.ppt', '.pptx', '.txt'}
        uploaded_files = []
        failed_files = []

        # Process each file
        for file in files:
            if not file or file.filename == '':
                continue

            filename = file.filename
            file_ext = os.path.splitext(filename)[1].lower()
            
            # Validate file extension
            if file_ext not in allowed_extensions:
                failed_files.append({
                    'filename': filename,
                    'error': f'Unsupported file format. Allowed formats: {", ".join(allowed_extensions)}'
                })
                continue

            filepath = os.path.join(level_folder, filename)
            
            # Check for duplicate files - prevent duplicates entirely
            # Check if file already exists in database with same name and collection
            existing_doc = db.session.query(Document).filter_by(
                filename=filename,
                collection_name=collection_name
            ).first()
            
            # Also check if file exists on disk
            if existing_doc or os.path.exists(filepath):
                failed_files.append({
                    'filename': filename,
                    'error': f'File "{filename}" already exists in collection "{collection_name}". Duplicate files are not allowed.'
                })
                continue

            try:
                file.save(filepath)

                # Index document
                try:
                    indexer = Indexer(filepath)
                    indexer.index_document(level=collection_name)
                except Exception as index_error:
                    # Clean up saved file if indexing fails
                    if os.path.exists(filepath):
                        os.remove(filepath)
                    failed_files.append({
                        'filename': filename,
                        'error': f'Failed to index document: {str(index_error)}'
                    })
                    continue

                # Save metadata in DB with username
                try:
                    doc = Document(
                        filename=filename,
                        collection_name=collection_name,
                        uploaded_by=session['username'],
                        category_id=category_id if category_id else None
                    )
                    db.session.add(doc)
                    db.session.commit()
                    uploaded_files.append({
                        'filename': filename,
                        'id': doc.id
                    })
                except Exception as db_error:
                    # Clean up saved file if DB save fails
                    if os.path.exists(filepath):
                        os.remove(filepath)
                    db.session.rollback()
                    failed_files.append({
                        'filename': filename,
                        'error': f'Failed to save document metadata: {str(db_error)}'
                    })
                    continue

            except Exception as e:
                failed_files.append({
                    'filename': filename,
                    'error': f'Upload failed: {str(e)}'
                })

        if len(uploaded_files) == 0:
            return jsonify({
                'error': 'All files failed to upload',
                'failed_files': failed_files
            }), 400

        return jsonify({
            'success': True,
            'message': f'Successfully uploaded {len(uploaded_files)} file(s)',
            'uploaded_files': uploaded_files,
            'failed_files': failed_files if failed_files else None
        }), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

# Category Management Endpoints
@app.route('/api/categories', methods=['GET'])
@login_required
def get_categories():
    """Get all categories"""
    try:
        categories = db.session.query(Category).order_by(Category.name).all()
        return jsonify({
            'categories': [{
                'id': cat.id,
                'name': cat.name,
                'description': cat.description,
                'color': cat.color,
                'created_at': cat.created_at.isoformat() if cat.created_at else None,
                'created_by': cat.created_by,
                'document_count': len(cat.documents) if cat.documents else 0
            } for cat in categories]
        }), 200
    except Exception as e:
        return jsonify({'error': f'Failed to fetch categories: {str(e)}'}), 500

@app.route('/api/categories', methods=['POST'])
@role_required('b-manager')
def create_category():
    """Create a new category"""
    try:
        data = request.get_json()
        name = data.get('name', '').strip()
        description = data.get('description', '').strip()
        color = data.get('color', '#4F46E5')

        if not name:
            return jsonify({'error': 'Category name is required'}), 400

        # Check if category already exists
        existing = db.session.query(Category).filter_by(name=name).first()
        if existing:
            return jsonify({'error': 'Category with this name already exists'}), 400

        category = Category(
            name=name,
            description=description,
            color=color,
            created_by=session['username']
        )
        db.session.add(category)
        db.session.commit()

        return jsonify({
            'message': 'Category created successfully',
            'category': {
                'id': category.id,
                'name': category.name,
                'description': category.description,
                'color': category.color
            }
        }), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Failed to create category: {str(e)}'}), 500

@app.route('/api/categories/<int:category_id>', methods=['PUT'])
@role_required('b-manager')
def update_category(category_id):
    """Update a category"""
    try:
        category = db.session.get(Category, category_id)
        if not category:
            return jsonify({'error': 'Category not found'}), 404

        data = request.get_json()
        name = data.get('name', '').strip()
        description = data.get('description', '').strip()
        color = data.get('color')

        if name:
            # Check if another category with this name exists
            existing = db.session.query(Category).filter_by(name=name).filter(Category.id != category_id).first()
            if existing:
                return jsonify({'error': 'Category with this name already exists'}), 400
            category.name = name

        if description is not None:
            category.description = description
        if color:
            category.color = color

        db.session.commit()
        return jsonify({
            'message': 'Category updated successfully',
            'category': {
                'id': category.id,
                'name': category.name,
                'description': category.description,
                'color': category.color
            }
        }), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Failed to update category: {str(e)}'}), 500

@app.route('/api/categories/<int:category_id>', methods=['DELETE'])
@role_required('b-manager')
def delete_category(category_id):
    """Delete a category"""
    try:
        category = db.session.get(Category, category_id)
        if not category:
            return jsonify({'error': 'Category not found'}), 404

        # Remove category from documents (set to None)
        db.session.query(Document).filter_by(category_id=category_id).update({'category_id': None})
        
        db.session.delete(category)
        db.session.commit()
        return jsonify({'message': 'Category deleted successfully'}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Failed to delete category: {str(e)}'}), 500

# User AI Settings Endpoints
@app.route('/api/settings/ai', methods=['GET'])
@login_required
def get_ai_settings():
    """Get user's AI settings"""
    try:
        user = db.session.get(User, session.get('user_id'))
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        user_settings = db.session.query(UserSettings).filter_by(user_id=user.username).first()
        if not user_settings:
            # Create default settings
            user_settings = UserSettings(
                user_id=user.username,
                ai_model='gemini-3-pro-preview',
                temperature=1.0,
                max_context_messages=20,
                context_strategy='truncate'
            )
            db.session.add(user_settings)
            db.session.commit()
        
        return jsonify({
            'ai_model': user_settings.ai_model,
            'temperature': user_settings.temperature,
            'max_context_messages': user_settings.max_context_messages,
            'context_strategy': user_settings.context_strategy
        }), 200
    except Exception as e:
        return jsonify({'error': f'Failed to get settings: {str(e)}'}), 500

@app.route('/api/settings/ai', methods=['PUT'])
@login_required
def update_ai_settings():
    """Update user's AI settings"""
    try:
        user = db.session.get(User, session.get('user_id'))
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        data = request.get_json()
        
        user_settings = db.session.query(UserSettings).filter_by(user_id=user.username).first()
        if not user_settings:
            user_settings = UserSettings(user_id=user.username)
            db.session.add(user_settings)
        
        # Available models grouped by provider (top 2 per provider)
        available_models = {
            'gemini': [
                'gemini-3-pro-preview',
                'gemini-2.0-flash-exp'
            ],
            'openai': [
                'gpt-5',
                'gpt-5.2'
            ],
            'anthropic': [
                'claude-sonnet-4-5',
                'claude-opus-4-5'
            ]
        }
        
        # Flatten for validation
        all_models = []
        for provider_models in available_models.values():
            all_models.extend(provider_models)
        
        if 'ai_model' in data:
            if data['ai_model'] in all_models:
                user_settings.ai_model = data['ai_model']
            else:
                return jsonify({'error': 'Invalid AI model'}), 400
        
        # Temperature is fixed at 0.5, not user-configurable
        user_settings.temperature = 0.5
        
        if 'max_context_messages' in data:
            max_ctx = int(data['max_context_messages'])
            if 5 <= max_ctx <= 100:
                user_settings.max_context_messages = max_ctx
            else:
                return jsonify({'error': 'Max context messages must be between 5 and 100'}), 400
        
        if 'context_strategy' in data:
            if data['context_strategy'] in ['truncate', 'summarize']:
                user_settings.context_strategy = data['context_strategy']
            else:
                return jsonify({'error': 'Invalid context strategy'}), 400
        
        db.session.commit()
        
        return jsonify({
            'message': 'Settings updated successfully',
            'settings': {
                'ai_model': user_settings.ai_model,
                'temperature': user_settings.temperature,
                'max_context_messages': user_settings.max_context_messages,
                'context_strategy': user_settings.context_strategy
            }
        }), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Failed to update settings: {str(e)}'}), 500

@app.route('/api/users', methods=['GET', 'POST'])
@role_required('admin')
def manage_users():
    if request.method == 'POST':
        data = request.get_json()
        
        if db.session.query(User).filter_by(username=data['username']).first():
            return jsonify({'error': 'Username already exists'}), 400
        
        if db.session.query(User).filter_by(email=data['email']).first():
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
    
    users = db.session.query(User).all()
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
    user = db.session.get(User, user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    if request.method == 'PUT':
        data = request.get_json()
        was_active = user.is_active
        if 'role' in data:
            user.role = data['role']
        if 'is_active' in data:
            new_active_status = data['is_active']
            user.is_active = new_active_status
            # Increment session_version whenever is_active changes to invalidate all sessions
            # This ensures users are logged out from all devices when status changes
            if was_active != new_active_status:
                user.session_version = (user.session_version or 0) + 1
        db.session.commit()
        return jsonify({'success': True})
    
    if request.method == 'DELETE':
        # IMPORTANT: User is referenced by multiple tables via username FKs.
        # Deleting the User without deleting dependents causes SQLAlchemy to
        # null out child FKs (e.g., ChatHistory.user_id), which violates NOT NULL.
        try:
            username = user.username

            # Prevent deleting yourself (optional safety)
            if session.get('username') == username:
                return jsonify({'error': 'You cannot delete your own account while logged in.'}), 400

            # Prevent deleting the built-in admin account
            if username == 'admin':
                return jsonify({'error': 'The admin account cannot be deleted.'}), 400

            # Ensure admin exists (we reassign ownership to admin)
            admin_user = db.session.query(User).filter_by(username='admin').first()
            if not admin_user:
                return jsonify({'error': 'Cannot delete user because admin account does not exist.'}), 500

            # Transfer ownership of uploaded documents to admin (keep documents)
            db.session.query(Document).filter_by(uploaded_by=username).update(
                {'uploaded_by': 'admin'},
                synchronize_session=False
            )

            # Transfer ownership of categories to admin (keep categories)
            db.session.query(Category).filter_by(created_by=username).update(
                {'created_by': 'admin'},
                synchronize_session=False
            )

            # Delete chat history rows first
            db.session.query(ChatHistory).filter_by(user_id=username).delete(synchronize_session=False)

            # Delete conversations (and any remaining messages by conversation_id)
            conv_ids = [c.id for c in db.session.query(Conversation.id).filter_by(user_id=username).all()]
            if conv_ids:
                db.session.query(ChatHistory).filter(ChatHistory.conversation_id.in_(conv_ids)).delete(synchronize_session=False)
                db.session.query(Conversation).filter(Conversation.id.in_(conv_ids)).delete(synchronize_session=False)

            # Delete user settings
            db.session.query(UserSettings).filter_by(user_id=username).delete(synchronize_session=False)

            db.session.delete(user)
            db.session.commit()
            return jsonify({'success': True})
        except Exception as e:
            db.session.rollback()
            return jsonify({'error': f'Failed to delete user: {str(e)}'}), 500

@app.route('/api/chat-history')
@login_required
def get_chat_history():
    conversation_id = request.args.get('conversation_id', type=int)
    user = db.session.get(User, session.get('user_id'))
    
    if conversation_id:
        # Verify conversation belongs to user
        conversation = db.session.query(Conversation).filter_by(
            id=conversation_id,
            user_id=user.username
        ).first()
        if not conversation:
            return jsonify({'error': 'Conversation not found'}), 404
        
        history = db.session.query(ChatHistory).filter_by(
            conversation_id=conversation_id
        ).order_by(ChatHistory.timestamp.asc()).limit(100).all()
    else:
        # Fallback to old behavior for backward compatibility
        collection_name = request.args.get('collection', None)
        query = db.session.query(ChatHistory).filter_by(user_id=user.username)
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
    user = db.session.get(User, session.get('user_id'))
    
    if not user:
        # print(f"DEBUG: User not found for session user_id: {session.get('user_id')}")
        return jsonify({'error': 'User not found'}), 404
    
    # print(f"DEBUG: manage_conversations - User: {user.username}, ID: {user.id}, Session user_id: {session.get('user_id')}")
    
    if request.method == 'POST':
        # Create new conversation
        data = request.get_json()
        title = data.get('title', 'New Conversation')
        collection_name = data.get('collection_name', 'level-1')
        
        conversation = Conversation(
            user_id=user.username,  # Store username
            title=title,
            collection_name=collection_name
        )
        db.session.add(conversation)
        db.session.commit()
        
        return jsonify(conversation.to_dict()), 201
    
    # GET: List all conversations
    # Use username from user object or fallback to session username
    username = user.username if user else session.get('username')
    # print(f"DEBUG: Querying conversations for user_id: {username}")
    # print(f"DEBUG: User object - ID: {user.id if user else 'None'}, Username: {user.username if user else 'None'}")
    # print(f"DEBUG: Session - user_id: {session.get('user_id')}, username: {session.get('username')}")
    
    if not username:
        # print("DEBUG: ERROR - No username available!")
        return jsonify({'error': 'User information not available'}), 400
    
    # Query conversations by username (user_id in Conversation table stores username)
    conversations = db.session.query(Conversation).filter_by(user_id=username).order_by(
        Conversation.updated_at.desc()
    ).all()
    
    # print(f"DEBUG: Found {len(conversations)} conversations for user {username}")
    # for conv in conversations:
    #     print(f"DEBUG: Conversation ID: {conv.id}, Title: {conv.title}, User ID: {conv.user_id}, Updated: {conv.updated_at}")
    
    # Check if user wants to include conversations from 'admin' user
    # This handles the case where user was previously logged in as admin
    include_admin = request.args.get('include_admin', 'false').lower() == 'true'
    
    # If user has admin role OR explicitly requests admin conversations, include them
    if (user and user.role == 'admin') or include_admin:
        # print(f"DEBUG: Including admin conversations (role={user.role if user else 'None'}, include_admin={include_admin})")
        admin_conversations = db.session.query(Conversation).filter_by(user_id='admin').order_by(
            Conversation.updated_at.desc()
        ).all()
        # Combine and deduplicate
        all_conv_ids = {c.id for c in conversations}
        for admin_conv in admin_conversations:
            if admin_conv.id not in all_conv_ids:
                conversations.append(admin_conv)
                all_conv_ids.add(admin_conv.id)
        # Re-sort by updated_at
        conversations.sort(key=lambda x: x.updated_at, reverse=True)
        # print(f"DEBUG: After including admin conversations: {len(conversations)} total")
    
    # Also check all conversations (for debugging)
    # all_conversations = db.session.query(Conversation).all()
    # print(f"DEBUG: Total conversations in DB: {len(all_conversations)}")
    # for conv in all_conversations:
    #     print(f"DEBUG: All - Conversation ID: {conv.id}, User ID: {conv.user_id}, Title: {conv.title}")
    
    # If no conversations found, try alternative queries (for debugging)
    # if len(conversations) == 0:
    #     print("DEBUG: No conversations found with username filter. Trying alternative queries...")
    #     # Try with user ID (in case old data was stored with ID)
    #     alt_conversations = db.session.query(Conversation).filter_by(user_id=str(user.id)).all()
    #     print(f"DEBUG: Found {len(alt_conversations)} conversations with user_id={user.id}")
    #     
    #     # Try case-insensitive search
    #     from sqlalchemy import func
    #     case_insensitive = db.session.query(Conversation).filter(
    #         func.lower(Conversation.user_id) == func.lower(username)
    #     ).all()
    #     print(f"DEBUG: Found {len(case_insensitive)} conversations with case-insensitive search")
    
    result = [c.to_dict() for c in conversations]
    # print(f"DEBUG: Returning {len(result)} conversations as JSON")
    return jsonify(result)

@app.route('/api/conversations/migrate', methods=['POST'])
@login_required
def migrate_conversations():
    """Migrate conversations from one user to another (e.g., from admin to current user)"""
    user = db.session.get(User, session.get('user_id'))
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    data = request.get_json() or {}
    from_user = data.get('from_user', 'admin')
    to_user = user.username
    
    if from_user == to_user:
        return jsonify({'error': 'Cannot migrate to same user'}), 400
    
    # Get conversations from source user
    conversations = db.session.query(Conversation).filter_by(user_id=from_user).all()
    
    if not conversations:
        return jsonify({'message': f'No conversations found for user {from_user}', 'migrated': 0}), 200
    
    # Migrate conversations
    migrated_count = 0
    for conv in conversations:
        conv.user_id = to_user
        migrated_count += 1
    
    db.session.commit()
    
    return jsonify({
        'message': f'Successfully migrated {migrated_count} conversations from {from_user} to {to_user}',
        'migrated': migrated_count
    }), 200

@app.route('/api/conversations/<int:conversation_id>', methods=['PUT', 'DELETE'])
@login_required
def manage_conversation(conversation_id):
    user = db.session.get(User, session.get('user_id'))
    # print(f"DEBUG: manage_conversation - conversation_id: {conversation_id}, user: {user.username}, method: {request.method}")
    
    conversation = db.session.query(Conversation).filter_by(
        id=conversation_id,
        user_id=user.username
    ).first()
    
    if not conversation:
        # print(f"DEBUG: Conversation {conversation_id} not found for user {user.username}")
        return jsonify({'error': 'Conversation not found'}), 404
    
    if request.method == 'PUT':
        # Update conversation title
        data = request.get_json()
        # print(f"DEBUG: PUT request data: {data}")
        if 'title' in data:
            new_title = data['title'].strip()
            # print(f"DEBUG: Updating title from '{conversation.title}' to '{new_title}'")
            if new_title:
                conversation.title = new_title
                db.session.commit()
                # print(f"DEBUG: Title updated successfully")
            # else:
            #     print(f"DEBUG: Warning - new title is empty, not updating")
        # else:
        #     print(f"DEBUG: Warning - 'title' not in request data")
        return jsonify(conversation.to_dict())
    
    elif request.method == 'DELETE':
        # Delete conversation and its messages
        try:
            db.session.query(ChatHistory).filter_by(conversation_id=conversation_id).delete()
            db.session.delete(conversation)
            db.session.commit()
            return jsonify({'success': True})
        except Exception as e:
            db.session.rollback()
            return jsonify({'error': f'Failed to delete conversation: {str(e)}'}), 500

@app.route('/api/search')
@login_required
def global_search():
    query = request.args.get('q', '').strip()
    category_id = request.args.get('category', type=int)
    if len(query) < 2:
        return jsonify([])
    
    user = db.session.get(User, session.get('user_id'))
    results = []
    
    # Search conversations
    conversations = db.session.query(Conversation).filter(
        Conversation.user_id == user.username,
        Conversation.title.contains(query)
    ).limit(5).all()
    
    for conv in conversations:
        results.append({
            'type': 'conversation',
            'id': conv.id,
            'title': conv.title,
            'snippet': f"Conversation from {conv.created_at.strftime('%Y-%m-%d')}"
        })
    
    # Search documents
    allowed_collections = get_allowed_collections(user.role)
    document_query = db.session.query(Document).filter(
        Document.collection_name.in_(allowed_collections),
        Document.filename.contains(query)
    )
    
    # Filter by category if provided
    if category_id:
        document_query = document_query.filter(Document.category_id == category_id)
    
    documents = document_query.limit(5).all()
    
    for doc in documents:
        category_name = doc.category.name if doc.category else 'Uncategorized'
        results.append({
            'type': 'document',
            'id': doc.id,
            'title': doc.filename,
            'snippet': f"Document in {doc.collection_name}",
            'category': category_name
        })
    
    return jsonify(results)

@app.route('/api/documents', methods=['GET'])
@login_required
def get_documents():
    """Get documents with optional filtering by category and collection"""
    user = db.session.get(User, session.get('user_id'))
    category_id = request.args.get('category_id', type=int)
    collection_name = request.args.get('collection_name', type=str)
    search_query = request.args.get('q', '').strip()
    
    # Build query
    query = db.session.query(Document)
    
    # Apply role-based collection filtering
    allowed_collections = get_allowed_collections(user.role)
    query = query.filter(Document.collection_name.in_(allowed_collections))
    
    # Filter by category
    if category_id:
        query = query.filter(Document.category_id == category_id)
    
    # Filter by collection
    if collection_name and collection_name in allowed_collections:
        query = query.filter(Document.collection_name == collection_name)
    
    # Search by filename
    if search_query:
        query = query.filter(Document.filename.contains(search_query))
    
    documents = query.order_by(Document.uploaded_at.desc()).all()
    
    return jsonify({
        'documents': [{
            'id': doc.id,
            'filename': doc.filename,
            'collection_name': doc.collection_name,
            'uploaded_by': doc.uploaded_by,
            'uploaded_at': doc.uploaded_at.isoformat() if doc.uploaded_at else None,
            'category': {
                'id': doc.category.id,
                'name': doc.category.name,
                'color': doc.category.color
            } if doc.category else None
        } for doc in documents]
    }), 200

@app.route('/api/conversations/<int:conversation_id>/export')
@login_required
def export_conversation(conversation_id):
    user = db.session.get(User, session.get('user_id'))
    conversation = db.session.query(Conversation).filter_by(
        id=conversation_id,
        user_id=user.username
    ).first_or_404()
    
    format_type = request.args.get('format', 'markdown').lower()
    history = db.session.query(ChatHistory).filter_by(
        conversation_id=conversation_id
    ).order_by(ChatHistory.timestamp.asc()).all()
    
    if format_type == 'markdown':
        content = f"# {conversation.title}\n\n"
        content += f"**Created:** {conversation.created_at.strftime('%Y-%m-%d %H:%M')}\n\n"
        content += "---\n\n"
        
        for item in history:
            content += f"## User\n\n{item.message}\n\n"
            content += f"## Assistant\n\n{item.response}\n\n"
            content += "---\n\n"
        
        return jsonify({
            'content': content,
            'filename': f"{conversation.title.replace(' ', '_')}.md",
            'mime_type': 'text/markdown'
        })
    elif format_type == 'txt':
        content = f"{conversation.title}\n"
        content += f"Created: {conversation.created_at.strftime('%Y-%m-%d %H:%M')}\n"
        content += "=" * 50 + "\n\n"
        
        for item in history:
            content += f"User: {item.message}\n\n"
            content += f"Assistant: {item.response}\n\n"
            content += "-" * 50 + "\n\n"
        
        return jsonify({
            'content': content,
            'filename': f"{conversation.title.replace(' ', '_')}.txt",
            'mime_type': 'text/plain'
        })
    else:
        return jsonify({'error': 'Unsupported format'}), 400


# Saved queries endpoints removed - feature no longer used

@app.route('/api/reset-password', methods=['POST'])
@login_required
def reset_password():
    try:
        data = request.get_json()
        current_password = data.get('current_password')
        new_password = data.get('new_password')
        
        if not current_password or not new_password:
            return jsonify({'error': 'Current password and new password are required'}), 400
        
        if len(new_password) < 6:
            return jsonify({'error': 'New password must be at least 6 characters long'}), 400
        
        user = db.session.get(User, session['user_id'])
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Verify current password
        if not check_password_hash(user.password, current_password):
            return jsonify({'error': 'Current password is incorrect'}), 401
        
        # Update password
        user.password = generate_password_hash(new_password)
        db.session.commit()
        
        return jsonify({'success': True, 'message': 'Password changed successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Failed to reset password: {str(e)}'}), 500

@app.route('/api/documents/<int:document_id>', methods=['PUT', 'DELETE'])
@role_required('admin')
def manage_document(document_id):
    document = db.session.get(Document, document_id)
    if not document:
        return jsonify({'error': 'Document not found'}), 404
    
    if request.method == 'PUT':
        data = request.get_json()
        
        old_filename = document.filename
        old_collection = document.collection_name
        
        new_filename = data.get('filename', document.filename).strip()
        new_collection = data.get('collection_name', document.collection_name)
        reindex = data.get('reindex', False)
        file_updated = data.get('file_updated', False)  # Flag if file was replaced
        
        # Validate collection name
        if new_collection not in COLLECTION_LEVELS:
            return jsonify({'error': 'Invalid collection level'}), 400
        
        # Find the actual file location (might be in wrong folder or uploads root)
        def find_file_location(filename, collection_name):
            """Search for file in correct location, wrong collection folders, or uploads root"""
            # First check correct location
            correct_path = os.path.join('uploads', collection_name, filename)
            if os.path.exists(correct_path):
                return correct_path, collection_name
            
            # Check uploads root
            root_path = os.path.join('uploads', filename)
            if os.path.exists(root_path):
                return root_path, None
            
            # Check all collection folders
            for collection in COLLECTION_LEVELS:
                check_path = os.path.join('uploads', collection, filename)
                if os.path.exists(check_path):
                    return check_path, collection
            
            return None, None
        
        # Find where the file actually is
        actual_filepath, actual_collection = find_file_location(old_filename, old_collection)
        
        if not actual_filepath:
            return jsonify({'error': 'Document file not found'}), 404
        
        # Handle file replacement - re-index the document
        if file_updated and os.path.exists(actual_filepath):
            try:
                # Delete old chunks from ChromaDB
                indexer = Indexer(actual_filepath)
                indexer.delete_document_from_index(level=actual_collection or old_collection, file_path=actual_filepath)
                
                # Re-index the updated document
                indexer.index_document(level=actual_collection or old_collection, replace_existing=True)
                print(f"✅ Re-indexed document: {old_filename} in collection: {actual_collection or old_collection}")
            except Exception as e:
                return jsonify({'error': f'Re-indexing failed: {str(e)}'}), 500
        
        # Determine final filename and collection
        final_filename = new_filename if new_filename else old_filename
        final_collection = new_collection
        
        # Update filename in database if changed
        if new_filename and new_filename != old_filename:
            document.filename = new_filename
        
        # Handle collection change - move file to correct location
        if final_collection != old_collection or actual_collection != final_collection:
            try:
                # Create new folder
                new_folder = os.path.join('uploads', final_collection)
                os.makedirs(new_folder, exist_ok=True)
                
                # Determine target file path
                target_filepath = os.path.join(new_folder, final_filename)
                
                # Check for duplicates in target location - prevent duplicates entirely
                if os.path.exists(target_filepath) and target_filepath != actual_filepath:
                    # Check if another document already exists with this filename in the target collection
                    existing_doc = db.session.query(Document).filter_by(
                        filename=final_filename,
                        collection_name=final_collection
                    ).filter(Document.id != document.id).first()
                    
                    if existing_doc:
                        return jsonify({
                            'error': f'File "{final_filename}" already exists in collection "{final_collection}". Duplicate files are not allowed.'
                        }), 400
                    
                    # If file exists on disk but not in database, it's an orphaned file
                    # We'll still prevent the move to avoid overwriting
                    return jsonify({
                        'error': f'File "{final_filename}" already exists in the target location. Duplicate files are not allowed.'
                    }), 400
                
                # Move file to new location (only if different from current location)
                if target_filepath != actual_filepath:
                    if os.path.exists(actual_filepath):
                        import shutil
                        
                        # Delete from old collection index BEFORE moving (if collection changed)
                        if actual_collection and actual_collection != final_collection:
                            try:
                                indexer = Indexer(actual_filepath)  # Use old path for deletion
                                indexer.delete_document_from_index(level=actual_collection, file_path=actual_filepath)
                                print(f"✅ Deleted document from old collection index: {actual_collection}")
                            except Exception as e:
                                print(f"⚠️ Warning: Could not delete from old index: {e}")
                        
                        # Move the file
                        shutil.move(actual_filepath, target_filepath)
                        print(f"✅ Moved file from {actual_filepath} to {target_filepath}")
                        
                        # Re-index in new collection if reindex is requested or collection changed
                        if reindex or (final_collection != old_collection):
                            try:
                                indexer_new = Indexer(target_filepath)
                                indexer_new.index_document(level=final_collection, replace_existing=True)
                                print(f"✅ Re-indexed document in new collection: {final_collection}")
                            except Exception as e:
                                print(f"⚠️ Warning: Could not re-index in new collection: {e}")
                
                # Update collection name in database
                document.collection_name = final_collection
                
            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                print(f"Error moving file: {error_trace}")
                return jsonify({'error': f'Failed to move file: {str(e)}'}), 500
        elif final_filename != old_filename:
            # Only filename changed, no collection change - just rename in place
            current_folder = os.path.join('uploads', old_collection)
            new_filepath = os.path.join(current_folder, final_filename)
            
            # Check for duplicates - prevent duplicates entirely
            if os.path.exists(new_filepath) and new_filepath != actual_filepath:
                # Check if another document already exists with this filename in the same collection
                existing_doc = db.session.query(Document).filter_by(
                    filename=final_filename,
                    collection_name=old_collection
                ).filter(Document.id != document.id).first()
                
                if existing_doc:
                    return jsonify({
                        'error': f'File "{final_filename}" already exists in collection "{old_collection}". Duplicate files are not allowed.'
                    }), 400
                
                # If file exists on disk but not in database, prevent overwriting
                return jsonify({
                    'error': f'File "{final_filename}" already exists in the target location. Duplicate files are not allowed.'
                }), 400
            
            if os.path.exists(actual_filepath) and not os.path.exists(new_filepath):
                import shutil
                shutil.move(actual_filepath, new_filepath)
                # Re-index with new filename
                try:
                    indexer = Indexer(new_filepath)
                    indexer.delete_document_from_index(level=old_collection, file_path=actual_filepath)
                    indexer.index_document(level=old_collection, replace_existing=True)
                except Exception as e:
                    print(f"⚠️ Warning: Could not re-index after filename change: {e}")
        
        db.session.commit()
        return jsonify({'success': True, 'message': 'Document updated successfully'})
    
    elif request.method == 'DELETE':
        try:
            # Get file path
            folder = os.path.join('uploads', document.collection_name)
            filepath = os.path.join(folder, document.filename)
            
            # Delete from Chroma vector store
            try:
                from client.src.components.retriever import Retriever
                retriever = Retriever(collection_name=document.collection_name)
                vector_store = retriever.get_vector_store()
                
                # Get all documents from the collection
                all_docs = vector_store._collection.get(include=['metadatas'])
                
                # Find document IDs that match the source file path
                ids_to_delete = []
                if all_docs.get('ids') and all_docs.get('metadatas'):
                    # Normalize paths for comparison
                    normalized_filepath = os.path.normpath(filepath)
                    normalized_filename = document.filename
                    
                    for doc_id, metadata in zip(all_docs['ids'], all_docs['metadatas']):
                        # Check if metadata has source matching the filepath
                        # Handle both absolute and relative paths
                        source = metadata.get('source', '') if metadata else ''
                        if source:
                            # Normalize the source path for comparison
                            normalized_source = os.path.normpath(source)
                            # Check if source matches filepath or ends with filename
                            if (normalized_source == normalized_filepath or 
                                normalized_source.endswith(normalized_filename) or
                                os.path.basename(normalized_source) == normalized_filename):
                                ids_to_delete.append(doc_id)
                
                # Delete matching documents from Chroma
                if ids_to_delete:
                    vector_store.delete(ids=ids_to_delete)
                    print(f"✅ Deleted {len(ids_to_delete)} document chunks from Chroma vector store")
                else:
                    print("⚠️ No matching documents found in Chroma vector store")
                    
            except Exception as chroma_error:
                print(f"⚠️ Warning: Failed to delete from Chroma vector store: {str(chroma_error)}")
                # Continue with file and database deletion even if Chroma deletion fails
            
            # Delete physical file if it exists
            if os.path.exists(filepath):
                os.remove(filepath)
            
            # Delete from database
            db.session.delete(document)
            db.session.commit()
            
            return jsonify({
                'success': True, 
                'message': 'Document deleted successfully from database, file system, and vector store.'
            })
        except Exception as e:
            return jsonify({'error': f'Failed to delete document: {str(e)}'}), 500

@app.route('/api/documents/<int:document_id>/reindex', methods=['POST'])
@role_required('admin')
def reindex_document(document_id):
    """Re-index a document in ChromaDB (useful when document file is updated)"""
    document = db.session.get(Document, document_id)
    if not document:
        return jsonify({'error': 'Document not found'}), 404
    
    try:
        # Get file path
        folder = os.path.join('uploads', document.collection_name)
        filepath = os.path.join(folder, document.filename)
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'Document file not found'}), 404
        
        # Delete old chunks and re-index
        indexer = Indexer(filepath)
        indexer.index_document(level=document.collection_name, replace_existing=True)
        
        return jsonify({
            'success': True,
            'message': f'Document "{document.filename}" re-indexed successfully'
        }), 200
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Re-indexing error: {error_trace}")
        return jsonify({'error': f'Re-indexing failed: {str(e)}'}), 500

        # Initialize database and create default users
def init_db():
    with app.app_context():
        try:
            # Create all tables
            db.create_all()
            print("✅ Database tables initialized successfully")
            
            # Migrate existing database: Add session_version column if it doesn't exist
            try:
                from sqlalchemy import inspect, text
                inspector = inspect(db.engine)
                columns = [col['name'] for col in inspector.get_columns('user')]
                if 'session_version' not in columns:
                    # Add session_version column for session invalidation
                    with db.engine.connect() as conn:
                        conn.execute(text('ALTER TABLE user ADD COLUMN session_version INTEGER DEFAULT 0'))
                        conn.commit()
                    print("✅ Added session_version column to user table")
            except Exception as migration_error:
                # Column might already exist or migration not needed
                # This is expected if the column already exists or if using a fresh database
                pass
            
            # Create database indexes for optimization
            if OPTIMIZATIONS_ENABLED:
                try:
                    from utils.db_optimizer import create_indexes
                    # Import models for index creation
                    models = type('Models', (), {
                        'User': User,
                        'Conversation': Conversation,
                        'ChatHistory': ChatHistory,
                        'Document': Document,
                        'UserSettings': UserSettings
                    })
                    create_indexes(db, models)
                    print("✅ Database indexes created successfully")
                except Exception as e:
                    print(f"⚠️ Warning: Could not create indexes: {e}")
        except Exception as e:
            import traceback
            print(f"⚠️ Warning: Database initialization error: {e}")
            print(traceback.format_exc())
            # Try to create only missing tables
            try:
                db.create_all()
            except:
                pass
        
        # Create default users if they don't exist
        default_users = [
            {'username': 'admin', 'email': 'admin@example.com', 'password': 'admin123', 'role': 'admin'}
        ]
        
        for user_data in default_users:
            if not db.session.query(User).filter_by(username=user_data['username']).first():
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
    app.run(debug=True, host='localhost', port=5000)