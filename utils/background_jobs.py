"""
Background job processing for heavy tasks
"""
import os
from datetime import datetime
import logging
from threading import Thread
from queue import Queue
import time

logger = logging.getLogger(__name__)

# Job queue
job_queue = Queue()
job_status = {}

# Check if Celery is available
USE_CELERY = os.getenv('CELERY_BROKER_URL') is not None

if USE_CELERY:
    try:
        from celery import Celery
        celery_app = Celery('rag_chatbot',
                           broker=os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0'),
                           backend=os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0'))
        celery_app.conf.update(
            task_serializer='json',
            accept_content=['json'],
            result_serializer='json',
            timezone='UTC',
            enable_utc=True,
        )
    except ImportError:
        USE_CELERY = False
        logger.warning("Celery not available, using thread-based jobs")

def process_document_indexing(document_id, file_path, collection_name, replace_existing=False):
    """Process document indexing in background"""
    if USE_CELERY:
        return index_document_task.delay(document_id, file_path, collection_name, replace_existing)
    else:
        # Use thread-based processing
        thread = Thread(target=_index_document_thread, args=(document_id, file_path, collection_name, replace_existing))
        thread.daemon = True
        thread.start()
        return {'status': 'started', 'job_id': f'thread_{document_id}'}

def _index_document_thread(document_id, file_path, collection_name, replace_existing):
    """Thread-based document indexing"""
    try:
        job_id = f'index_{document_id}'
        job_status[job_id] = {'status': 'processing', 'progress': 0, 'started_at': datetime.utcnow()}
        
        from client.src.components.indexer import Indexer
        indexer = Indexer()
        
        # Update progress
        job_status[job_id]['progress'] = 25
        job_status[job_id]['message'] = 'Loading document...'
        
        # Index document
        job_status[job_id]['progress'] = 50
        job_status[job_id]['message'] = 'Indexing document...'
        
        indexer.index_document(collection_name, file_path, replace_existing=replace_existing)
        
        job_status[job_id]['progress'] = 100
        job_status[job_id]['status'] = 'completed'
        job_status[job_id]['message'] = 'Indexing completed'
        job_status[job_id]['completed_at'] = datetime.utcnow()
        
    except Exception as e:
        logger.error(f"Error indexing document {document_id}: {str(e)}")
        job_status[job_id]['status'] = 'failed'
        job_status[job_id]['error'] = str(e)
        job_status[job_id]['completed_at'] = datetime.utcnow()

def get_job_status(job_id):
    """Get status of a background job"""
    if USE_CELERY:
        task = celery_app.AsyncResult(job_id)
        return {
            'status': task.state,
            'progress': task.info.get('progress', 0) if task.info else 0,
            'message': task.info.get('message', '') if task.info else '',
        }
    else:
        return job_status.get(job_id, {'status': 'not_found'})

if USE_CELERY:
    @celery_app.task(bind=True)
    def index_document_task(self, document_id, file_path, collection_name, replace_existing=False):
        """Celery task for document indexing"""
        try:
            self.update_state(state='PROCESSING', meta={'progress': 0, 'message': 'Starting...'})
            
            from client.src.components.indexer import Indexer
            indexer = Indexer()
            
            self.update_state(state='PROCESSING', meta={'progress': 25, 'message': 'Loading document...'})
            self.update_state(state='PROCESSING', meta={'progress': 50, 'message': 'Indexing document...'})
            
            indexer.index_document(collection_name, file_path, replace_existing=replace_existing)
            
            self.update_state(state='SUCCESS', meta={'progress': 100, 'message': 'Indexing completed'})
            return {'status': 'completed', 'document_id': document_id}
            
        except Exception as e:
            logger.error(f"Error in indexing task: {str(e)}")
            self.update_state(state='FAILURE', meta={'error': str(e)})
            raise

