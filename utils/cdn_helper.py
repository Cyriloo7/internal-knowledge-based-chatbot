"""
CDN integration helper for static assets
"""
import os
from flask import url_for

def get_cdn_url(path):
    """Get CDN URL for a static asset"""
    cdn_domain = os.getenv('CDN_DOMAIN', '')
    use_cdn = os.getenv('USE_CDN', 'False').lower() == 'true'
    
    if use_cdn and cdn_domain:
        # Remove leading slash if present
        path = path.lstrip('/')
        # Ensure CDN domain doesn't have trailing slash
        cdn_domain = cdn_domain.rstrip('/')
        return f"{cdn_domain}/{path}"
    else:
        # Use local URL
        return url_for('static', filename=path)

def get_static_url(filename):
    """Get URL for static file (CDN or local)"""
    return get_cdn_url(f"static/{filename}")

