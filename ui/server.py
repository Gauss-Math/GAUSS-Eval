#!/usr/bin/env python3
"""
Simple HTTP server for Gauss Eval UI that provides environment variable access.
"""

import os
import json
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import sys

class GaussJudgeHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urlparse(self.path)
        
        # Handle API endpoint for environment variables
        if parsed_path.path == '/api/env':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            # Get GAUSS_EVAL_ROOT environment variable
            gauss_root = os.environ.get('GAUSS_EVAL_ROOT', '')
            
            response = {
                'GAUSS_EVAL_ROOT': gauss_root,
                'detected': bool(gauss_root),
                'cwd': os.getcwd()
            }
            
            self.wfile.write(json.dumps(response).encode())
            return
        
        # Handle regular file serving
        super().do_GET()
    
    def do_POST(self):
        parsed_path = urlparse(self.path)
        
        # Handle API endpoint for saving files
        if parsed_path.path == '/api/save':
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode('utf-8'))
                
                gauss_root = os.environ.get('GAUSS_EVAL_ROOT', '')
                if not gauss_root:
                    self.send_response(400)
                    self.send_header('Content-type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps({'error': 'GAUSS_EVAL_ROOT not set'}).encode())
                    return
                
                filename = data.get('filename', '')
                content = data.get('content', '')
                
                if not filename:
                    self.send_response(400)
                    self.send_header('Content-type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps({'error': 'Filename required'}).encode())
                    return
                
                # Save file to GAUSS_EVAL_ROOT directory
                file_path = os.path.join(gauss_root, filename)
                
                # Ensure we don't write outside the project directory
                if not os.path.abspath(file_path).startswith(os.path.abspath(gauss_root)):
                    self.send_response(400)
                    self.send_header('Content-type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps({'error': 'Invalid file path'}).encode())
                    return
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                response = {
                    'success': True,
                    'filename': filename,
                    'path': file_path,
                    'message': f'File saved successfully to {file_path}'
                }
                
                self.wfile.write(json.dumps(response).encode())
                
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(e)}).encode())
        else:
            self.send_response(404)
            self.end_headers()

def run_server(port=8000):
    """Run the server on the specified port."""
    server_address = ('', port)
    httpd = HTTPServer(server_address, GaussJudgeHandler)
    
    # Check if GAUSS_EVAL_ROOT is set
    gauss_root = os.environ.get('GAUSS_EVAL_ROOT', '')
    if gauss_root:
        print(f"✅ GAUSS_EVAL_ROOT detected: {gauss_root}")
    else:
        print("⚠️  GAUSS_EVAL_ROOT not set. Please run:")
        print("   export GAUSS_EVAL_ROOT=/path/to/gauss-judge")
        print("   uv sync")
    
    print(f"🚀 Starting Gauss Eval UI server on http://localhost:{port}")
    print(f"📁 Serving from: {os.getcwd()}")
    print(f"🌐 Open: http://localhost:{port}/index.html")
    print("Press Ctrl+C to stop the server")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
        httpd.server_close()

if __name__ == '__main__':
    port = 8000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print("Invalid port number. Using default port 8000.")
    
    run_server(port)
