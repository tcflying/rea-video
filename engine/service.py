"""HTTP API and page for the film queue. Bind to localhost unless REA_TOKEN is set."""
import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import jobs

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / 'engine' / 'web' / 'index.html'
TOKEN = os.environ.get('REA_TOKEN', '')
HOST = os.environ.get('REA_HOST', '127.0.0.1')
PORT = int(os.environ.get('REA_PORT', '8099'))


def tail(path, limit=8000):
    p = Path(path)
    if not p.is_file():
        return ''
    with open(p, 'rb') as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - limit))
        return f.read().decode('utf-8', 'replace')


def health():
    queue = jobs._h3('/queue')
    busy = None
    if queue is not None:
        busy = len(queue.get('queue_running', [])) + len(queue.get('queue_pending', []))
    return {'queue': len(jobs.listing()), 'comfyui': queue is not None, 'comfyui_busy': busy,
            'waiting': sum(1 for j in jobs.listing() if j['state'] == 'queued'),
            'running': [j['id'] for j in jobs.listing() if j['state'] == 'running']}


class Handler(BaseHTTPRequestHandler):
    server_version = 'rea-video/1.0'

    def log_message(self, fmt, *args):
        print('%s %s' % (self.address_string(), fmt % args), flush=True)

    def _send(self, code, body, kind='application/json', extra=None):
        self.send_response(code)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(body)))
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != 'HEAD':
            self.wfile.write(body)

    def _json(self, code, payload):
        self._send(code, json.dumps(payload, ensure_ascii=False).encode('utf-8'))

    def _body(self):
        size = int(self.headers.get('Content-Length') or 0)
        return json.loads(self.rfile.read(size) or b'{}')

    def _allowed(self):
        return not TOKEN or self.headers.get('Authorization') == 'Bearer ' + TOKEN

    def do_GET(self):
        if not self._allowed():
            return self._json(401, {'error': 'token required'})
        path = self.path.split('?')[0]
        if path in ('/', '/index.html'):
            return self._send(200, PAGE.read_bytes(), 'text/html; charset=utf-8')
        if path == '/api/health':
            return self._json(200, health())
        if path == '/api/jobs':
            return self._json(200, {'jobs': jobs.listing()})
        match = re.fullmatch(r'/api/jobs/([0-9a-f]{12})(/log)?', path)
        if match:
            job = jobs.get(match.group(1))
            if not job:
                return self._json(404, {'error': 'no such job'})
            if match.group(2):
                return self._json(200, {'log': tail(job['log'])})
            return self._json(200, {'job': job})
        if path == '/api/media':
            return self._media()
        return self._json(404, {'error': 'unknown route'})

    def _media(self):
        query = self.path.partition('?')[2]
        job_id = next((p.split('=', 1)[1] for p in query.split('&') if p.startswith('id=')), '')
        job = jobs.get(job_id)
        if not job:
            return self._json(404, {'error': 'no such job'})
        film = (Path(job['out']) / 'remake' / 'remake.mp4').resolve()
        if not film.is_file():
            return self._json(404, {'error': 'film not rendered yet'})
        size = film.stat().st_size
        start, end = 0, size - 1
        header = self.headers.get('Range')
        if header:
            m = re.fullmatch(r'bytes=(\d*)-(\d*)', header.strip())
            if m:
                start = int(m.group(1) or 0)
                end = min(int(m.group(2) or size - 1), size - 1)
        length = end - start + 1
        self.send_response(206 if header else 200)
        self.send_header('Content-Type', 'video/mp4')
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Content-Length', str(length))
        if header:
            self.send_header('Content-Range', 'bytes %d-%d/%d' % (start, end, size))
        self.end_headers()
        with open(film, 'rb') as f:
            f.seek(start)
            left = length
            while left > 0:
                chunk = f.read(min(262144, left))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)

    def do_POST(self):
        if not self._allowed():
            return self._json(401, {'error': 'token required'})
        path = self.path.split('?')[0]
        try:
            body = self._body()
            if path == '/api/jobs':
                job = jobs.submit(body['source'], body['photos'], body['out'],
                                  body.get('narration', True), body.get('skip_gate', False))
                return self._json(201, {'job': job})
            match = re.fullmatch(r'/api/jobs/([0-9a-f]{12})/(cancel|resume|reorder)', path)
            if not match:
                return self._json(404, {'error': 'unknown route'})
            job_id, action = match.groups()
            if action == 'cancel':
                return self._json(200, {'job': jobs.cancel(job_id)})
            if action == 'reorder':
                return self._json(200, {'job': jobs.reorder(job_id, body.get('before'))})
            result = jobs.resume(job_id, bool(body.get('confirm')))
            return self._json(200, result)
        except KeyError as exc:
            return self._json(404, {'error': 'not found: %s' % exc})
        except (ValueError, OSError) as exc:
            return self._json(400, {'error': str(exc)})


def main():
    import threading

    jobs.mark_interrupted()
    thread = threading.Thread(target=jobs.worker, daemon=True)
    thread.start()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print('rea-video service on http://%s:%d' % (HOST, PORT), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
