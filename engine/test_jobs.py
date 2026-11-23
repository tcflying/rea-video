"""State-machine checks for the film queue; no GPU, no network."""
import json
import os
import tempfile
from pathlib import Path

scratch = Path(tempfile.mkdtemp(prefix='rea-queue-'))
os.environ['REA_QUEUE'] = str(scratch / 'queue.json')

import jobs  # noqa: E402  (the queue path comes from the environment)

source = scratch / 'source.mp4'
photo = scratch / 'product.png'
source.write_bytes(b'x')
photo.write_bytes(b'x')

a = jobs.submit(source, [photo], scratch / 'film-a')
b = jobs.submit(source, [photo], scratch / 'film-b')
assert [j['id'] for j in jobs.listing()] == [a['id'], b['id']]

jobs.reorder(b['id'], before=a['id'])
assert [j['id'] for j in jobs.listing()] == [b['id'], a['id']]
jobs.cancel(a['id'])
assert jobs.get(a['id'])['state'] == 'cancelled'
try:
    jobs.reorder(a['id'])
    raise AssertionError('a cancelled job must not be reorderable')
except ValueError:
    pass

doc = json.loads(Path(os.environ['REA_QUEUE']).read_text(encoding='utf-8'))
for job in doc['jobs']:
    if job['id'] == b['id']:
        job['state'] = 'running'
Path(os.environ['REA_QUEUE']).write_text(json.dumps(doc), encoding='utf-8')
assert jobs.mark_interrupted() == 1
assert jobs.get(b['id'])['state'] == 'interrupted'
assert jobs.get(b['id'])['reconciliation']['verdict'] == 'not_started'

preview = jobs.resume(b['id'], confirm=False)
assert preview['resumed'] is False and 'confirm' in preview['reason']
assert jobs.get(b['id'])['state'] == 'interrupted'
assert jobs.resume(b['id'], confirm=True)['resumed'] is True
assert jobs.get(b['id'])['state'] == 'queued'
try:
    jobs.resume(b['id'], confirm=True)
    raise AssertionError('a queued job does not need resume')
except ValueError:
    pass

print('queue checks passed')
