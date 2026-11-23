"""Acceptance gate: structural checks, narration audit, mandatory per-shot review.

The 7B local model cannot tell a white lace-up from a pink knit sock-shoe, so product
identity is decided by a reviewer looking at contact sheets, one verdict per shot; the
gate refuses a film whose shots were not all reviewed against that exact file.
"""
import argparse
import array
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory

from PIL import Image, ImageDraw

from auto_plan import dump

GATE_VERSION = 3
COLOURS = ['white', 'black', 'pink', 'grey', 'gray', 'blue', 'red', 'beige', 'green', 'brown', 'purple', 'multicolour']
VERDICTS = ('pass', 'fail')


def frames(path):
    out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
                          '-show_entries', 'stream=nb_read_frames', '-of', 'csv=p=0', str(path)],
                         check=True, capture_output=True, text=True).stdout
    return int(out.strip())


def duration(path):
    out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'a:0', '-show_entries',
                          'stream=duration', '-of', 'csv=p=0', str(path)],
                         check=True, capture_output=True, text=True).stdout.strip()
    return float(out) if out and out != 'N/A' else None


def envelope(path):
    """Loudness envelope at 50 ms resolution, enough to tell the same performance from a new one."""
    raw = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(path), '-map', '0:a:0',
                          '-f', 's16le', '-ar', '48000', '-ac', '1', '-'],
                         check=True, capture_output=True).stdout
    step = 2400
    samples = array.array('h')
    samples.frombytes(raw[:len(raw) // 2 * 2])
    return [math.sqrt(sum(v * v for v in samples[i:i + step]) / step) for i in range(0, len(samples) - step, step)]


def similarity(a, b):
    n = min(len(a), len(b))
    if not n:
        return 0.0
    a, b = a[:n], b[:n]
    ma, mb = sum(a) / n, sum(b) / n
    x = [v - ma for v in a]
    y = [v - mb for v in b]
    den = (sum(v * v for v in x) * sum(v * v for v in y)) ** 0.5
    return sum(p * q for p, q in zip(x, y)) / den if den else 0.0


def resolve_source(b):
    path = Path(b['source']['path'])
    if path.is_file():
        return path
    index = Path('engine/rea-home/sources.json')
    if index.is_file():
        for value in json.loads(index.read_text(encoding='utf-8')).values():
            candidate = Path(value)
            if candidate.is_file() and candidate.name == path.name:
                return candidate
    return path


def mids(bounds):
    return [(shot, start + max(0, (end - start) // 2)) for shot, start, end in bounds]


def samples(bounds):
    """Two frames per shot: a defect late in a shot is invisible in a single mid frame."""
    picks = []
    for shot, start, end in bounds:
        span = max(2, end - start)
        picks.append((shot, start + span // 3))
        picks.append((shot, start + (2 * span) // 3))
    return picks


def sheets(film, bounds, target, columns=11, width=432):
    """Tile two frames per shot into labelled contact sheets a reviewer can judge."""
    target.mkdir(parents=True, exist_ok=True)
    picks = samples(bounds)
    made = []
    for offset in range(0, len(picks), columns * 2):
        chunk = picks[offset:offset + columns * 2]
        with TemporaryDirectory() as scratch:
            images = []
            for _, frame in chunk:
                still = Path(scratch) / ('%03d.jpg' % frame)
                subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-ss', '%.3f' % (frame / 30.0),
                                '-i', str(film), '-frames:v', '1', '-vf', 'scale=%d:-2' % width,
                                '-q:v', '2', str(still)], check=True)
                images.append(Image.open(still).convert('RGB'))
        height = images[0].height + 26
        rows = -(-len(chunk) // columns)
        sheet = Image.new('RGB', (width * columns, height * rows), 'white')
        draw = ImageDraw.Draw(sheet)
        for i, ((shot, frame), image) in enumerate(zip(chunk, images)):
            row, col = divmod(i, columns)
            sheet.paste(image, (col * width, row * height + 26))
            draw.text((col * width + 6, row * height + 8), 'shot %d  frame %d' % (shot, frame), fill='black')
        path = target / ('sheet-%02d-%02d.jpg' % (chunk[0][0], chunk[-1][0]))
        sheet.save(path, quality=88)
        made.append(path)
    return made


def shot_digest(film, start, span):
    """A shot that looks the same keeps its review, even when the film around it changed."""
    still = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-ss', '%.3f' % ((start + span // 2) / 30.0),
                            '-i', str(film), '-frames:v', '1', '-vf', 'scale=160:-2',
                            '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], check=True, capture_output=True).stdout
    return hashlib.sha256(still).hexdigest()


def review_state(remake, film, bounds):
    path = remake / 'review.json'
    saved = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}
    rows = saved.get('shots', {})
    missing, failed, stale = [], [], []
    for shot, start, end in bounds:
        row = rows.get(str(shot))
        digest = shot_digest(film, start, end - start)
        if not row or row.get('verdict') not in VERDICTS:
            missing.append(shot)
        elif row.get('shot_sha256') != digest:
            stale.append(shot)
            missing.append(shot)
        elif row['verdict'] == 'fail':
            failed.append(shot)
    return {'path': str(path), 'reviewed': len(bounds) - len(missing), 'stale': stale,
            'missing': missing, 'failed': failed, 'rows': rows}


def record(remake, film, bounds, entries):
    path = remake / 'review.json'
    saved = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}
    rows = saved.get('shots', {})
    spans = {shot: (start, end - start) for shot, start, end in bounds}
    for entry in entries:
        shot, _, verdict = entry.partition('=')
        verdict, _, note = verdict.partition(':')
        if verdict not in VERDICTS:
            raise SystemExit('Expected <shot>=pass|fail[:note], got %r' % entry)
        shot = int(shot)
        if shot not in spans:
            raise SystemExit('No such shot: %d' % shot)
        rows[str(shot)] = {'verdict': verdict, 'note': note.strip(),
                           'shot_sha256': shot_digest(film, *spans[shot])}
    dump(path, {'gate_version': GATE_VERSION, 'shots': rows})
    return len(rows)


def audit(lines, actions):
    """Deterministic narration audit: the words must describe the product in the pictures."""
    text = ' '.join(str(line) for line in lines)
    words = re.findall(r"[a-z']+", text.lower())
    spoken = [c for c in COLOURS if c in words]
    facts = ' '.join(str(item.get('target_evidence', '')) for item in actions.values()).lower()
    product = [c for c in COLOURS if c in set(re.findall(r"[a-z']+", facts))]
    numbers = sorted(set(re.findall(r'\d+(?:\.\d+)?', text)))
    return {'spoken_colours': spoken, 'product_colours': product,
            'colour_conflicts': [c for c in spoken if c not in product],
            'numbers_in_speech_unverified': [n for n in numbers if n not in facts]}


def contract_check(remake, actions):
    """A shot may only reproduce an action the confirmed product parts support."""
    from contract import parts_in, positive, required

    path = remake / 'contract.json'
    if not path.is_file():
        return {'errors': ['contract.json missing: product compatibility was never applied'], 'absent': []}
    doc = json.loads(path.read_text(encoding='utf-8'))
    present = set(doc.get('product', {}).get('parts', []))
    absent = set(doc.get('absent', []))
    errors = []
    if 'laces' not in absent | present or 'box' not in absent | present:
        errors.append('contract never decided whether the product has laces or packaging')
    for key, row in actions.items():
        missing = required(positive(row.get('source_action', ''))) - present
        if missing and row.get('decision') != 'adapt':
            errors.append('shot %s keeps an action the product cannot do: %s' % (key, sorted(missing)))
        if row.get('decision') == 'adapt':
            invented = parts_in(positive(row.get('motion', ''))) - present
            if invented:
                errors.append('shot %s adaptation invents parts: %s' % (key, sorted(invented)))
    return {'errors': errors, 'absent': sorted(absent), 'present': sorted(present)}


def check(out, photos=None):
    out = Path(out).resolve()
    remake = out / 'remake'
    b = json.loads((out / 'breakdown.json').read_text(encoding='utf-8'))
    actions = json.loads((remake / 'actions.json').read_text(encoding='utf-8'))
    film = remake / 'remake.mp4'
    digest = hashlib.sha256(film.read_bytes()).hexdigest()
    bounds = [(int(s['index']), int(s['start_frame']), int(s['end_frame'])) for s in b['shots']]
    fps = b['source']['fps']
    report = {'gate_version': GATE_VERSION, 'film': str(film), 'film_sha256': digest,
              'frames': frames(film), 'fps': fps, 'shots': len(b['shots']),
              'audio': {}, 'claims': {}, 'review': {}, 'verdict': 'rejected'}
    expected = round(b['source']['duration'] * fps)
    if report['frames'] != expected:
        report['error'] = 'Film has %d frames, the source needs %d' % (report['frames'], expected)
    narration = remake / 'narration.json'
    record = json.loads(narration.read_text(encoding='utf-8')) if narration.is_file() else {}
    spoken = ([row['text'] for row in record['lines']] if record.get('lines')
              else [row.get('text', '') for row in b.get('speech', {}).get('lines', [])])
    report['narration'] = {'origin': record.get('origin'), 'lines': spoken,
                           'asr': (record.get('audit') or {}).get('asr')}
    report['claims'] = audit(spoken, actions)
    audio = {'picture_duration': round(report['frames'] / fps, 3), 'duration': duration(film)}
    source = resolve_source(b)
    if b['source'].get('has_audio') and source.is_file() and audio['duration']:
        strength = similarity(envelope(source), envelope(film))
        audio.update(source=str(source), performance_similarity=round(strength, 4),
                     same_performance=strength >= 0.99,
                     status='source_unchanged' if strength >= 0.99 else 'rewritten')
    else:
        audio.update(status='silent_source', same_performance=False)
    report['audio'] = audio
    report['review'] = review_state(remake, film, bounds)
    report['contract'] = contract_check(remake, actions)
    if report['contract']['errors']:
        report['error'] = '; '.join(report['contract']['errors'])
    audio['h3_bed'] = (remake / 'h3-audio.wav').is_file()
    audio['status'] = ('rewritten+h3_bed' if audio['h3_bed'] and report['narration']['origin']
                       else audio['status'])
    audio_ok = report['audio']['status'] in ('rewritten', 'rewritten+h3_bed', 'silent_source')
    if audio_ok and report['audio']['status'].startswith('rewritten') and not report['narration']['origin']:
        audio_ok = False
    ok = ('error' not in report and not report['claims']['colour_conflicts'] and audio_ok
          and not report['review']['missing'] and not report['review']['failed'])
    report['verdict'] = 'accepted' if ok else 'rejected'
    dump(remake / 'acceptance.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('--sheet', action='store_true', help='write review contact sheets')
    parser.add_argument('--review', nargs='+', metavar='SHOT=pass|fail[:note]')
    args = parser.parse_args()
    out = args.out.resolve()
    remake = out / 'remake'
    b = json.loads((out / 'breakdown.json').read_text(encoding='utf-8'))
    bounds = [(int(s['index']), int(s['start_frame']), int(s['end_frame'])) for s in b['shots']]
    film = remake / 'remake.mp4'
    digest = hashlib.sha256(film.read_bytes()).hexdigest()
    if args.review:
        print('Recorded %d review rows' % record(remake, film, bounds, args.review))
    if args.sheet:
        for path in sheets(film, bounds, remake / 'review'):
            print(path)
    report = check(out)
    print(json.dumps({k: report[k] for k in ('verdict', 'frames', 'shots')}, ensure_ascii=False))
    print(json.dumps(report['claims'], ensure_ascii=False))
    print(json.dumps({k: report['audio'].get(k) for k in ('status', 'performance_similarity', 'duration')}, ensure_ascii=False))
    print(json.dumps({k: report['review'][k] for k in ('reviewed', 'missing', 'failed')}, ensure_ascii=False))
    return 0 if report['verdict'] == 'accepted' else 2


if __name__ == '__main__':
    sys.exit(main())