"""Rewrite the film narration for the replacement product and verify it by machine.

The source performance cannot stay: it names the source product ("perfect white sneakers").
So the script is drafted from the product facts, synthesised locally, checked against those
facts byte-for-byte by keyword audit, transcribed back with ASR, and only then muxed in.
"""
import argparse
import difflib
import json
import re
import statistics
import subprocess
import sys
import wave
from pathlib import Path

from auto_plan import dump
from local_vision import MODEL, local_vision

COLOURS = ('white', 'black', 'pink', 'grey', 'gray', 'blue', 'red', 'beige', 'green', 'brown', 'purple')
CLAIM_WORDS = ('machine washable', 'waterproof', 'organic', 'orthopedic', 'leather', 'breathable',
               'non-toxic', 'antibacterial', 'award', 'bestseller', 'doctor', 'pediatrician')
VOICE = 'en_US-lessac-medium'

PROMPT = ('Write the voice-over for a short vertical product video, in English, for the product described below. '
          'Three sentences, 50 to 60 words total, spoken by one warm female voice. '
          'Use only the facts given; never invent a size, material, technology, certification, price, warranty or health claim. '
          'Say what the product is, how a hand puts it on, and why it suits a first walker. '
          'No brand names, no exclamation marks, no stage directions. Return the three sentences only.')


def facts(actions, speech):
    return {
        'product': ' '.join(str(v.get('target_evidence', '')).strip() for v in actions.values() if v.get('target_evidence')),
        'source_lines': [row.get('text', '') for row in speech.get('lines', [])],
    }


def chat(port, instruction):
    from urllib.request import ProxyHandler, Request, build_opener

    class LocalOnly(ProxyHandler):
        def __init__(self):
            super().__init__({})

        def proxy_open(self, req, proxy, type):
            return None

    body = {'model': MODEL, 'temperature': 0.3, 'seed': 20261106, 'max_tokens': 400,
            'messages': [{'role': 'user', 'content': instruction}]}
    req = Request(f'http://127.0.0.1:{port}/v1/chat/completions', data=json.dumps(body).encode(),
                  headers={'Content-Type': 'application/json'})
    with build_opener(ProxyHandler({}), LocalOnly()).open(req, timeout=240) as resp:
        return json.load(resp)['choices'][0]['message']['content'].strip()


def sentences(answer):
    flat = ' '.join(answer.split())
    found = [s.strip() for s in re.split(r'(?<=[.!?])\s+', flat) if s.strip()]
    if not 2 <= len(found) <= 4:
        raise ValueError(f'Expected 2-4 sentences, got {len(found)}: {flat[:200]}')
    return found


def audit(lines, facts_text, absent=()):
    """Every colour, claim and part must already be in the product facts."""
    from contract import PARTS

    words = set(re.findall(r"[a-z']+", ' '.join(lines).lower()))
    product = set(re.findall(r"[a-z']+", facts_text.lower()))
    detail = ' '.join(lines).lower()
    return {'unknown_colours': sorted(c for c in COLOURS if c in words and c not in product),
            'unknown_claims': sorted(c for c in CLAIM_WORDS if c in detail),
            'invented_parts': sorted(name for name in absent
                                     if any(re.search(r'\b' + re.escape(w) + r'\b', detail) for w in PARTS[name])),
            'numbers': sorted(set(re.findall(r'\d+(?:\.\d+)?', detail)))}


def words(text):
    return len(re.findall(r"[A-Za-z']+", text))


def synth(lines, out, scales=None):
    from piper import PiperVoice, SynthesisConfig
    from piper.download_voices import download_voice

    voices = (Path(__file__).resolve().parent / 'voices')
    voices.mkdir(exist_ok=True)
    model = voices / f'{VOICE}.onnx'
    if not model.is_file():
        download_voice(VOICE, voices)
    voice = PiperVoice.load(model)
    scales = scales or [1.05] * len(lines)
    made = []
    for i, (line, scale) in enumerate(zip(lines, scales), 1):
        path = out / f'line{i}.wav'
        with wave.open(str(path), 'wb') as wav:
            voice.synthesize_wav(line, wav, syn_config=SynthesisConfig(length_scale=round(scale, 3)))
        with wave.open(str(path)) as wav:
            made.append({'line': i, 'text': line, 'path': str(path),
                         'seconds': round(wav.getnframes() / wav.getframerate(), 3),
                         'length_scale': round(scale, 3), 'words': words(line)})
    return made


def source_pace(b):
    """Words per second of the actual source track, the tempo the voice has to match."""
    rows = [r for r in b.get('speech', {}).get('lines', [])
            if r.get('text') and r.get('start') is not None and r.get('end')]
    rates = [words(r['text']) / max(0.6, float(r['end']) - float(r['start'])) for r in rows]
    return round(statistics.median(rates), 3) if rates else None


def perform(lines, out, target):
    """Synthesise at the source tempo: length_scale moves duration almost linearly."""
    made = synth(lines, out)
    if not target:
        return made
    for _ in range(2):
        scales = [min(1.3, max(0.8, 1.05 * (row['words'] / target) / row['seconds'])) for row in made]
        made = synth(lines, out, scales)
    return made


def place(made, total, b):
    """Keep the source pause structure: each line starts where the source line started."""
    src = [r for r in b.get('speech', {}).get('lines', [])
           if r.get('text') and r.get('start') is not None]
    if len(src) >= len(made):
        earliest = 0.0
        for i, row in enumerate(made):
            at = max(float(src[i]['start']) + 0.15, earliest)
            row['start'] = round(max(0.0, min(at, total - row['seconds'] - 0.3)), 3)
            earliest = row['start'] + row['seconds'] + 0.35
        return made
    return timeline(made, total)


def mix(film, voice, bed, out):
    """The shot sound H3 generated stays under the rewritten voice, ducked while it speaks."""
    tmp = Path(out) / 'remake.mix.mp4'
    graph = ('[2:a]aresample=48000,volume=-20dB[bed];[1:a]aresample=48000,asplit=2[vo][key];'
             '[bed][key]sidechaincompress=threshold=0.02:ratio=12:attack=5:release=300[duck];'
             '[duck][vo]amix=inputs=2:normalize=0,alimiter=limit=0.95,loudnorm=I=-16:TP=-1.5:LRA=11[aout]')
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', str(film), '-i', str(voice), '-i', str(bed),
                    '-filter_complex', graph, '-map', '0:v:0', '-map', '[aout]', '-c:v', 'copy',
                    '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart', str(tmp)], check=True)
    tmp.replace(film)
    return Path(film)


def timeline(lines, total, gap=1.2):
    """Space the lines over the film, each starting 0.35 s after a shot boundary."""
    span = sum(row['seconds'] for row in lines) + gap * (len(lines) - 1)
    if span > total - 0.6:
        raise RuntimeError(f'Narration needs {span:.2f}s of {total:.2f}s')
    lead = max(0.6, (total - span) / (len(lines) + 1))
    at = lead
    for row in lines:
        row['start'] = round(at, 3)
        at += row['seconds'] + max(gap, lead)
    return lines


def track(lines, total, path):
    filters = []
    inputs = []
    for i, row in enumerate(lines):
        inputs.extend(['-i', row['path']])
        filters.append(f'[{i}:a]adelay={int(row["start"] * 1000)}|{int(row["start"] * 1000)},aresample=48000[a{i}]')
    mix = ''.join(f'[a{i}]' for i in range(len(lines)))
    graph = ';'.join(filters) + f';{mix}amix=inputs={len(lines)}:normalize=0[mix];'
    graph += f'[mix]apad,atrim=0:{total:.3f},loudnorm=I=-16:TP=-1.5:LRA=11[aout]'
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y', *inputs, '-filter_complex', graph,
                    '-map', '[aout]', '-c:a', 'pcm_s16le', str(path)], check=True)
    return path


def transcribe(path):
    import sherpa_onnx

    root = Path(__file__).resolve().parent / 'rea-models' / 'sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17'
    recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=str(root / 'model.int8.onnx'), tokens=str(root / 'tokens.txt'), use_itn=True, num_threads=4)
    raw = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(path), '-f', 's16le',
                          '-ac', '1', '-ar', '16000', '-'], check=True, capture_output=True).stdout
    import array

    samples = array.array('h')
    samples.frombytes(raw[:len(raw) // 2 * 2])
    stream = recognizer.create_stream()
    stream.accept_waveform(16000, [s / 32768.0 for s in samples])
    recognizer.decode_stream(stream)
    return stream.result.text.strip()


def accuracy(script, heard):
    clean = lambda s: re.findall(r"[a-z0-9']+", s.lower())
    a, b = clean(script), clean(heard)
    if not a:
        return 0.0, []
    matcher = difflib.SequenceMatcher(None, a, b)
    missing = [a[i] for tag, i1, i2, j1, j2 in matcher.get_opcodes() if tag in ('delete', 'replace') for i in range(i1, i2)]
    return round(matcher.ratio(), 4), missing


def apply(film, voice_track, out):
    tmp = out / 'remake.with-narration.mp4'
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', str(film), '-i', str(voice_track),
                    '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
                    '-movflags', '+faststart', str(tmp)], check=True)
    tmp.replace(film)
    return film

def prepare(out, port=None, author=None):
    """Draft, synthesise and verify the narration track; return its path."""
    out = Path(out).resolve()
    remake = out / 'remake'
    b = json.loads((out / 'breakdown.json').read_text(encoding='utf-8'))
    actions = json.loads((remake / 'actions.json').read_text(encoding='utf-8'))
    total = round(float(b['source']['duration']), 3)
    known = facts(actions, b.get('speech', {}))
    saved = remake / 'narration.json'
    prior = json.loads(saved.read_text(encoding='utf-8')) if saved.is_file() else {}
    if author:
        lines, origin = [s.strip() for s in author if s.strip()], 'operator'
    elif prior.get('lines') and prior.get('product') == known['product']:
        lines, origin = [row['text'] for row in prior['lines']], 'cached'
    else:
        port = port or 18095
        with local_vision(port, MODEL):
            answer = chat(port, PROMPT + '\n\nProduct facts:\n' + known['product'][:1800])
        lines, origin = sentences(answer), 'drafted'
    doc = remake / 'contract.json'
    absent = json.loads(doc.read_text(encoding='utf-8')).get('absent', []) if doc.is_file() else []
    report = audit(lines, known['product'], absent)
    if (report['unknown_colours'] or report['unknown_claims'] or report['invented_parts']
            or [n for n in report['numbers'] if n not in known['product']]):
        raise RuntimeError(f'Narration claims facts the product does not have: {report}')
    target = source_pace(b)
    made = place(perform(lines, remake, target), total, b)
    track_path = track(made, total, remake / 'narration.wav')
    heard = transcribe(track_path)
    got, missing = accuracy(' '.join(lines), heard)
    report['asr'] = {'text': heard, 'accuracy': got, 'missing_words': missing}
    spoken = sum(row['words'] for row in made)
    speech = sum(row['seconds'] for row in made)
    report['prosody'] = {'source_words_per_s': target,
                         'voice_words_per_s': round(spoken / speech, 3) if speech else None,
                         'source_lines': len(known['source_lines']), 'placed_at': [r['start'] for r in made],
                         'note': 'tempo and pause placement match the source; voice identity is not cloned'}
    dump(saved, {'product': known['product'], 'origin': origin, 'lines': made, 'audit': report,
                 'source_lines': known['source_lines'], 'wav': str(track_path)})
    print(json.dumps({'narration': origin, 'lines': len(made), 'seconds': total,
                      'asr_accuracy': got, 'missing': missing, 'prosody': report['prosody'],
                      'audit': report}, ensure_ascii=False), flush=True)
    if got < 0.85:
        raise RuntimeError(f'ASR mismatch ({got}): {heard[:200]}')
    return track_path


def apply_track(film, wav, out):
    tmp = Path(out) / 'remake.with-narration.mp4'
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', str(film), '-i', str(wav),
                    '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
                    '-movflags', '+faststart', str(tmp)], check=True)
    tmp.replace(film)
    return Path(film)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('--vision-port', type=int, default=18095)
    parser.add_argument('--script', nargs='+', help='operator-authored sentences; skips drafting')
    args = parser.parse_args()
    prepare(args.out, args.vision_port, args.script)


if __name__ == '__main__':
    main()