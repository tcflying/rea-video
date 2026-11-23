"""Run rea-video's source analysis, product swap plan and local H3 remake."""
import argparse
from collections import Counter
import json
import logging
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from PIL import Image, ImageDraw, ImageOps
import shutil
import time
from urllib.request import ProxyHandler, Request, build_opener
from uuid import uuid4

from auto_plan import (FPS, INPUT, MODEL_LENGTH, NO_FASTENING_BAN, PACKAGING_BAN, ROOT, VALID, LocalOnly, apply_approvals, cmd,
                       dump, ff, graph_for, inspect_images, inventory, jpeg, plan_action, text)
from local_vision import MODEL, local_vision

ENGINE = ROOT / 'engine'
VISION_PORT = int(os.environ.get('REA_VISION_PORT') or 18095)


def configure():
    os.environ['REA_VIDEO_HOME'] = str(ENGINE / 'rea-home')
    os.environ['REA_VIDEO_MODELS'] = str(ENGINE / 'rea-models')
    os.environ['NO_PROXY'] = os.environ['no_proxy'] = '127.0.0.1,localhost'
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def analyze(source, out, port):
    from rea_video import pipeline
    from rea_video.media import video_info
    from rea_video.vlm import VLMClient

    class LocalFrames(VLMClient):
        def ask_json(self, prompt, media, max_tokens=4096):
            with TemporaryDirectory() as scratch:
                frames = []
                for i, (path, mime) in enumerate(media):
                    image = Path(scratch) / f'{i}.jpg'
                    if mime.startswith('video/'):
                        duration = video_info(path).duration
                        tiles = Image.new('RGB', (960, 750), 'white')
                        draw = ImageDraw.Draw(tiles)
                        for j in range(12):
                            at = duration * (j + 0.5) / 12
                            frame = Path(scratch) / 'frame.jpg'
                            ff('-y', '-ss', str(at), '-i', path, '-frames:v', '1', frame)
                            with Image.open(frame) as original:
                                tile = ImageOps.contain(original, (150, 338))
                                x, y = (j % 6) * 160, (j // 6) * 375
                                tiles.paste(tile, (x + (160 - tile.width) // 2, y))
                                draw.text((x + 8, y + 344), f'{at:.2f}s', fill='black')
                        tiles.save(image, quality=86)
                        prompt += '\nThe attached contact sheet samples the video chronologically, left to right, top to bottom. Use the printed times. Audio is unavailable in this view.'
                    else:
                        with Image.open(path) as original:
                            ImageOps.contain(original.convert('RGB'), (580, 580)).save(image, quality=88)
                    frames.append((image, 'image/jpeg'))
                return super().ask_json(prompt, frames, min(max_tokens, 4000))

    # rea-video retains its review and classification logic. This adapter only changes wire media.
    pipeline.VLMClient = LocalFrames
    with local_vision(port, MODEL) as url:
        os.environ.update(REA_VLM_BASE_URL=url, REA_VLM_API_KEY='local',
                          REA_VLM_MODEL=MODEL, REA_VLM_MEDIA='frames', REA_VLM_TIMEOUT='240')
        result = pipeline.decompose(source, out, use_vlm=True, workers=1, lang='zh')
    if not result['analysis']['vlm']['used']:
        raise RuntimeError(f"rea-video visual analysis failed: {result['analysis']['vlm']}")
    print(json.dumps({'shots': len(result['shots']), 'vlm': result['analysis']['vlm']}, ensure_ascii=False), flush=True)


LOCAL_H3 = 'minimax-h3-local'
H3_URL = os.environ.get('REA_ENGINE_URL') or 'http://127.0.0.1:1022'


def register_local_h3():
    from rea_video.plan import Planner, user_capabilities

    planner = Planner()
    model = {
        'id': LOCAL_H3, 'name': 'MiniMax H3 Local', 'category': 'video', 'status': 'ga',
        'primitives': ['video.t2v', 'video.multi_reference', 'video.reference_video', 'video.multi_shot'],
        'limits': {'duration_s': {'min': VALID / FPS, 'max': VALID / FPS},
                   'ref_video_s': {'max': VALID / FPS}, 'refs': {'image': 3, 'video': 1}},
        'planning': {'strategy': 'remix'},
        'prompting': planner.prompting['minimax-h3'],
        'notes': 'Local Director ref2va 4-step graph at 24fps and 124 frames. '
                 'Use up to 120 timeline frames per call; hold the final reference frame for padding.'
    }
    path = user_capabilities() / 'models' / (LOCAL_H3 + '.yaml')
    path.parent.mkdir(parents=True, exist_ok=True)
    dump(path, {'models': [model]})


def save_plan(out, photos):
    from rea_video.generation import save_intent
    from rea_video.pipeline import finalize
    from rea_video.util import read_json

    register_local_h3()
    save_intent(out, {'model': LOCAL_H3, 'generate': 'all', 'swaps': [
        {'from': '原片中展示的商品', 'to': f'替换商品视角 {i}', 'image': str(p)}
        for i, p in enumerate(photos, 1)
    ]})
    return finalize(read_json(out / 'breakdown.json'), out, engine=LOCAL_H3)


def frame_count(path):
    return int(cmd('ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
                   '-show_entries', 'stream=nb_read_frames', '-of', 'csv=p=0', path).strip())


def rationale(b, out, photos, port):
    path = out / 'remake' / 'actions.json'
    actions = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}
    expected = {str(s['index'] if isinstance(s, dict) else s) for s in b['shots']}
    stale = set(actions) - expected
    if stale:
        raise RuntimeError('actions.json has rows outside the plan: ' + ', '.join(sorted(stale)) + '; delete remake/actions.json and re-plan')
    if actions and any('absences' not in (row.get('product') or {}) for row in actions.values()):
        raise RuntimeError('actions.json predates product-absence planning; delete remake/actions.json and re-plan')
    from rea_video.remake import frame_bounds

    # Fixed 24fps for frame sampling; the official breakdown and assembly keep native source timing.
    visual = out / 'remake' / 'source24.mp4'
    if not visual.is_file():
        ff('-i', b['source']['path'], '-map', '0:v:0', '-vf', f'fps={FPS}', '-an',
           '-c:v', 'libx264', '-pix_fmt', 'yuv420p', visual)
    total = frame_count(visual)
    target = [item for i, p in enumerate(photos, 1) for item in (text(f'TARGET product photo {i}'), jpeg(p))]
    inv = None
    if expected - set(actions):
        with local_vision(port, MODEL):
            inv = inventory(port, MODEL, target)
            facts = (inv['construction'] + ' Visible parts: ' + inv['parts']).strip()
            for shot, (start, length) in frame_bounds(b['shots'], FPS).items():
                if str(shot) in actions:
                    continue
                start = min(start, total - 1)
                p, samples, kind, source, product = plan_action(visual, start, min(total, start + length), target, facts, port, MODEL, inventory=inv)
                actions[str(shot)] = {**p, 'samples': samples, 'kind': kind, 'source': source, 'product': product}
                dump(path, actions)
                print(f"Shot {shot:03d} {p['decision']}: {p['reason']}", flush=True)
    approvals_path = out / 'remake' / 'approvals.json'
    if approvals_path.is_file():
        absent = inv['absences'] if inv else ''
        if not absent:
            # A resumed run has no live inventory; every row carries the absences it was planned with.
            absent = ', '.join(sorted({part.strip() for row in actions.values()
                                       for part in str((row.get('product') or {}).get('absences', '')).split(',')
                                       if part.strip()}))
        actions = apply_approvals(actions, json.loads(approvals_path.read_text(encoding='utf-8')), absent)
    blocked = [f"{i}: {p['needed_view']}" for i, p in actions.items() if p['decision'] == 'need_material']
    if blocked:
        raise RuntimeError('More evidence needed for shot(s): ' + ' / '.join(blocked))
    return actions


def h3_prompt(unit, actions, nphotos, contract=None):
    tags = ', '.join(f'<Picture {i}>' for i in range(1, nphotos + 1))
    # Merge and split the per-shot lists: one shot may name only 'laces', another 'laces, a box'.
    absent = sorted({part.strip() for brief in unit['brief']
                     for part in str((actions[str(brief['shot'])].get('product') or {}).get('absences', '')).split(',')
                     if part.strip()})
    # The pictures alone did not hold identity: shots 13-14 kept the source's white shoes.
    # State the target construction in the prompt and repeat the constraint on every shot.
    facts = ' '.join(actions[str(unit['shots'][0])].get('target_evidence', '').split())[:320]
    lead = (f'subject_definitions: {tags} show different views of the same replacement product and define its identity.\n'
            'summary: Reproduce the framing, cuts, camera, scene, timing, people and hands of <Video 1>. '
            f'The featured product is always the replacement in {tags}; replace the source product wherever it appears.\n'
            f'product_identity: {facts}\n' +
            (('absent_from_target: ' + ', '.join(absent) +
              f'. The replacement in {tags} has none of these; never render them, nor any box, bag or packaging the pictures do not show.\n') if absent else '') +
            (('product_limits: the replacement has no ' + ', '.join(sorted((contract or {}).get('forbidden', {}).values())) +
              '. No hand ties, buckles, laces, zips, unboxes or handles packaging in any shot; build every action from '
              'the parts visible in the pictures.\n') if (contract or {}).get('forbidden') else '') +
            'retention_analysis: <Video 1> partially_preserved for environment, camera and movement. '
            'Colour, print, sole, closures and openings belong to the replacement in the pictures.\n'
            'sanitised_hands: fingers stay outside the shoe at all times; never place a thumb or finger inside '
            'the knit opening or through the upper.\n'
            'detailed_description: Preserve poses and visible occlusion. Render a clean picture for later titling.')
    if PACKAGING_BAN not in lead:
        lead += '\n' + PACKAGING_BAN
    if NO_FASTENING_BAN not in lead:
        lead += '\n' + NO_FASTENING_BAN

    lines = []
    for i, brief in enumerate(unit['brief'], 1):
        decision = actions[str(brief['shot'])]
        if i == 1:
            when = ''
        else:
            ms = round(brief['at'] * 1000)
            when = f'At {ms // 60000:02d}:{(ms % 60000) / 1000:06.3f}, cut to the next source shot. '
        motion = decision['motion'] if decision['decision'] == 'adapt' else 'Follow this source shot’s visible action and framing.'
        lines.append(f'[Shot {i}] {when}{motion} Carry the replacement product from {tags}; never restore the source product.')
    return '\n'.join([lead, *lines, 'overall_soundscape: Natural ambience matching the scene.', 'non_diegetic_music: N/A.'])


def ban_absences(contract, actions):
    """An analyst-declared absence must forbid that part even when the facts text omits it."""
    from contract import ABSENT_WORDS, parts_in
    declared = ' '.join(str((row.get('product') or {}).get('absences', '')) for row in actions.values())
    forbidden = contract.setdefault('forbidden', {})
    for name in parts_in(declared):
        # ponytail: setdefault keeps a richer word from reconcile; extend only the named parts.
        forbidden.setdefault(name, ABSENT_WORDS.get(name, name))
    return contract


def stage(b, out, photos, actions):
    from rea_video.generation import checks
    from rea_video.integrations.capsule_cinema import ref_clips
    from rea_video.remake import frame_bounds

    plan = b['production_plan']
    units = plan['units']
    bounds = frame_bounds(b['shots'], b['source']['fps'])
    planned = [shot for u in units for shot in u['shots']]
    if plan['model'] != LOCAL_H3 or plan['strategy'] != 'remix':
        raise RuntimeError(f"Expected local H3 remix plan, got {plan['model']} / {plan['strategy']}")
    if not units or sorted(planned) != sorted(bounds):
        raise RuntimeError(f'H3 plan does not cover every shot exactly once: {sorted(set(bounds) - set(planned))}')
    if set(actions) != {str(shot) for shot in bounds}:
        raise RuntimeError(f'actions.json does not match the shot plan: expected {sorted(str(s) for s in bounds)}, '
                           f'got {sorted(actions)}')
    from contract import reconcile
    facts = next((row.get('target_evidence') for row in actions.values() if row.get('target_evidence')), '')
    contract = reconcile(actions, facts) if facts else {'forbidden': {}, 'product': {'parts': []}}
    contract = ban_absences(contract, actions)
    dump(out / 'remake' / 'actions.json', actions)
    dump(out / 'remake' / 'contract.json', contract)
    prompts = {u['id']: h3_prompt(u, actions, len(photos), contract) for u in units}
    audit = checks(plan, b['intent'], prompts, plan.get('prompting'), sum(u['duration'] for u in units), out)
    errors = [c['msg'] for c in audit if c['level'] == 'error']
    if errors:
        raise RuntimeError('rea-video plan validation failed: ' + ' / '.join(errors))
    dump(out / 'remake' / 'prompts.json', prompts)
    dump(out / 'remake' / 'h3_checks.json', audit)
    run = 'rea-' + uuid4().hex[:16]
    stage_dir = INPUT / run
    stage_dir.mkdir(parents=True)
    graph_dir = out / 'remake' / 'graphs'
    graph_dir.mkdir(exist_ok=True)
    photo_refs = []
    for i, photo in enumerate(photos, 1):
        path = stage_dir / f'product{i}.png'
        ff('-i', photo, '-frames:v', '1', path)
        photo_refs.append(path.relative_to(INPUT).as_posix())
    template = json.loads((ROOT / 'armA_seg01_graph.json').read_text(encoding='utf-8'))
    jobs = []
    for u in units:
        uid = u['id']
        if not re.fullmatch(r'G[1-9][0-9]*', uid):
            raise ValueError(f'Invalid unit ID: {uid}')
        refs = ref_clips(b, out, u)
        calls = u['calls']
        if set(refs) != {(i, '@视频1') for i in range(1, calls + 1)}:
            raise RuntimeError(f'{uid} requires one reference video per H3 call')
        frames = sum(bounds[shot][1] for shot in u['shots'])
        native_fps = b['source']['fps']
        for i in range(1, calls + 1):
            key = uid if calls == 1 else f'{uid}-{i}'
            part_frames = frames * i // calls - frames * (i - 1) // calls
            expected = max(1, round(part_frames * FPS / native_fps))
            if expected > VALID:
                raise RuntimeError(f'{key} exceeds the H3 timeline frame budget: {expected}')
            clip = stage_dir / f'{key}.mp4'
            src = refs[i, '@视频1']
            ff('-i', src, '-an', '-vf', f'fps={FPS},trim=end_frame={expected},'
               f'tpad=stop_mode=clone:stop={MODEL_LENGTH}', '-frames:v', str(MODEL_LENGTH),
               '-c:v', 'libx264', '-pix_fmt', 'yuv420p', clip)
            if frame_count(clip) != MODEL_LENGTH or not 0 < expected <= MODEL_LENGTH:
                raise RuntimeError(f'{key} has invalid H3 reference length: {expected}')
            decision = actions[str(u['shots'][0])]
            p = {k: decision[k] for k in ('decision', 'source_action', 'target_evidence', 'motion', 'reason', 'needed_view')}
            graph = graph_for(template, p, clip.relative_to(INPUT).as_posix(), photo_refs, f'{run}/{key}')
            graph['42']['inputs']['prompt'] = prompts[uid]
            path = graph_dir / f'{key}.json'
            dump(path, graph)
            jobs.append({'key': key, 'unit': uid, 'part': i, 'parts': calls, 'shots': u['shots'],
                         'trim_to': part_frames / native_fps, 'h3_frames': expected, 'graph': str(path)})
    dump(out / 'remake' / 'h3_jobs.json', jobs)
    print(f'Prepared {len(jobs)} H3 calls for {len(planned)} shots', flush=True)


def h3_json(path, body=None):
    req = Request(H3_URL + path, None if body is None else json.dumps(body).encode(),
                  headers={'Content-Type': 'application/json'})
    with build_opener(ProxyHandler({}), LocalOnly()).open(req, timeout=30) as response:
        return json.load(response)


def run_graph(graph, record):
    state = json.loads(record.read_text(encoding='utf-8')) if record.is_file() else {}
    pid = state.get('prompt_id')
    history = state.get('history')
    if history and history['status']['status_str'] != 'success':
        record.replace(record.with_name(f'{record.stem}.{pid}.failed.json'))
        state, pid = {}, None
    if pid and not state.get('history'):
        history = h3_json('/history/' + pid).get(pid)
        if not history:
            queue = h3_json('/queue')
            if not any(row[1] == pid for key in ('queue_running', 'queue_pending') for row in queue[key]):
                pid = None
    if not pid:
        # Director's AWQ encoder fits the standard Windows checkpoint mapping path.
        graph['2']['inputs']['clip_name'] = os.environ.get('REA_CLIP_NAME') or 'qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors'
        pid = h3_json('/prompt', {'prompt': graph, 'client_id': 'rea-video-local'})['prompt_id']
        state = {'prompt_id': pid}
        dump(record, state)
        print(f'{record.stem} submitted {pid}', flush=True)
    deadline = time.monotonic() + 1800
    while not state.get('history') and time.monotonic() < deadline:
        state['history'] = h3_json('/history/' + pid).get(pid)
        if state['history']:
            dump(record, state)
        else:
            time.sleep(5)
    history = state.get('history')
    if not history:
        raise TimeoutError(f'{record.stem} still pending: {pid}')
    if history['status']['status_str'] != 'success':
        errors = [info for name, info in history['status']['messages'] if name == 'execution_error']
        raise RuntimeError(f'{record.stem} H3 failed: {errors[0].get("exception_message") if errors else history["status"]}')
    node = history['outputs']['7']
    entry, = [x for rows in node.values() if isinstance(rows, list) for x in rows
              if isinstance(x, dict) and x.get('type') == 'output' and x.get('filename', '').endswith('.mp4')]
    root = (ENGINE / 'output').resolve()
    raw = (root / entry.get('subfolder', '') / entry['filename']).resolve()
    if not raw.is_relative_to(root) or frame_count(raw) != MODEL_LENGTH:
        raise RuntimeError(f'Unexpected H3 media: {raw}')
    return raw


def assemble_exact(breakdown, results, out_dir, fps):
    from unittest.mock import patch

    from rea_video import remake

    original_ff = remake._ff

    def encode(args, cwd=None):
        if args[:2] == ['-f', 'concat']:
            # Color metadata can vary between shots. Keep the filter running so
            # its frame counter continues across cuts on the source frame grid.
            args = ['-reinit_filter', '0', *args[:-3], '-vf', f'settb=1/{fps},setpts=N',
                    '-r', str(fps), '-fps_mode', 'cfr', '-c:v', 'libx264', '-crf', '18',
                    '-pix_fmt', 'yuv420p', args[-1]]
        # Source audio can end before its final picture. Preserve both streams.
        original_ff([arg for arg in args if arg != '-shortest'], cwd)

    with patch.object(remake, '_ff', encode):
        return json.loads(remake.assemble(breakdown, results, out_dir=out_dir, overlays=True))


def results_from_clips(jobs, records):
    results = {}
    for job in jobs:
        key = job['key']
        clip = records / f'{key}.mp4'
        if not clip.is_file() or frame_count(clip) != job['h3_frames']:
            raise RuntimeError(f'{key} clip missing or mistimed: {clip}')
        results[key] = {k: job[k] for k in ('unit', 'part', 'parts', 'shots', 'trim_to')}
        results[key]['video'] = str(clip)
    return results


def finish(out, gate=True, narration=True):
    """Assemble the rendered clips, put the new narration in, then run the gate."""
    from rea_video.util import read_json

    b = read_json(out / 'breakdown.json')
    remake = out / 'remake'
    jobs = read_json(remake / 'h3_jobs.json')
    dump(remake / 'results.json', {'results': results_from_clips(jobs, remake / 'history')})
    fps = b['source']['fps']
    log = assemble_exact(out / 'breakdown.json', remake / 'results.json', remake, fps)
    if log['legend'] != [[s['index'], 'generated'] for s in b['shots']]:
        raise RuntimeError('rea-video assembly fell back to non-H3 footage')
    final = Path(log['remake'])
    source_frames = round(b['source']['duration'] * fps)
    for media in (Path(log['compare']), final):
        found = frame_count(media)
        audio = cmd('ffprobe', '-v', 'error', '-select_streams', 'a:0', '-show_entries', 'stream=codec_name',
                    '-of', 'csv=p=0', media).strip()
        if found != source_frames or bool(audio) != b['source']['has_audio']:
            raise RuntimeError(f'{media.name}: {found}/{source_frames} frames, audio={audio.decode()}')
    film = remake / 'remake.mp4'
    if final != film:
        shutil.copy2(final, film)
    print(json.dumps({'film': str(film), 'frames': source_frames, 'shots': len(b['shots']),
                      'calls': len(jobs),
                      'actions': Counter(p['decision'] for p in read_json(remake / 'actions.json').values()),
                      'audio': audio.decode(), 'overlays': log.get('overlays')}, ensure_ascii=False), flush=True)
    bed = h3_bed(remake, jobs)
    if narration:
        from narration import mix, prepare

        voice = prepare(out)
        if voice:
            if bed:
                mix(film, voice, bed, remake)
            else:
                from narration import apply_track
                apply_track(film, voice, remake)
            print(json.dumps({'narration': str(voice), 'bed': str(bed) if bed else None,
                              'voice': 'piper ' + 'en_US-lessac-medium'}), flush=True)
    if not gate:
        print('Acceptance gate skipped on request: this film is not accepted.', flush=True)
        return
    from check_film import check

    report = check(out)
    print(json.dumps({'acceptance': report['verdict'],
                      'claim_conflicts': report['claims'].get('colour_conflicts', []),
                      'review_missing': report['review']['missing'],
                      'review_failed': report['review']['failed'],
                      'audio': report['audio'].get('status'),
                      'report': str(remake / 'acceptance.json')}, ensure_ascii=False), flush=True)
    if report['verdict'] != 'accepted':
        raise RuntimeError('Film rejected by the acceptance gate; see remake/acceptance.json')


def render(out, port=VISION_PORT, gate=True, narration=True, only=None):
    from rea_video.util import read_json

    b = read_json(out / 'breakdown.json')
    remake = out / 'remake'
    jobs = read_json(remake / 'h3_jobs.json')
    if only:
        wanted = {k.strip() for k in only if k.strip()}
        jobs = [j for j in jobs if j['key'] in wanted]
        unknown = wanted - {j['key'] for j in jobs}
        if unknown:
            raise RuntimeError(f'Unknown H3 call: {sorted(unknown)}')
    records = remake / 'history'
    records.mkdir(exist_ok=True)
    for job in jobs:
        key = job['key']
        graph = read_json(Path(job['graph']))
        raw = run_graph(graph, records / f'{key}.json')
        clip = records / f'{key}.mp4'
        # Keep the shot sound H3 generated; the mix decides how loud it sits under the voice.
        ff('-y', '-i', raw, '-frames:v', str(job['h3_frames']), '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
           '-c:a', 'aac', '-b:a', '192k', '-ar', '48000', clip)
        if frame_count(clip) != job['h3_frames']:
            raise RuntimeError(f'{key} trim verification failed')
        print(f'{key} rendered {job["h3_frames"]} timeline frames', flush=True)
    if only:
        print('Rendered a subset; run finish to assemble the film.', flush=True)
        return
    finish(out, gate, narration)


def h3_bed(remake, jobs):
    """Join the audio H3 generated per call into one bed that follows the picture cuts."""
    inputs, filters = [], []
    for i, job in enumerate(jobs):
        clip = remake / 'history' / f'{job["key"]}.mp4'
        if not clip.is_file():
            return None
        inputs += ['-i', str(clip)]
        filters.append(f'[{i}:a]atrim=0:{job["trim_to"]:.3f},asetpts=N/SR/TB[a{i}]')
    if not filters:
        return None
    graph = ';'.join(filters) + ';' + ''.join(f'[a{i}]' for i in range(len(jobs)))
    graph += f'concat=n={len(jobs)}:v=0:a=1,aresample=48000[out]'
    path = remake / 'h3-audio.wav'
    cmd('ffmpeg', '-nostdin', '-v', 'error', '-y', *inputs, '-filter_complex', graph,
        '-map', '[out]', '-c:a', 'pcm_s16le', path)
    return path if path.is_file() and path.stat().st_size > 4096 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='stage', required=True)
    for name in ('run', 'analyze', 'plan', 'render', 'finish'):
        p = sub.add_parser(name)
        p.add_argument('out', type=Path)
        if name in ('run', 'analyze'):
            p.add_argument('source', type=Path)
        if name in ('run', 'plan'):
            p.add_argument('photos', type=Path, nargs='+')
        p.add_argument('--vision-port', type=int, default=VISION_PORT)
        if name in ('render', 'finish'):
            p.add_argument('--skip-gate', action='store_true', help='assemble without acceptance; never reports success')
            p.add_argument('--no-narration', action='store_true', help='keep the source performance')
        if name == 'render':
            p.add_argument('--only', help='comma-separated H3 calls to render; finish later assembles')
            p.add_argument('--no-finish', action='store_true', help='render clips only')
        if name == 'plan':
            p.add_argument('--reprompt', action='store_true',
                           help='rebuild prompts and graphs, superseding this directory H3 history')
    args = parser.parse_args()
    out = args.out.resolve()
    configure()
    if args.stage in ('run', 'plan'):
        photos = [p.resolve() for p in args.photos]
        if not 1 <= len(photos) <= 3 or not all(p.is_file() for p in photos):
            parser.error('Provide 1–3 existing product photos')
    if args.stage in ('run', 'analyze'):
        if not args.source.is_file():
            parser.error('Source video is missing')
        if out.exists():
            from rea_video.util import read_json

            b = read_json(out / 'breakdown.json')
            if Path(b['source']['path']).resolve() != args.source.resolve() or b['analysis']['vlm']['used']:
                parser.error('Use a new output directory or resume the same failed analysis')
        analyze(args.source.resolve(), out, args.vision_port)
    if args.stage in ('run', 'plan'):
        records = out / 'remake' / 'history'
        if getattr(args, 'reprompt', False):
            for record in records.glob('G*.json'):
                record.replace(record.with_name(record.name + '.superseded'))
        for record in records.glob('G*.json'):
            state = json.loads(record.read_text(encoding='utf-8'))
            if (state.get('history') or {}).get('status', {}).get('status_str') != 'error':
                parser.error('This directory has submitted or successful H3 jobs. Use render to resume it.')
        b = save_plan(out, photos)
        actions = rationale(b, out, photos, args.vision_port)
        stage(b, out, photos, actions)
        from rea_video.pipeline import finalize

        finalize(b, out, engine=LOCAL_H3)
    if args.stage in ('run', 'render'):
        only = [k for k in (getattr(args, 'only', None) or '').split(',') if k] or None
        if getattr(args, 'no_finish', False):
            only = only or [j['key'] for j in json.loads((out / 'remake' / 'h3_jobs.json').read_text(encoding='utf-8'))]
        render(out, args.vision_port, not getattr(args, 'skip_gate', False),
               not getattr(args, 'no_narration', False), only)
    if args.stage == 'finish':
        finish(out, not getattr(args, 'skip_gate', False), not getattr(args, 'no_narration', False))


if __name__ == '__main__':
    main()
