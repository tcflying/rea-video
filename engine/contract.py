"""Decide which source actions are physically possible on the replacement product.

A local model describes what it sees; this module keeps only the parts an action actually
operates, compares them with the parts the pictures confirm, and rewrites impossible actions
using present parts only. Laces, straps and packaging therefore cannot be invented.
"""
import json
import re
from pathlib import Path

PARTS = {
    'laces': ('lace', 'laces', 'shoelace', 'shoelaces'),
    'strap': ('strap', 'straps', 'velcro', 'hook and loop', 'fastening', 'fastener'),
    'buckle': ('buckle', 'buckles'),
    'zipper': ('zipper', 'zippers', 'zip', 'zips'),
    'tongue': ('tongue',),
    'box': ('box', 'boxes', 'packaging', 'carton', 'package', 'packaged'),
    'insert': ('insert', 'insole', 'footbed'),
    'sole': ('sole', 'soles', 'outsole', 'tread'),
    'opening': ('opening', 'collar', 'cuff', 'slip on', 'slip-on', 'slipping on'),
}
OPERATE = ('tie|ties|tied|tying|fasten|fastens|fastened|buckle|buckles|buckled|lace|laces|laced|lacing|'
           'zip|zips|zipped|open|opens|opened|close|closes|closed|pull|pulls|pulled|pulling|tug|tugs|'
           'tugging|adjust|adjusts|adjusting|manipulate|manipulates|manipulating|lift|lifts|lifting|'
           'remove|removes|removing|insert|inserts|inserting|press|presses|pressing|push|pushes|pushing|'
           'slide|slides|sliding|hook|hooks|hooking|peel|peels|unwrap|unwraps|unbox|unboxes|'
           'take out|takes out|taking out|grip|grips|gripping|hold|holds|holding|touch|touches|touching')
CANON = ('laces', 'strap', 'buckle', 'zipper', 'tongue', 'box', 'insert', 'sole', 'opening')
ABSENT_WORDS = {'laces': 'laces', 'strap': 'strap', 'buckle': 'buckle', 'zipper': 'zipper',
                'tongue': 'tongue', 'box': 'packaging or box', 'insert': 'insole'}


def parts_in(value):
    text = ' ' + ' '.join(str(value).lower().split()) + ' '
    return {name for name, words in PARTS.items()
            if any(re.search(r'\b' + re.escape(word) + r'\b', text) for word in words)}


def required(value):
    """Parts the description says are operated; a passing colour mention is not an action."""
    text = ' '.join(str(value).lower().split())
    found = set()
    if re.search(r'\b(box|boxes|packaging|carton|package)\b', text):
        found.add('box')
    for name, words in PARTS.items():
        if name == 'box':
            continue
        for word in words:
            for match in re.finditer(r'\b' + re.escape(word) + r'\b', text):
                window = text[max(0, match.start() - 70):match.end() + 70]
                if re.search(r'\b(?:' + OPERATE + r')\b', window):
                    found.add(name)
                    break
    return found


PART_WORDS = '|'.join(re.escape(word) for words in PARTS.values() for word in words)


def positive(value):
    """Drop negated phrases: an adaptation says what is absent, it does not use it.

    Postfix forms run first: 'Laces are not included.' would otherwise leave the bare noun behind.
    """
    text = ' '.join(str(value).split())
    text = re.sub(r'\b(?:' + PART_WORDS + r')\s+(?:are|is)\s+not\s+(?:included|present|supplied|attached|visible)\b[^.;]*',
                  ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(?:' + PART_WORDS + r')[\s-]+free\b', ' ', text, flags=re.IGNORECASE)
    return re.sub(r'\b(?:no|not|without|lacks?|lacking|never|none)\b[^.;]*', ' ', text, flags=re.IGNORECASE)


def profile(facts):
    present = parts_in(positive(facts))
    low = facts.lower()
    slip = ('slip' in low and ' on' in low) or 'opening' in present or 'collar' in low
    return {'parts': sorted(present), 'slip_on': bool(slip)}


def substitute(missing, product):
    if missing == {'box'}:
        return ('Hold both shoes in the hands and turn them slowly, showing the knitted upper and the pink sole. '
                'No packaging appears in this shot.')
    if missing & {'laces', 'strap', 'buckle', 'zipper'}:
        if product['slip_on']:
            return ('Pinch the top edge of the opening between thumb and forefinger, stretch it gently wide '
                    'and lower the foot in, keeping every finger outside the knit. The shoe has no laces and no strap.')
        return 'Rotate the shoe in the hand and press the upper with the thumb to show the material.'
    if missing & {'tongue', 'insert'}:
        return 'Hold the shoe at the opening and turn it so the inside of the collar is visible.'
    return 'Hold the shoe and turn it slowly to show the replacement product.'


def reconcile(actions, facts):
    product = profile(facts)
    changed, motives = [], {}
    for key, row in actions.items():
        if row.get('decision') != 'keep':
            continue
        missing = required(row.get('source_action', '')) - set(product['parts'])
        if not missing:
            continue
        row['decision'] = 'adapt'
        row['missing_parts'] = sorted(missing)
        row['motion'] = substitute(missing, product)
        row['reason'] = ((row.get('reason') or '') + ' Target lacks: ' + ', '.join(sorted(missing))).strip()[:3500]
        changed.append(key)
        motives[key] = sorted(missing)
    absent = {name for name in CANON if name not in product['parts']}
    adapted = {k: list(v['missing_parts']) for k, v in actions.items() if v.get('missing_parts')}
    return {'product': product, 'adapted_here': changed, 'motives': motives, 'adapted': adapted,
            'absent': sorted(absent),
            'forbidden': {name: ABSENT_WORDS[name] for name in CANON if name in absent and name in ABSENT_WORDS}}


def main():
    out = Path('evidence/rea-video/full35')
    path = out / 'remake' / 'actions.json'
    actions = json.loads(path.read_text(encoding='utf-8'))
    facts = next(row['target_evidence'] for row in actions.values() if row.get('target_evidence'))
    report = reconcile(actions, facts)
    path.write_text(json.dumps(actions, ensure_ascii=False, indent=1), encoding='utf-8')
    (out / 'remake' / 'contract.json').write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({'parts': report['product']['parts'], 'absent': report['absent'],
                      'adapted': report['motives'], 'keep': sum(1 for r in actions.values() if r['decision'] == 'keep'),
                      'adapt': sum(1 for r in actions.values() if r['decision'] == 'adapt')}, ensure_ascii=False))
    assert 'laces' in report['absent'] and 'box' in report['absent']
    assert all('missing_parts' not in r or r['decision'] == 'adapt' for r in actions.values())


if __name__ == '__main__':
    main()