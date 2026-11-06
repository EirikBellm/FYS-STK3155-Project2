"""Check AL-PUB v2 and construct the fixed publication holdout.

Student: Specified duplicate handling, publication grouping and the fixed
development/test split design.

LLM-assisted: Codex implemented the AL-PUB archive checks, duplicate handling,
publication grouping, and split manifests from the specified data-handling
requirements on 27 October 2026.
"""
import argparse
import csv
import hashlib
import io
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/alpub_v2'
ARCHIVE = ROOT / 'data/alpub_v2/ALPUB_v2.zip'
SEED = 2026
NAMES = ['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon', 'Zeta', 'Eta',
         'Theta', 'Iota', 'Kappa', 'Lambda', 'Mu', 'Nu', 'Xi', 'Omicron',
         'Pi', 'Rho', 'LunateSigma', 'Tau', 'Upsilon', 'Phi', 'Chi',
         'Psi', 'Omega']
LETTERS = list('αβγδεζηθικλμνξοπρϲτυφχψω')
parser = argparse.ArgumentParser()
parser.add_argument('--stage', choices=['audit'], default='audit')
stage = parser.parse_args().stage
OUT.mkdir(parents=True, exist_ok=True)


def save_csv(path, rows):
    """Write records to a CSV file.

    LLM-assisted: Codex implemented this output helper.
    """
    if rows:
        with path.open('w', newline='', encoding='utf8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)




if stage == 'audit':
    rows = []
    hashes = defaultdict(list)
    failures = []
    with zipfile.ZipFile(ARCHIVE) as z:
        # fail visibly on archive corruption or an unknown filename.
        assert z.testzip() is None, 'Archive CRC check failed'
        paths = sorted(n for n in z.namelist() if n.lower().endswith('.jpg'))
        for i, path in enumerate(paths):
            name = Path(path).name
            label = path.split('/')[2]
            assert label in NAMES, f'Unknown label: {path}'
            match = re.search(r'POxy\.v(\d+)\.n?(\d+)', name)
            assert match, f'Unparsed papyrus identifier: {path}'
            # Keep all parts of a published item together. One spelling omits n.
            document = f'POxy.n{int(match[2])}'
            raw = z.read(path)
            try:
                with Image.open(io.BytesIO(raw)) as image:
                    rgb = image.convert('RGB')
                    digest = hashlib.sha256(
                        str(image.size).encode() + rgb.tobytes()
                    ).hexdigest()
                    width, height = image.size
            except (OSError, ValueError) as error:
                # record failed decodes explicitly rather than hide them.
                failures.append({'path': path, 'error': str(error)})
                continue
            hashes[digest].append(len(rows))
            rows.append({
                'path': path, 'label': label, 'document': document,
                'publication_volume': int(match[1]),
                'source_image': name.split('_')[1],
                'width': width, 'height': height,
                'file_sha256': hashlib.sha256(raw).hexdigest(),
                'pixel_sha256': digest, 'status': 'kept',
            })
            if (i + 1) % 50000 == 0:
                print('Audited', i + 1, 'images', flush=True)
        (OUT / 'alpub_v2_inventory.txt').write_bytes(z.read('alpub_v2_inventory.txt'))
    duplicates = []
    for digest, indices in hashes.items():
        if len(indices) < 2:
            continue
        labels = {rows[i]['label'] for i in indices}
        conflict = len(labels) > 1
        for j, i in enumerate(indices):
            if conflict or j > 0:
                rows[i]['status'] = 'conflicting_pixels' if conflict else 'duplicate_pixels'
            duplicates.append({
                'path': rows[i]['path'], 'label': rows[i]['label'],
                'document': rows[i]['document'], 'pixel_sha256': digest,
                'conflicting_labels': conflict, 'status': rows[i]['status'],
            })
    # Pick the holdout before fitting models. No image-level random holdout.
    groups = np.array(sorted({r['document'] for r in rows if r['status'] == 'kept'}))
    rng = np.random.default_rng(SEED)
    rng.shuffle(groups)
    holdout = set(groups[:int(np.ceil(.2 * len(groups)))])
    for row in rows:
        row['split'] = ('excluded' if row['status'] != 'kept' else
                        'reserved_test' if row['document'] in holdout else 'development')
    save_csv(OUT / 'manifest.csv', rows)
    save_csv(OUT / 'duplicates.csv', duplicates)
    save_csv(OUT / 'decode_failures.csv', failures)
    class_counts = []
    for label in NAMES:
        subset = [r for r in rows if r['label'] == label]
        kept = [r for r in subset if r['status'] == 'kept']
        class_counts.append({
            'label': label, 'raw_count': len(subset), 'kept': len(kept),
            'documents': len({r['document'] for r in kept}),
            'development': sum(r['split'] == 'development' for r in kept),
            'reserved_test': sum(r['split'] == 'reserved_test' for r in kept),
        })
    save_csv(OUT / 'class_counts.csv', class_counts)
    audit = {
        'archive_sha256': hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
        'raw_images': len(paths), 'decode_failures': len(failures),
        'kept': sum(r['status'] == 'kept' for r in rows),
        'removed_by_status': dict(Counter(r['status'] for r in rows if r['status'] != 'kept')),
        'duplicate_pixel_groups': sum(len(v) > 1 for v in hashes.values()),
        'conflicting_pixel_groups': sum(len({rows[i]['label'] for i in v}) > 1 for v in hashes.values()),
        'documents': len(groups), 'reserved_test_documents': sorted(holdout),
        'split_counts': dict(Counter(r['split'] for r in rows)),
        'seed': SEED,
        'group_rule': 'POxy publication item number, normalized across parts and optional n spelling',
        'preprocessing': 'White alpha composite, grayscale, per-image autocontrast, aspect-ratio resize within 26x26, white pad to 28x28, invert /255',
        'duplicate_policy': 'Remove all conflicting-label pixel matches, keep first same-label pixel match',
        'limits': ['Publication IDs are not verified physical-manuscript or writer identities',
                   'Joins across different publication entries are not checked',
                   'No writer identifiers or verified writer mapping in this release',
                   'No comprehensive near-duplicate or expert label review'],
    }
    (OUT / 'audit.json').write_text(json.dumps(audit, indent=2))
    print('Audit complete:', audit['kept'], 'kept crops across',
          audit['documents'], 'publication groups', flush=True)
