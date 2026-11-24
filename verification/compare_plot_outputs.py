"""Compare every regenerated plot with the matching release renderer output.

Uses the existing exact comparison policy: every PDF page at 150 dpi and
decoded image pixels. This tests faithful replotting, not raw-data reanalysis.
Author-supplied manuscript comparisons are recorded separately in the release.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--expected-root', type=Path)
    parser.add_argument('--only', nargs='+', help='Optional stage names')
    args = parser.parse_args()
    data = args.data_root.absolute()
    os.environ['EICHE_FISCHER_DATA_ROOT'] = str(data)
    from verification.compare_release_outputs import Fingerprinter, POLICY
    expected = args.expected_root or data / 'Reference/Plot_Renderer_Outputs'
    observed = data / 'Results'
    readers = Fingerprinter(expected, ROOT), Fingerprinter(observed, ROOT)
    records = []
    for source_file in sorted((data / 'Plot_Source').glob('*/source.json')):
        source = json.loads(source_file.read_text(encoding='utf-8'))
        if args.only and source['stage'] not in args.only:
            continue
        for relative in source['outputs']:
            path = Path(relative)
            if path.is_absolute() or '..' in path.parts:
                raise ValueError('Invalid output path: ' + relative)
            baseline, result = expected / path, observed / path
            status = 'missing' if not result.is_file() else 'missing_reference' if not baseline.is_file() else None
            if status is None:
                status = 'matched' if readers[0].artifact(baseline) == readers[1].artifact(result) else 'different'
            records.append({'stage': source['stage'], 'path': relative, 'status': status})
            print(f'{status}: {relative}', flush=True)
    counts = {status: sum(r['status'] == status for r in records) for status in ('matched', 'different', 'missing', 'missing_reference')}
    passed = bool(records) and counts['matched'] == len(records)
    report = {'status': 'passed' if passed else 'failed', 'counts': counts, 'records': records,
              'comparison_policy': POLICY, 'scope': 'Faithful replot from the frozen figure-source release; not raw-cohort reanalysis.'}
    output = data / 'Results/comparison.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': report['status'], 'counts': counts}))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
