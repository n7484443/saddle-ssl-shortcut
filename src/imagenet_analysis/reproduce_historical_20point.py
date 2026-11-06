"""User-confirmed historical 20x20 sweep; isolated from existing 10x9 caches."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import scaling_cache as sc
import train_scaling_pipeline as tp

ROOT = Path(__file__).resolve().parent
STATE = ROOT / 'historical_20point_status.json'

def main():
    import torch
    # Leave ample VRAM for the separately authorized MNIST run.
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(.25)
    tp.STATUS = STATE
    tp.STATE = dict(pid=os.getpid(), status='starting_historical_20point')
    base = ROOT / 'cache/train_runs/f237da3a3d776a322b1a'
    previous = json.loads((base / 'protocol.json').read_text(encoding='utf-8'))
    config = dict(previous['config'], images_per_group=0,
                  amplitudes=np.linspace(.2, 2., 20).tolist(), extents=np.linspace(.2, 1., 20).tolist())
    data = json.loads(tp.RECORDS.read_text(encoding='utf-8'))
    groups, audit = sc.select_groups(data['records'], config)
    assert {c: [r['stem'] for r in rows] for c, rows in groups.items()} == previous['selected_images']
    assert audit['computed_groups'] == 188 and audit['computed_images'] == 9073
    outcome = tp.compute_run(config, data, 'train_45_55_classnorm_historical_20point')
    directory = ROOT / 'cache/train_runs' / outcome['key']
    assert outcome['summary']['amplitude']['total_modes'] == 9063
    # Check actual fresh kernel evaluation against independently reconstructed amplitude.
    reference = json.loads((ROOT.parent / 'output/imagenet_historical_20point_amplitude.json').read_text())
    np.testing.assert_allclose(outcome['summary']['amplitude']['pooled_mode_median_exponent'], reference['median_exponent'], rtol=1e-8, atol=1e-9)
    sc.write_json(ROOT / 'cache/historical_20point_result.json', outcome)
    manifest_path = ROOT / 'cache/train_study_results.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest = [r for r in manifest if r['name'] != outcome['name']]
    manifest.append(outcome)
    sc.write_json(manifest_path, manifest)
    from imagenet_reproduction import app
    _, definitions = app.run()
    definitions['export_figure'](definitions['result'])
    tp.status(status='completed', key=outcome['key'], directory=str(directory),
              summary=outcome['summary'], report=str(ROOT.parent / 'output/pdf/imagenet_scaling_reproduction.pdf'))
    print(json.dumps(outcome['summary'], indent=2), flush=True)

if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    try:
        main()
    except Exception as exc:
        tp.STATUS = STATE
        tp.status(status='failed', error=repr(exc))
        raise
