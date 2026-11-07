"""ImageNet amplitude and extent scaling on a 20x20 grid."""
import json
from pathlib import Path
import sys
import numpy as np
import scaling_cache as sc
import train_scaling_pipeline as tp

ROOT = Path(__file__).resolve().parent

def main():
    import torch
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(.25)
    previous = json.loads((ROOT / 'cache/historical_20point_result.json').read_text(encoding='utf-8'))
    config = dict(previous['config'], images_per_group=0,
                  amplitudes=np.linspace(.2, 2., 20).tolist(), extents=np.linspace(.2, 1., 20).tolist())
    data = json.loads(tp.RECORDS.read_text(encoding='utf-8'))
    outcome = tp.compute_run(config, data, 'train_45_55_classnorm_historical_20point')
    sc.write_json(ROOT / 'cache/historical_20point_result.json', outcome)
if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
