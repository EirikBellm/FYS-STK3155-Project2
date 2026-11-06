"""Calculate image-independent classification baselines from saved splits.

No fitting, simulation or model selection occurs.

Student: Specified the analytical baseline formulas and comparison design.

LLM-assisted: Codex implemented the saved-manifest loading, class-frequency
calculations and output formatting on 6 November 2026.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/classification_part_e'
manifest = pd.read_csv(OUT / 'reconstructed_manifest.csv')
counts = pd.crosstab(manifest.label, manifest.split)
original = pd.read_csv(ROOT / 'results/alpub_v2/class_counts.csv').set_index('label')
for split in ['development', 'reserved_test']:
    np.testing.assert_array_equal(counts[split], original.loc[counts.index, split])
train = counts.development
test = counts.reserved_test
assert len(counts) == 24 and train.sum() == 159053 and test.sum() == 46723
majority = train.idxmax()
result = {
    'training_split': 'full development set used for final model training',
    'training_n': int(train.sum()),
    'test_n': int(test.sum()),
    'classes': len(counts),
    'uniform_expected_accuracy': 1 / len(counts),
    'frequency_expected_accuracy': float(((train / train.sum()) * (test / test.sum())).sum()),
    'frequency_rule': 'sum_k (train_count_k / training_n) * (test_count_k / test_n)',
    'majority_class': majority,
    'majority_test_accuracy': float(test[majority] / test.sum()),
    'class_counts': {
        name: {'training': int(train[name]), 'test': int(test[name])}
        for name in counts.index
    },
}
(OUT / 'image_independent_baselines.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k != 'class_counts'}, indent=2))
