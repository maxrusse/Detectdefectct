# Custom Data Loading Guide

This guide explains how to load data from your nested JSON structure with STag-based splitting.

---

## Your Data Structure

Your data has a nested JSON format where:

- **Files are grouped by study/patient**
- **Split is determined by STag field:**
  - `Tx` → Training set
  - `Ty` → Validation set
  - `Tz` → Test set

- **Each case has 3 files:**
  - `base.nii` → CT scan
  - `mask1.nii.gz` → Bone mask
  - `mask2.nii.gz` → Tumor mask

---

## Example Input JSON

```json
{
  "json": [
    [
      {
        "Study001": [
          {
            "patients_id": "patient_001",
            "studies_id": "study_001",
            "SubFolder": "case001",
            "STag": "Tx",
            "Filename": "base.nii",
            "absFilePath": "/path/to/data/patient_001/study_001/case001/base.nii"
          },
          {
            "patients_id": "patient_001",
            "studies_id": "study_001",
            "SubFolder": "case001",
            "STag": "Tx",
            "Filename": "mask1.nii.gz",
            "absFilePath": "/path/to/data/patient_001/study_001/case001/mask1.nii.gz"
          },
          {
            "patients_id": "patient_001",
            "studies_id": "study_001",
            "SubFolder": "case001",
            "STag": "Tx",
            "Filename": "mask2.nii.gz",
            "absFilePath": "/path/to/data/patient_001/study_001/case001/mask2.nii.gz"
          }
        ]
      }
    ]
  ]
}
```

---

## Step-by-Step Usage

### 1. Parse Your Nested JSON

```bash
python scripts/parse_nested_json.py \
    --input /path/to/your/nested_data.json \
    --output data_config.json
```

**Options:**

```bash
--input              Input JSON file (required)
--output             Output file (default: data_config.json)
--base-filename      CT scan filename (default: base.nii)
--mask1-filename     Bone mask filename (default: mask1.nii.gz)
--mask2-filename     Tumor mask filename (default: mask2.nii.gz)
```

**Example with custom filenames:**

```bash
python scripts/parse_nested_json.py \
    --input my_data.json \
    --output processed_data.json \
    --base-filename ct_scan.nii \
    --mask1-filename bone_segmentation.nii.gz \
    --mask2-filename tumor_segmentation.nii.gz
```

### 2. Verify Output

The script will create a standard format JSON:

```json
{
  "train": [
    {
      "ct": "/path/to/case001/base.nii",
      "mask": "/path/to/case001/mask1.nii.gz",
      "mask1": "/path/to/case001/mask2.nii.gz",
      "patients_id": "patient_001",
      "study_id": "study_001",
      "SubFolder": "case001"
    }
  ],
  "valid": [...],
  "test": [...]
}
```

### 3. Train Your Model

```bash
python scripts/train.py \
    --config config/a6000_optimized.yaml \
    --data data_config.json
```

### 4. Test Your Model

```bash
python scripts/test.py \
    --model results/best_model.pth \
    --data data_config.json \
    --model-name swin
```

---

## Understanding the Mapping

### File Naming Convention

| Your File | Pipeline Expects | Maps To |
|-----------|-----------------|---------|
| `base.nii` | `ct` | CT scan (input) |
| `mask1.nii.gz` | `mask` | Bone segmentation |
| `mask2.nii.gz` | `mask1` | Tumor segmentation |

### Label Merging Logic

The pipeline automatically merges the two masks:

```python
# Priority: Tumor > Bone > Background
label = np.zeros_like(bone_data)
label[bone_data > 0] = 1   # Healthy bone
label[tumor_data > 0] = 2  # Tumor (overwrites bone)
```

**Output Classes:**
- `0` = Background
- `1` = Healthy Bone
- `2` = Tumor

---

## STag Splitting Logic

The parser automatically assigns splits based on STag:

```python
def split_from_stag(stag):
    if "Tx" in stag:  return "train"
    if "Ty" in stag:  return "valid"
    if "Tz" in stag:  return "test"
```

**Examples:**

| STag Value | Split | Usage |
|------------|-------|-------|
| `Tx` | train | Training data |
| `Tx_001` | train | Also training |
| `Ty` | valid | Validation data |
| `Ty_subset` | valid | Also validation |
| `Tz` | test | Test data |
| `Tz_final` | test | Also test |

---

## Troubleshooting

### No Training Samples Found

**Error:**
```
❌ ERROR: No training samples found!
```

**Solution:**
- Check that your JSON has records with `STag` containing `Tx`
- Verify all three files exist for each case

### Missing Files Warning

**Warning:**
```
⚠ Skipping case patient_001/study_001/case001 (train): missing mask2.nii.gz
```

**Solution:**
- Ensure all three files exist: base.nii, mask1.nii.gz, mask2.nii.gz
- Check that `absFilePath` points to existing files
- Verify filenames match exactly (case-sensitive)

### Different Filename Convention

If your files have different names:

```bash
python scripts/parse_nested_json.py \
    --input my_data.json \
    --base-filename ct.nii.gz \
    --mask1-filename segmentation_bone.nii.gz \
    --mask2-filename segmentation_tumor.nii.gz
```

---

## Complete Workflow Example

```bash
# 1. Clone and setup
git clone https://github.com/maxrusse/Detectdefectct.git
cd Detectdefectct
conda env create -f environment.yml
conda activate jaw-segmentation

# 2. Parse your data
python scripts/parse_nested_json.py \
    --input /data/my_nested_structure.json \
    --output data_config.json

# 3. Verify split statistics
cat data_config.json | python -m json.tool | grep -E '"train"|"valid"|"test"' | head -3

# 4. Train
python scripts/train.py \
    --config config/a6000_optimized.yaml \
    --data data_config.json \
    --output ./results

# 5. Test
python scripts/test.py \
    --model results/best_model.pth \
    --data data_config.json \
    --model-name swin
```

---

## Python API Usage

```python
import json
from scripts.parse_nested_json import (
    flatten_records,
    group_files_by_case,
    build_samples
)

# Load your data
with open("my_data.json") as f:
    data = json.load(f)["json"]

# Process
records = flatten_records(data)
groups, filenames = group_files_by_case(records)
samples, skipped = build_samples(groups, filenames)

# Use in training
from src.data import MultiMaskDataset, get_transforms
from src.utils import get_default_config

config = get_default_config()
train_dataset = MultiMaskDataset(
    samples['train'],
    transforms=get_transforms("train", config)
)
```

---

## FAQ

**Q: Do I need separate testing scripts?**

A: No! The `test.py` script already handles testing. Just use:
```bash
python scripts/test.py --model best_model.pth --data data_config.json
```

**Q: Can I use the hybrid train/valid/test split?**

A: Yes! The parser automatically creates all three splits based on your STag values.

**Q: What if some cases don't have all three files?**

A: The parser skips incomplete cases and reports them. You'll see:
```
⚠ Skipping case X: missing Y
```

**Q: Can I customize the split logic?**

A: Yes! Edit `parse_nested_json.py` and modify the `split_from_stag()` function.

---

## Next Steps

1. ✅ Parse your nested JSON → `data_config.json`
2. ✅ Verify splits (check train/valid/test counts)
3. ✅ Train model with `scripts/train.py`
4. ✅ Evaluate with `scripts/test.py`

See main README.md for full documentation.
