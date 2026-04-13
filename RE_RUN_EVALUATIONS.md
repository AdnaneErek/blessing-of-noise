# Re-running Evaluations

The previous evaluations ran but didn't complete linear probing (no accuracy metrics). 

## Fixes Applied

1. ✅ **Task name detection** - Now correctly detects Diagnosis vs Prognosis per dataset
2. ✅ **Linear probing condition** - Only runs for Diagnosis datasets (which have labels)

## Re-run Evaluations

Now that fixes are in place, re-run:

```bash
./run_evaluate_all.sh
```

This will:
- Re-evaluate all datasets with correct task types
- Run linear probing on diagnosis datasets (CWRU, JNUB, KAUG17, HSG18)
- Skip linear probing for prognosis (XJTU-SY) - it doesn't have classification labels
- Generate new result files with accuracy metrics

## Expected Output

After re-running, `results/pretrained_accuracy_summary.json` will contain:

```json
{
  "datasets": [
    {
      "dataset": "CWRU",
      "status": "success",
      "test_accuracy": 0.82,
      "test_f1": 0.81,
      ...
    },
    ...
  ]
}
```

## Note

I've removed the old result files so they'll be regenerated with the fixes.
