# Quick Start: Action Memory Training

## 🚀 Start Training (Recommended)

```bash
cd run0
python stwStart_consistency.py --config config_consistency_with_memory.yaml
```

This will train with **3-channel observations** `[u, w, prev_action]` for smooth control.

---

## 📊 Key Differences

### Old Approach (2 channels)
```yaml
# config_consistency.yaml
# No observation section → defaults to 2 channels
```
- Observations: `[u, w]` - shape (2, 8, 8)
- Policy has no action memory
- May exhibit bang-bang behavior

### New Approach (3 channels) ✨
```yaml
# config_consistency_with_memory.yaml
observation:
  include_prev_action: true   # Enable action memory
  prev_action_scale: 1.0      # Scale factor
```
- Observations: `[u, w, prev_action]` - shape (3, 8, 8)
- Policy sees what it just did
- Learns smooth control naturally

---

## 🎯 Expected Improvements

| Metric | Without Memory | With Memory (Expected) |
|--------|----------------|------------------------|
| Temporal gradient RMS | ~0.4 | ~0.1-0.2 |
| High-frequency content | High | Reduced |
| Bang-bang behavior | Present | Reduced |
| Drag reduction | Baseline | Maintained or better |

---

## 📁 Files Modified

1. ✅ `config_consistency_with_memory.yaml` - New config
2. ✅ `stwEnv_pettingzoo.py` - Added prev_action tracking
3. ✅ `models_consistency.py` - Support 2 or 3 input channels
4. ✅ `stwStart_consistency.py` - Auto-detect channels

---

## 🔧 Resume Training

```bash
# Resume from latest checkpoint
python stwStart_consistency.py --config config_consistency_with_memory.yaml --resume

# Resume from specific checkpoint
python stwStart_consistency.py --config config_consistency_with_memory.yaml \
    --checkpoint ./checkpoints_consistency/checkpoint_step_7200.pt
```

---

## 📈 Monitor Training

```bash
# Start TensorBoard
tensorboard --logdir=./logs_consistency

# Watch for:
# - Episode/consistency_loss (should stabilize)
# - Metrics showing temporal smoothness
# - Fields/action_changes visualization
```

---

## 🔄 Backwards Compatibility

Old config still works unchanged:
```bash
python stwStart_consistency.py --config config_consistency.yaml
```

This uses 2-channel observations for backwards compatibility.

---

## 💡 Tuning

If actions are **too stiff** (not responsive enough):
```yaml
observation:
  prev_action_scale: 0.5  # Reduce influence (was 1.0)
```

If actions are **still too jumpy**:
```yaml
observation:
  prev_action_scale: 1.5  # Increase influence (was 1.0)
```

---

## 📖 Full Details

See `IMPLEMENTATION_SUMMARY.md` for complete implementation details.
