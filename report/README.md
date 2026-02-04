# Action Memory Implementation Report

This directory contains the LaTeX documentation for the action memory implementation in multi-agent reinforcement learning for turbulent flow control.

## Document Contents

**`action_memory_implementation.tex`** - Main LaTeX document (15 pages) covering:

1. **Introduction** - Problem statement and root cause analysis of bang-bang control
2. **Mathematical Formulation** - Rigorous MDP formulation with and without action memory
3. **Mechanism** - Detailed explanation of how action memory prevents bang-bang behavior
4. **Critic Architecture** - Analysis of apparent redundancy and why it's necessary
5. **Implementation Details** - Environment, network architectures, and training algorithm
6. **Backwards Compatibility** - How the system maintains compatibility with old configs
7. **Zero-Mean Constraint** - Critical implementation details for gradient flow and physical validity
8. **Expected Results** - Predicted improvements in smoothness and performance metrics

**`network_diagram.tex`** - Standalone TikZ diagram showing:

- Complete network architecture (actor + critic)
- Observation flow with 3-channel inputs (u, w, prev_action)
- Differentiable zero-mean correction
- Noise addition and clipping during training
- Gradient backpropagation paths
- Action memory feedback loop
- Training vs inference differences

## Compilation

### Prerequisites

- LaTeX distribution (TeX Live, MiKTeX, or MacTeX)
- Required packages: amsmath, amssymb, algorithm, algorithmic, booktabs, hyperref, etc.

### Build Instructions

**Using Make (recommended):**
```bash
cd report
make              # Compile both document and diagram
make view         # Compile and open main document (macOS)
make clean        # Remove auxiliary files
make cleanall     # Remove all generated files including PDFs
make rebuild      # Clean and rebuild
```

**Manual compilation:**
```bash
# Main document
pdflatex action_memory_implementation.tex
pdflatex action_memory_implementation.tex  # Run twice for cross-references

# Network diagram
pdflatex network_diagram.tex
```

### Output Files

- `action_memory_implementation.pdf` - Full technical report (15 pages)
- `network_diagram.pdf` - Network architecture diagram (1 page, standalone)

## Key Equations

### State Augmentation
Without action memory:
```
s_t = [u_t, w_t]
```

With action memory:
```
s_t = [u_t, w_t, a_{t-1}]
```

### Critic Input
Without action memory:
```
Critic receives: [u, w, a_current] (3 channels)
```

With action memory:
```
Critic receives: [u, w, a_prev, a_current] (4 channels)
```

### Why This Works

The Q-function with action memory:
```
Q([u_t, w_t, a_{t-1}], a_t) = r_t + γ E[Q([u_{t+1}, w_{t+1}, a_t], a_{t+1})]
```

Creates temporal coupling: choosing `a_t` affects the next state because the next state includes `a_t`.

## File Structure

```
report/
├── action_memory_implementation.tex    # Main LaTeX document (15 pages)
├── network_diagram.tex                 # TikZ network architecture diagram
├── Makefile                            # Build automation
├── README.md                           # This file
├── action_memory_implementation.pdf    # Compiled main document
└── network_diagram.pdf                 # Compiled diagram
```

## Notes

- All LaTeX temporary files (`.aux`, `.log`, etc.) are gitignored
- The PDF can be version controlled if desired (not currently in `.gitignore`)
- For questions about the implementation, see `../run0/IMPLEMENTATION_SUMMARY.md`
