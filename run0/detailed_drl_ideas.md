# Detailed Guide: DRL-Based Research Ideas for Turbulent Flow Control

This document details the Deep Reinforcement Learning (DRL) research directions we discussed, focusing on enhancing your existing setup.

**Status:**
*   [x] **Symbolic Distillation (PySR):** Implemented (see `generate_pysr_data.py`, `pysr_analysis.py`).
*   [x] **Cooperative Multi-Agent RL (GNNs):** Implemented (see `models_pettingzoo.py`).
*   [ ] **Global-Local Manager:** Conceptual.
*   [ ] **Asynchronous Control:** Conceptual.
*   [ ] **End-to-End Control:** Conceptual.

---

## 1. Symbolic Distillation (PySR) - *Implemented*
*Goal: Extract an interpretable equation from the "black box" neural network.*

### The Concept
You do not need to retrain your agent. You can "distill" your existing, expensive CNN policy into a simple, interpretable equation.

### The Workflow
1.  **Generate a Dataset:**
    *   Load your trained agent (`best_model.pt`).
    *   Run the environment for $N$ steps.
    *   Save pairs of **Inputs** (local $u, w$ patches, gradients $\partial u/\partial y$, $\partial w/\partial x$) and **Outputs** (actuation $a$).
2.  **Run PySR:**
    *   Feed this dataset into PySR.
    *   PySR searches for equations $a = f(features)$ that minimize error + complexity.
3.  **Validate:**
    *   Take the best equation found (e.g., $a = 0.5 u^2 - \nabla w$).
    *   Hard-code this equation into a new `Policy` class and run CaNS.

### Why this is powerful
*   **Interpretability:** Reveals the physical law the CNN is approximating.
*   **Speed:** Equations evaluate in nanoseconds.

---

## 2. Cooperative Multi-Agent RL (GNNs) - *Implemented*
*Goal: Allow agents to "talk" to neighbors to perceive larger structures.*

### The Concept
Replace the independent CNN encoder with a Graph Neural Network (GNN). This allows information to flow between agents before they decide on an action.

### How it works
1.  **Encoder:** Each agent processes its 8x8 patch $\rightarrow$ Feature Vector $h_i$.
2.  **Communication (GNN):** Agent $i$ exchanges $h_i$ with neighbors.
3.  **Update:** Agent $i$ updates its belief: $h_i' = \text{MLP}(h_i, \sum h_{neighbors})$.
4.  **Decoder:** Agent $i$ generates action $a_i$ from $h_i'$.

### Why this is powerful
*   **Pre-actuation:** Agents can see a "wave" coming from a neighbor *before* it hits their local patch.
*   **Coherence:** Enables control of structures larger than the 8x8 patch size.

---

## 3. Global-Local "Manager" (Zero-Momentum Compliant)
*Goal: Introduce global coordination without violating physical constraints.*

### The Concept
A **Global Agent** sees the full flow field but does **NOT** output a physical force (which would violate mass conservation). Instead, it outputs a **Context Vector** $z$ that acts as an extra input to the Local Agents.

### Mechanism
*   **Global Agent Output:** $z = [z_1, z_2, ...]$ (Abstract command, e.g., "High Turbulence Mode").
*   **Local Agent Input:** `[Local Patch (8x8), Global Context z]`.
*   **Local Agent Output:** $a_{ij}$ (Physical wall actuation).
*   **Constraint:** The zero-mean constraint is applied to the final $a_{ij}$ field, ensuring physics compliance.

---

## 4. Asynchronous Control (Frame Skipping)
*Goal: Handle real-world actuator latency and frequency limits.*

### The Concept
Real actuators cannot update every simulation time step ($dt$). Simply applying a policy trained for $dt$ at $10dt$ intervals fails.

### The Solution: Retraining
You must **RETRAIN** the agent with a "hold" constraint.
*   During training, force the agent to commit to an action for $k=10$ steps.
*   The agent learns to predict the flow evolution over $10dt$ and chooses a robust action.
*   This discovers the *true* limit of actuation frequency.

---

## 5. End-to-End Control (Wall-Only)
*Goal: Remove the need for flow reconstruction (U-Net).*

### The Concept
Map wall sensors ($p_w, \tau_w$) directly to actuation ($v_w$), skipping the "state estimation" step.

### The Challenge & Fix
*   **Problem:** The sensor reads its own actuation effect ("spillover"), creating a feedback loop.
*   **Fix:** **Subtractive Feedback**.
    *   Input to NN = $Measurement_{raw} - \text{ExpectedEffect}(Action_{prev})$.
    *   This isolates the turbulent signal from the control signal.
