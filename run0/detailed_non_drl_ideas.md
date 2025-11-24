# Detailed Guide: Bayesian Optimization and Cluster-Based Control for Turbulence

This document details two powerful, data-driven alternatives to Deep Reinforcement Learning (DRL) for turbulent flow control: **Bayesian Optimization** and **Cluster-Based Reduced Order Models (CROM)**.

These methods address your key constraints:
1.  **Data Efficiency:** They require far fewer simulations than DRL.
2.  **Interpretability:** They produce control laws that are easier to understand.
3.  **Actuation Constraints:** They can be explicitly designed to respect frequency and amplitude limits.

---

## 1. Bayesian Optimization (Data-Efficient Tuning)

### The Concept
Instead of training a neural network with thousands of parameters to learn a policy from scratch, we assume a **functional form** for the control law (based on physics or intuition) and use Bayesian Optimization to find the optimal parameters.

**Proposed Control Law:**
You already have a robust opposition control-like law. Let's generalize it:
$$ v_{wall}(x, z, t) = A \cdot \tanh\left( \frac{u(x, z_{sens}, t) - \mu_u}{\sigma_u} \cdot B + C \right) $$
*   **Parameters to Optimize ($\theta$):**
    *   $A$: Amplitude scaling.
    *   $B$: Sensitivity/Gain.
    *   $C$: Bias/Offset.
    *   $z_{sens}$: Sensing height (if variable).
    *   $\Delta t_{act}$: Actuation update frequency (to handle latency).

### The Algorithm (Gaussian Processes)
Bayesian Optimization uses a **Gaussian Process (GP)** to model the objective function (Drag Reduction) over the parameter space.

1.  **Prior:** Start with a belief about how Drag depends on $\theta$.
2.  **Acquisition:** The GP suggests the next set of parameters $\theta_{next}$ to try. It balances:
    *   **Exploitation:** Trying values near known good results.
    *   **Exploration:** Trying values in uncertain regions.
3.  **Evaluation:** Run `CaNS` with $\theta_{next}$ for a short duration (e.g., 2000 steps) and measure Drag.
4.  **Update:** Feed the result back into the GP to refine the model.
5.  **Repeat:** Typically converges in **20-50 iterations**.

### Implementation Steps

1.  **Define the Objective Function:**
    Create a Python wrapper around `CaNS` that takes parameters $\theta$, modifies `input.nml` or the control code, runs the simulation, and returns the average drag.

2.  **Use a Library:**
    Use `scikit-optimize` or `BoTorch`.

    ```python
    from skopt import gp_minimize

    def objective_function(params):
        A, B, C = params
        # 1. Update config/code with new A, B, C
        # 2. Run CaNS (mpi execution)
        # 3. Parse output to get Drag
        return drag_coefficient

    # Define bounds for A, B, C
    space = [(-1.0, 1.0), (0.1, 5.0), (-0.5, 0.5)]

    # Run Optimization
    res = gp_minimize(objective_function, space, n_calls=50)
    ```

### Why This Fits You
*   **Low Cost:** You only need ~50 short simulations, not millions of steps.
*   **Robustness:** GPs handle noisy drag measurements naturally.
*   **Simplicity:** You optimize 3-5 physically meaningful numbers, not a black-box neural network.

---

## 2. Cluster-Based Reduced Order Models (CROM)

### The Concept
Turbulence is chaotic but not random. It cycles through recurrent patterns (coherent structures) like streaks, rolls, and bursts. CROM simplifies the infinite-dimensional flow into a **finite-state machine**.

**Idea:**
1.  Identify the key "states" of the flow (Clusters).
2.  Model the transitions between them (Markov Chain).
3.  Apply control *only* to prevent transitions to high-drag states.

### The Methodology

#### Step 1: Data Collection & Clustering
*   Run a baseline DNS simulation.
*   Collect snapshots of the velocity field (or wall quantities $p_w, \tau_w$).
*   Perform **K-Means Clustering** on these snapshots to group them into $K$ clusters (e.g., $K=10$).
    *   *Centroid $C_1$:* Quiescent flow (Low Drag).
    *   *Centroid $C_5$:* Strong sweep event (High Drag).
    *   *Centroid $C_9$:* Bursting event (Very High Drag).

#### Step 2: Transition Matrix
*   Calculate the probability of moving from Cluster $i$ to Cluster $j$ in one time step: $P_{ij}$.
*   This gives you a **Transition Matrix** $\mathbf{P}$.

#### Step 3: Control Strategy
*   **Goal:** Keep the system in "Low Drag" clusters.
*   **Action:** When the system is in Cluster $i$ and likely to move to a "bad" Cluster $j$, apply a specific actuation to "nudge" it towards a "good" Cluster $k$.
*   **Implementation:**
    *   Offline: Pre-compute the optimal nudge for each cluster transition.
    *   Online:
        1.  Measure current state.
        2.  Classify into nearest Cluster $i$.
        3.  Look up optimal action for Cluster $i$.
        4.  Apply action.

### Implementation Steps

1.  **Snapshot Generation:** Use your existing `CaNS` setup to save 2D wall slices every $N$ steps.
2.  **Clustering (Python):**
    ```python
    from sklearn.cluster import KMeans
    # data shape: [n_snapshots, n_features]
    kmeans = KMeans(n_clusters=10).fit(data)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    ```
3.  **Online Classifier:**
    Inside `stwEnv_pettingzoo.py` (or a new controller), instead of a Neural Network, use the saved Centroids to classify the current observation:
    ```python
    # Find nearest centroid
    distances = np.linalg.norm(current_obs - centroids, axis=1)
    current_cluster = np.argmin(distances)
    action = lookup_table[current_cluster]
    ```

### Why This Fits You
*   **Interpretability:** You can visualize exactly what "Cluster 5" looks like and why it causes drag.
*   **Sparse Control:** You don't need to actuate everywhere, all the time. You only actuate when a "burst" is imminent.
*   **Physics-Based:** It respects the coherent structure dynamics of the flow.
