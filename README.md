# fluffy-couscous

Capability eval suite for QuickDraw, stroke-sequence models only

Objective
Build a diagnostic benchmark that evaluates what a stroke-sequence recognizer can reliably do under realistic variations of pen trajectories, sampling, and stroke segmentation. The reasoning agent's role is to design the suite: which capabilities matter, which semantic confusions to focus on, and what parameter grids to sweep. The suite itself is deterministic code that perturbs vector strokes and measures performance.

Model assumptions
Input is vector strokes (list of strokes, each stroke is a sequence of 2D points; optional timestamps). Your model consumes either:

1. (x, y, pen_state) token sequences, or
2. per-stroke sequences with explicit pen-up separators.
   You should standardize preprocessing to a canonical representation, then apply perturbations in that representation.

What "capabilities" mean for stroke-sequence recognition

1. Early recognition: correct category from partial trajectory.
2. Sampling invariance: robust to different point densities and resampling artifacts.
3. Segmentation invariance: robust when users split or merge strokes differently.
4. Temporal robustness: robust to plausible stroke order differences and within-stroke time warps.
5. Quantization and device noise: robust to coordinate quantization, jitter, and latency.
6. Canonicalization sensitivity: dependence on translation, scale normalization, starting point, and coordinate frame.
7. Semantic confusions: performance in human-plausible neighborhoods (cup vs mug, violin vs guitar, etc.).

Agent responsibilities
A) Generate test taxonomy and parameter grids
Given the class list (or a subset) and constraints ("vector-only, no raster"), the agent outputs a machine-readable test plan: test families, parameter sweeps, and relative weights for an overall score.

B) Generate semantic confusion neighborhoods
The agent proposes clusters and pairs that are meaningful for sketches. You will report cluster metrics and worst-pair metrics, not just global accuracy.

C) Produce grounded diagnostic summaries
After running the suite, you feed the metrics back and the agent produces a structured "model card" style report tied to specific numbers (no free-form opinions).

Suite components for stroke sequences

1. Prefix-of-trajectory tests (early recognition)
   For each sketch, evaluate model on prefixes by:

* prefix by strokes: first k% of strokes
* prefix by points: first k% of points across all strokes
  k in {10, 20, 30, 50, 70, 100}
  Outputs: accuracy vs k, area under prefix curve, prefix@20 and prefix@30.

2. Point dropout and stroke dropout (missing data)

* stroke dropout: drop a fraction p of strokes (keep order), p in {0.1, 0.2, 0.3, 0.5}
* point dropout: drop points within strokes with probability p, then reconnect or keep gaps depending on your input format
  Outputs: robustness curves, worst-case drop.

3. Resampling invariance (device sampling differences)
   Operate per stroke:

* uniform resample to N points per stroke, N in {8, 16, 32, 64}
* arc-length resample
* random subsample to target ratio r in {0.25, 0.5, 0.75}
  Outputs: sensitivity to point density and resampling method.

4. Simplification (trajectory compression)

* Douglas-Peucker with tolerance t sweeping a few levels
* or target point reduction ratios r
  This probes whether the model relies on high-frequency wiggles versus global shape.

5. Segmentation perturbations (stroke split/merge)
   These are uniquely important for sequence models.

* split: break long strokes into 2-4 strokes at random interior points
* merge: merge consecutive strokes if their endpoints are close
* pen-up noise: insert extra pen-up events, or remove pen-up between near-contiguous strokes
  Outputs: robustness to different user stroke habits and preprocessing.

6. Temporal and ordering perturbations (sequence dependence)

* reverse stroke order
* shuffle strokes within small windows (local reorder)
* reverse points within strokes (a strong test, sometimes unrealistic but diagnostic)
* time warping: compress or expand point spacing along the stroke (simulate speed differences) without changing geometry
  Outputs: order sensitivity profile. Models that "memorize order" will collapse here.

7. Coordinate noise and quantization (hardware and logging artifacts)

* additive jitter in normalized coordinates (small sigma levels)
* quantize coordinates to b bits, b in {4, 6, 8}
* rounding and clipping stress test
  Outputs: accuracy vs noise strength.

8. Canonicalization stress tests (preprocessing skew)
   These expose pipeline brittleness.

* translation offsets before normalization
* scale perturbation
* rotation by small angles (for some classes rotation invariance is desired; keep angles small)
* reflection (optional, often should hurt, but it tells you if the model collapses left-right)
* start-point shift within a stroke: cyclic shift of points to simulate different "starting at a different point" habits
  Outputs: invariance curves and failure mode ranking.

9. Confusion neighborhood evaluation (semantic diagnostics)
   Using agent-defined clusters:

* cluster accuracy
* worst-pair accuracy among specified pairs
* confusion entropy within cluster
  This is where you demonstrate the suite is more informative than generic corruptions.

A minimal JSON spec the agent should emit
Top level:

* version, seed_policy, canonical_representation
* tests: list of test families with parameter grids
* confusion_sets: clusters and pairs
* scoring: weights per family and key headline metrics

Example structure:

* tests:

  * name: prefix_points, params: {k: [10,20,30,50,70,100]}
  * name: resample_uniform, params: {N: [8,16,32,64]}
  * name: split_merge, params: {split_k: [2,3,4], merge_threshold: [0.02,0.05]}
  * name: quantize, params: {bits: [4,6,8]}
* confusion_sets:

  * cluster: containers, classes: [cup, mug, bowl, bottle], pairs: [(cup,mug), (bowl,cup)]
* scoring:

  * headline: [clean_acc, prefix_auc, robustness_auc, worst_cluster_acc]

What you report in the paper

1. Main table: clean accuracy plus 6-10 capability scores (one per family).
2. Two curves: prefix curve and one robustness curve (resampling or dropout).
3. Confusion cluster heatmap for 2-3 clusters.
4. Failure mode ranking per model: which family causes largest relative drop.
5. Stability: variance across random seeds for stochastic tests.

Why this is a strong workshop project

* It is clearly "systems and evaluation" oriented, which workshops accept readily.
* It produces actionable diagnostics for improving stroke-sequence models.
