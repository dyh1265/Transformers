# JEPA Integration Plan (Nano-LLM)

## Goal
Add JEPA-style latent predictive learning to improve abstraction, robustness, and multi-step coherence, while preserving current autoregressive generation quality.

## Why JEPA (expected benefits)
- Better abstraction: predict latent structure, not only next-token surface form
- Noise robustness: less sensitivity to token-level perturbations
- Longer-horizon planning signal: better multi-step consistency
- Potential data efficiency: representation learning from predictive objectives
- Multi-modal extension path: text + code graph/image alignment later

## Scope (phased)

### Phase 1 - Baseline JEPA Auxiliary Loss
- Keep existing LM training pipeline
- Add JEPA loss on hidden states as auxiliary objective
- Total objective: `L_total = L_lm + lambda_jepa * L_jepa`

#### Deliverables
- Config flags:
  - `use_jepa: bool`
  - `lambda_jepa: float`
  - `jepa_target_span: int`
  - `jepa_context_span: int`
- Training logs include `lm_loss`, `jepa_loss`, `total_loss`
- Checkpoint includes JEPA-related config metadata

### Phase 2 - Target Encoder Stabilization
- Add EMA/frozen target encoder branch for JEPA targets
- Predictor maps context embedding -> target embedding
- Optional variance regularization to avoid collapse

#### Deliverables
- EMA update schedule
- Ablation: with/without EMA
- Stability metrics in training output

### Phase 3 - Memory/Graph Conditioning (optional)
- Add retrieval/graph memory vectors as additional JEPA context
- Compare:
  - vanilla LM
  - LM + JEPA
  - LM + JEPA + memory/graph

#### Deliverables
- Simple graph memory prototype (entities/relations)
- Prompt/context injection strategy
- Evaluation on long-context consistency tasks

## Technical design (initial)
1. Extract context and target spans from each sequence
2. Encode context with model hidden states
3. Predict target latent representation with small predictor head
4. Compute JEPA loss (MSE or cosine) against detached target embedding
5. Combine with LM loss using `lambda_jepa`

## Evaluation plan
- Perplexity / cross-entropy (LM quality)
- Long-range consistency probes (entity continuity, topic drift)
- Robustness checks (noisy prompt perturbation)
- Generation quality snapshots at fixed seeds and sampling settings

## Risks
- Representation collapse if JEPA dominates
- LM quality degradation if `lambda_jepa` too high
- Added training complexity / hyperparameter sensitivity

## Mitigations
- Start with small `lambda_jepa` (e.g., 0.05-0.2)
- Keep LM objective primary
- Run short ablations before long training runs

## Milestones
1. Week 1: Phase 1 implementation + smoke tests
2. Week 2: tuning + baseline comparison
3. Week 3: EMA target encoder + ablations
4. Week 4: optional memory/graph extension

## Success criteria
- No major regression in LM generation quality
- Measurable gain in coherence/consistency metrics
- Stable training with reproducible configs
