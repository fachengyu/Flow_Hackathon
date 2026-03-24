# FDA-style conditional flow on Allen Ca embeddings

**Flow matching for neural adaptation** (Wang et al., ICML 2025 — [OpenReview](https://openreview.net/pdf?id=nKJEAQ6JCY), [code](https://github.com/wangpuli/FDA)).

This folder implements a **paper-aligned learning objective**:

- **Source `x₀`:** standard Gaussian noise in embedding space (same as FDA generative pre-training, rather than using another empirical neural sample as the bridge endpoint).
- **Target `x₁`:** the neural representation you want at \(t=1\) (standardized **VISrl** or **VISp** Ca embedding for that trial).
- **Condition `c`:** the **observed neural window** for the same trial — typically standardized **VISp** when \(x₁\) is **VISrl** (cross-area, few-trial adaptation), or \(x₁\) plus Gaussian noise when training in a **single** area so the conditional flow is identifiable.

The **probability path** is the same as FDA’s SiT transport (`flow/transport/path.py`):  
\(x_t = \alpha(t) x_1 + \sigma(t) x_0\),  
\(u_t = \alpha'(t) x_1 + \sigma'(t) x_0\),  
with **IC** (linear) or **GVP** plan. The network predicts a **conditional** velocity \(v_\theta(x_t, t, c)\) and is trained with flow matching to match \(u_t\).

**Inference:** sample \(z \sim \mathcal{N}(0,I)\), set \(c\) to the test-trial observation, integrate \(dX/dt = v_\theta(X,t,c)\) from \(t=0\) to \(1\), then decode.

### Evaluation (aligned with the paper / official FDA repo)

The motor-cortex experiments in the paper report **\(R^2\) (%)** using `sklearn.metrics.r2_score` on **2D continuous behavior** (hand velocity); see `experiment/ft_func.py` in [wangpuli/FDA](https://github.com/wangpuli/FDA). Fine-tuning uses a **small target-session calibration set** (`tgt_train_ratio`, e.g. **2%** in `test_FDA_on_spikes_ft.py`).

This hackathon port adds `fda_evaluation.py`:

- **\(R^2\) (%)** = 100 × `r2_score` with a **2D phase coding** of movie frame index (sin/cos), as a same-dimensional linear-readout analogue to 2D velocity.
- **Few-trial calibration:** subsample the target-area **train** bank by `calibration_ratio` (same role as `tgt_train_ratio`) and fit decoders on that subset; optional **multi-seed** averaging.
- **kNN quantized frame accuracy** (Allen notebook convention) is reported on the **same** calibration splits for direct comparison.

| File | Role |
|------|------|
| `signal_model.py` | `ICPlan`, `GVPCPlan`, `ConditionalVelocityFieldMLP`, `fda_conditional_flow_matching_loss`, `integrate_rk4_conditional`, `aligned_minibatch`, `standard_gaussian_latent` (+ legacy unconditional `VelocityFieldMLP`) |
| `fda_evaluation.py` | `fda_metrics_sweep`, `print_fda_sweep_table`, \(R^2\) % + few-trial kNN |
| `signal_decode.py` | `allen_frame_id_decode` (`modality='ca'`) |
| Data | `../data/allen_data/VISp_*_ca_*_{train,test}.npy`, `VISrl_*` (see `signal_walkthrough.ipynb`) |

## Run

Open `signal_walkthrough.ipynb` from this directory (or any parent where `signal_model.py` can be resolved).

```bash
cd Flow_Hackathon/signal_flow
jupyter lab signal_walkthrough.ipynb
```

## Citation

```bibtex
@inproceedings{wang2025FDA,
  title     = {Flow Matching for Few-Trial Neural Adaptation with Stable Latent Dynamics},
  author    = {Wang, Puli and Qi, Yu and Wang, Yueming and Pan, Gang},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning (ICML)},
  year      = {2025}
}
```
