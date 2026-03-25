# OT flow matching on Allen Ca embeddings

Trains the same **optimal-transport conditional flow** (stochastic interpolant) as
`synthetic_gaussian/solution/gaussian_model.py`: sample `x0 ~ N(0,I)`, target `x1` =
standardized **VISp** Ca embeddings, MSE on the OT velocity, then **Euler** integrate from
noise toward the data distribution.

| File | Role |
|------|------|
| `signal_model.py` | `MLP_Residual`, `cfm_loss`, `integrate_euler`, `sample` |
| `fda_evaluation.py` | Optional **R² %** + few-trial kNN sweep (`fda_metrics_sweep`) |
| `signal_decode.py` | `allen_frame_id_decode` (`modality='ca'`) |
| Data | `../data/allen_data/VISp_*_ca_*_{train,test}.npy` (see notebook) |

## Run

```bash
cd Flow_Hackathon/signal_flow
jupyter lab signal_walkthrough.ipynb
```

## Citation (Gaussian hackathon reference)

The interpolant and loss match the synthetic Gaussian walkthrough / Lipman et al. OT-CFM setup used in `synthetic_gaussian/solution/gaussian_model.py`.
