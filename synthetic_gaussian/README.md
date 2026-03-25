# OT flow matching on block covariance Gaussians

Trains the same **optimal-transport conditional flow** (stochastic interpolant) as
`synthetic_gaussian/solution/gaussian_model.py`: sample `x0 ~ N(0,I)`, target `x1` =
some block covariance Gaussian.

| File | Role |
|------|------|
| `gaussian_model.py` | `MLP_Residual`, `cfm_loss`, `sample` |
| `gaussian_data.py` | `get_gaussian_dataset`, `covariance_error`, `wasserstein_error`|
| Data | Found in `gaussian_data.py`|

## Run

```bash
cd Flow_Hackathon/solution/synthetic_gaussian
jupyter lab gaussian_walkthrough.ipynb
```

## Citation (Gaussian hackathon reference)

Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., & Le, M. (2022). *Flow Matching for Generative Modeling*. Preprint. Meta AI (FAIR) & Weizmann Institute of Science.
