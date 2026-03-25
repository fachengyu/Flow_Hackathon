## Signal Data

We use Allen data consisting of Ca (30 Hz) and Neuropixels (120Hz) recording from pseudomice (stacked neurons from multiple mice), recorded while the Natural Movie1 stimulus (30sec, 30Hz) was passively shown during 10 repeats. The tutorial of loading Allen data and training corresponding encoders can be found below.
<https://cebra.ai/docs/demo_notebooks/Demo_Allen.html>

As an example, we jointly train Ca and Neuropixels using two cortexes, VISp and VISrl, and save the corresponding embeddings on which we train flows. These embeddings can be found

```text
├── data/
│   └──allen_data/
```

In the walkthrough notebook, we use Algorithm 1 and Algorithm 2 from stochastic interpolants <https://arxiv.org/pdf/2310.03725> to build the flow from VISp Ca embeddings to VISrl Ca embeddings, and we evaluate the flow performance by knn accuracy, MSE, and FID.