# The Design and Analysis of Benchmark Experiments — a reimplementation

A Python reimplementation of the simulation framework of

> T. Hothorn, F. Leisch, A. Zeileis, K. Hornik. *The Design and Analysis of Benchmark Experiments.* Journal of Computational and Graphical Statistics, 14(3), 2005.

The paper treats the performance of a learning algorithm as a **random variable** and asks how to compare two algorithms with proper statistical tests. This was my first project at the ARAMIS Lab (Paris Brain Institute, summer 2024), before working on performance uncertainty in medical imaging.

## Setup

Following the paper's regression example, data are generated as

$$y = 2x + \beta_2 x^2 + \varepsilon, \qquad x \sim \mathcal{U}(0, 5), \quad \varepsilon \sim \mathcal{N}(0, 1),$$

with $\beta_2$ ranging from 0 (both models correct) to 0.16 (the linear model is misspecified). A linear model (A) and a quadratic model (B) are trained on 150 samples and evaluated on test sets of increasing size. The question is when we can detect, from the test errors, that B is better than A.

## Notebooks

| Notebook | Content |
|---|---|
| `0_General_Idea.ipynb` | Formal setting: trained model, validation metric, performance as a random variable |
| `1_Data_Generating_Process.ipynb` | Data-generating process (and a template to plug in your own) |
| `2_Models_fit.ipynb` | Fit both models and plot fits and errors |
| `0_Bootstrapping_Methods.ipynb` | Standard error of the mean: parametric, jackknife, bootstrap, BCa |
| `3_Statistical_Analysis.ipynb` | Confidence intervals for each model and for the difference; paired tests; p-values and power as $\beta_2$ varies |
| `Other_Simulation.ipynb` | Additional simulations |

Shared functions are in `Tool_Functions.py`. Some comments in the notebooks are in French.

## Run

```bash
pip install numpy pandas matplotlib scipy scikit-learn
jupyter notebook
```
