# Utils

## ecg-image-kit

The [ecg_image_generator](./ecg_image_generator/) folder contains an enhanced version of the original [ecg-image-kit/ecg-image-generator](https://github.com/alphanumericslab/ecg-image-kit/tree/main/codes/ecg-image-generator) module. The enhancement mainly involves paths handling, and formatting using `pre-commit` hooks. The constants are kept in [a separate file](./ecg_image_generator/constants.py). A [folder containing json config files](./ecg_image_generator/Configs/) is also added to store the configurations for the `ecg_image_generator` module.

### Test of the original ecg-image-kit

```python
from scipy.stats import bernoulli
from extract_leads import get_paper_ecg

bernoulli_dc = bernoulli(0.2)
bernoulli_bw = bernoulli(0.2)
bernoulli_grid = bernoulli(1)
bernoulli_add_print = bernoulli(0.2)

get_paper_ecg(
    "/home/wenh06/Jupyter/wenhao/workspace/torch_ecg/sample-data/cinc2021/HR06000.mat",
    "/home/wenh06/Jupyter/wenhao/workspace/torch_ecg/sample-data/cinc2021/HR06000.hea",
    "/home/wenh06/Jupyter/Hot-data/cinc2024/",
    0,
    add_dc_pulse=bernoulli_dc,add_bw=bernoulli_bw,show_grid=bernoulli_grid,add_print=bernoulli_add_print,standard_colours=0
)
```

## ECG simulator

The EKF2 model for ECG denoising is described by the following system of ODEs

```math
\begin{equation}
\begin{cases}
    \dot{r} = r(1-r) \\
    \dot{\theta} = \omega \\
    \dot{z} = \sum\limits_{i\in\{P,Q,R,S,T\}} \dfrac{\alpha_i\omega}{b_i^2} \Delta\theta_i \exp{\left(-\dfrac{(\Delta\theta_i)^2}{2b_i^2}\right)}
\end{cases}
\end{equation}
```

where $r$ is the RR interval, $\theta$ is the phase, $z$ is the ECG data, $\omega$ is the angular frequency, $\alpha_i$ is the amplitude, $b_i$ is the standard deviation, and $\Delta\theta_i = \theta-\theta_i$.

The discrete form is

```math
\begin{equation}
\begin{cases}
    \theta_{k+1} = (\theta_k+\omega\delta) \mod{2\pi} \\
    z_{k+1} = -\sum\limits_{i\in\{P,Q,R,S,T\}} \delta \dfrac{\alpha_i\omega}{b_i^2} \Delta\theta_i \exp{\left(-\dfrac{(\Delta\theta_i)^2}{2b_i^2}\right)} + z_k + \eta
\end{cases}
\end{equation}
```

The ODE system has solution

```math
\begin{equation}
    z(t) = \sum\limits_{i\in\{P,Q,R,S,T\}} \alpha_i \exp{\left(-\dfrac{(\Delta\theta_i(t))^2}{2b_i^2}\right)} + const
\end{equation}
```

Let the loss (objective) function for the curve fitting be

```math
\begin{equation}
    L = \sqrt{\sum\limits_t (s(t)-z(t))^2}
\end{equation}
```

where $s(t)$ is the measurement value, $z(t)$ is the evovled value, at time $t$. Write $\Delta(t) = z(t)-s(t)$, then

```math
\begin{equation}
\begin{cases}
\dfrac{\partial L}{\partial \alpha_i} = \dfrac{\sum\limits_t \Delta(t) \exp{\left( -\dfrac{(\Delta\theta_i)^2}{2b_i^2} \right)}}{\sqrt{\sum\limits_t \Delta(t)^2}} \\
\dfrac{\partial L}{\partial b_i} = \dfrac{\sum\limits_t \Delta(t) \left( \dfrac{\alpha_i(\Delta\theta_i)^2}{b_i^3} \right) \exp{\left( -\dfrac{(\Delta\theta_i)^2}{2b_i^2} \right)}}{\sqrt{\sum\limits_t \Delta(t)^2}}
\end{cases}
\end{equation}
```

The EKF2 model can also be used to simulate the ECG data.

### References

1. Sameni R, Shamsollahi M B, Jutten C, et al. A nonlinear Bayesian filtering framework for ECG denoising[J]. IEEE Transactions on Biomedical Engineering, 2007, 54(12): 2172-2185.
2. Clifford G D, Shoeb A, McSharry P E, et al. Model-based filtering, compression and classification of the ECG[J]. International Journal of Bioelectromagnetism, 2005, 7(1): 158-161.
