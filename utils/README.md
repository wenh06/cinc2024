# Utils

## Test of ecg-image-kit

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
