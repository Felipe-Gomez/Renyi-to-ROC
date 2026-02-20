# Renyi-to-ROC
A fast implementation of the optimal conversion from a single Renyi Differential
Privacy (RDP) guarantee to a tradeoff function. Let $\alpha$ be the order of the Renyi
divergence and $\rho$ be the upper bound on the divergence. This repo implements the function
```get_FNRs(x, alpha, rho)```, which outputs the value of the tradeoff function at 
FPR value x. In particular:

If $\alpha > 1$, ```get_FNRs(x, alpha, rho)``` returns the smallest $y \in [0,1]$ satisfying:

$$\begin{aligned}
(1 - x)^{1-\alpha} y^\alpha + x^{1-\alpha} (1 - y)^\alpha
&\le e^{(\alpha-1)\rho} \\
y^{1-\alpha} (1 - x)^\alpha + (1 - y)^{1-\alpha} x^\alpha
&\le e^{(\alpha-1)\rho}.
\end{aligned}
$$

If $\alpha = 1$, ```get_FNRs(x, alpha, rho)``` returns the smallest $y \in [0,1]$ satisfying:

$$\begin{aligned}
y  \log \frac{y}{1 - x} + (1 - y)  \log \frac{1 - y}{x}
&\le \rho \\
x \log \frac{x}{1 - y} + (1 - x) \log \frac{1 - x}{y}
&\le \rho.
\end{aligned}$$

If $0 < \alpha < 1$, ```get_FNRs(x, alpha, rho)``` returns the smallest $y \in [0,1]$ satisfying:

$$\begin{aligned}
(1 - x)^{1-\alpha} y^\alpha + x^{1-\alpha} (1 - y)^\alpha
&\ge e^{(\alpha-1)\rho} \\
y^{1-\alpha} (1 - x)^\alpha + (1 - y)^{1-\alpha} x^\alpha
&\ge e^{(\alpha-1)\rho}.
\end{aligned}$$

For details on the proof of optimality,
see [[1](https://arxiv.org/abs/2602.04562)].

## Installation

```
pip install .
```

> Note: this package ships a pure-Python fallback and an optional Cython extension for speed.  
> By default `pip install .` will **try to compile** the Cython extension in an isolated build environment
> and fall back to the pure-Python implementation if compilation is not possible. 

Use the following flags if compilation is desired or not. 

### Skip compilation
```bash
# macOS / Linux
RENYI_TO_ROC_NO_COMPILE=1 pip install .

# Windows PowerShell
$env:RENYI_TO_ROC_NO_COMPILE = "1"; pip install .
```

### Require compilation

```bash
# macOS / Linux
RENYI_TO_ROC_REQUIRE_COMPILE=1 pip install .

# Windows PowerShell
$env:RENYI_TO_ROC_REQUIRE_COMPILE = "1"; pip install .
```

If the build environment lacks the necessary build tools or headers, 
installation will raise an error.



## Example Usage 


``` python
import numpy as np
from renyi_to_roc import get_FNR
alpha = 2
rho = 1.1
x_array = np.linspace(0,1,20)
fnr_array = get_FNR(x_array, alpha = alpha, rho = rho)
```
An optional ```tol``` parameter is allowed, which sets the tolerance
for computing ```fnr_array```. Default is ```1e-7```, which means 
every element in ```fnr_array``` is computed to 7 digits of accuracy. 

If an RDP profile $\rho(\alpha)$ is known, then we recommend calling
```get_FNR``` over a grid of orders $\alpha$ then taking the maximum to
get the resulting tradeoff curve. 

As an example, we investigate the tradeoff curve 
for $\rho$-zCDP. Since $\rho$-zCDP is defined only for orders $\alpha>1$, 
we construct our grid of alphas to this regime:

```python
alphas = np.linspace(1, 10, 100) # explicitly omit alpha < 1
rho_zcdp = 1.1
rhos = rho_zcdp * alphas
tol = 1e-7
x_array = np.linspace(0,1,1_000)
fnr_array = np.zeros_like(x)

for alpha, rho in zip(alphas, rhos):
    fnr_array = np.maximum(get_FNR(x_array, alpha = alpha, rho = rho, tol = tol), fnr_array)
```
## References

[1] [Anneliese Riess, Juan Felipe Gomez, Flavio du Pin Calmon, Julia Anne Schnabel, Georgios Kaissis. 2026. Optimal conversion from Rényi Differential Privacy to f-Differential Privacy](https://arxiv.org/abs/2602.04562)
