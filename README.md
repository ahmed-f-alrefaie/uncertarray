# uncertarray

[![Release](https://img.shields.io/github/v/release/ahmed-f-alrefaie/uncertarray)](https://img.shields.io/github/v/release/ahmed-f-alrefaie/uncertarray)
[![Build status](https://img.shields.io/github/actions/workflow/status/ahmed-f-alrefaie/uncertarray/main.yml?branch=main)](https://github.com/ahmed-f-alrefaie/uncertarray/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/ahmed-f-alrefaie/uncertarray/branch/main/graph/badge.svg)](https://codecov.io/gh/ahmed-f-alrefaie/uncertarray)
[![Commit activity](https://img.shields.io/github/commit-activity/m/ahmed-f-alrefaie/uncertarray)](https://img.shields.io/github/commit-activity/m/ahmed-f-alrefaie/uncertarray)
[![License](https://img.shields.io/github/license/ahmed-f-alrefaie/uncertarray)](https://img.shields.io/github/license/ahmed-f-alrefaie/uncertarray)

Numpy arrays with uncertainty propogation

- **Github repository**: <https://github.com/ahmed-f-alrefaie/uncertarray/>
- **Documentation** <https://ahmed-f-alrefaie.github.io/uncertarray/>


## Features

- Numpy array-like structure with uncertainty propagation
- Supports basic arithmetic operations with uncertainty handling
- Integration with NumPy ufuncs for element-wise operations


## Examples

```python
import numpy as np
from uncertarray import uncertarray

# Create uncertarray
x = uncertarray([1.0, 2.0, 3.0], [0.1, 0.2, 0.3])
y = uncertarray([4.0, 5.0, 6.0], [0.4, 0.5, 0.6])

# Perform arithmetic operations
z = x + y
print(z)  # Output: uncertarray with propagated uncertainties
# Use NumPy ufuncs
w = np.sqrt(x)
print(w)  # Output: uncertarray with propagated uncertainties
```

**This is still in early development. More features and improvements are coming soon!**
