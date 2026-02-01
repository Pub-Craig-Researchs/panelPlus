# panelPlus

[![MATLAB](https://img.shields.io/badge/Language-MATLAB-orange.svg)](https://www.mathworks.com/products/matlab.html)
[![Version](https://img.shields.io/badge/Version-Refined-blue.svg)](#)

**panelPlus** is an advanced econometric library refined from the original *A Panel Data Toolbox for MATLAB*. This version extends the classical panel data framework by integrating modern machine learning algorithms and robust causal inference methods.

---

## Key Enhancements

### 1. Architectural Reclassification
The functions have been logically reorganized into specialized packages to improve workflow efficiency:
* **timeseries/**: Tools for temporal dependencies and dynamic panel analysis.
* **machine_learning/**: High-dimensional modeling and predictive analysis.
* **causal_estimation/**: Modern identification strategies and treatment effect estimation.

### 2. New Econometric Estimators
This refined version introduces several state-of-the-art methods:
* **DDML**: Double Debiased Machine Learning for high-dimensional nuisance parameter handling.
* **Panel FGLS**: Feasible Generalized Least Squares for robust estimation under non-spherical errors.
* **Two-way Clustering**: Robust inference providing standard errors clustered by both entity and time.
* **Causal Forest**: Implementation of Generalized Random Forests for heterogeneous treatment effects.

---

## Development & Credits

> **Note on Implementation**: 
> The code in this repository was developed and optimized under the assistance of the **Claude LLM Agent**, focusing on structural integrity and computational efficiency.

* **Original Reference**: Álvarez, I., Barbero, J., & Zofío, J. L. (2017). "A Panel Data Toolbox for MATLAB." *Journal of Statistical Software*.

---

