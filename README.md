## momentGW

![CI](https://github.com/BoothGroup/momentGW/actions/workflows/ci.yaml/badge.svg)
![Code style](https://img.shields.io/badge/code%20style-black-000000.svg)

The `momentGW` code is a Python package for performing calculations within the *GW* approximation, along with other associated methods, using moment-conserving solutions to the Dyson equation.
A diverse range of self-consistent schemes are available, along with dTDA and dRPA polarizabilities, unrestricted and/or periodic boundary conditions, tensor hypercontraction, optical excitations, and more.

### Installation

The `momentGW` package, along with dependencies, can be installed as
```bash
git clone https://github.com/BoothGroup/momentGW.git
cd momentGW
python -m pip install . --user
```

### Usage

The `momentGW` solvers are built on top of the [PySCF](https://github.com/pyscf/pyscf) package, and the classes behave similarly to other post-mean-field method classes in PySCF, e.g.:
```python
from pyscf import gto, scf
from momentGW import GW
mol = gto.M(atom="H 0 0 0; Li 0 0 1.64", basis="6-31g")
mf = scf.RHF(mol).run()
gw = GW(mf)
gw.kernel(nmom_max=3)
```
The `examples` directory contains more detailed usage examples.

### Publications

The methods implemented in this package have been described in the following papers:
- [*"A 'moment-conserving' reformulation of GW theory"*](https://doi.org/10.1063/5.0143291)

The data presented in the publications can be found in the `benchmark` directory.

### Contributing

Contributions are welcome, and can be made by submitting a pull request to the `master` branch.
The code uses [NumPy-style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html) and is formatted using [`black`](https://black.readthedocs.io/en/stable/), [`isort`](https://pycqa.github.io/isort/), [`ssort`](https://github.com/bwhmather/ssort), and [`flake8`](https://flake8.pycqa.org/en/latest/).
