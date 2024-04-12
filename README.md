## momentGW

![CI](https://github.com/BoothGroup/momentGW/actions/workflows/ci.yaml/badge.svg)
![Code style](https://img.shields.io/badge/code%20style-black-000000.svg)

The `momentGW` code is a Python package for performing $GW$ approximation calculations, along with other related methods, using moment-conserving solutions to the Dyson equation.

#### Installation

The `momentGW` package, along with dependencies, can be installed as
```
git clone git@github.com:BoothGroup/momentGW.git
cd momentGW
python -m pip install . --user
```

#### Usage

The `momentGW` classes behave similarly to other post-mean-field method classes in PySCF. The `examples` directory contains examples for each solver.

#### Publications

The methods implemented in this package have been described in the following papers:
- [*"A 'moment-conserving' reformulation of GW theory"*](https://doi.org/10.1063/5.0143291)

The data presented in the publications can be found in the `benchmark` directory.
