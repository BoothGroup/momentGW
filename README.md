## Moment-conserving GW

This repository contains the code and implementation for the paper *"A 'moment-conserving' reformulation of GW theory"*.

#### Installation

Install [Vayesta](https://github.com/BoothGroup/Vayesta), which includes other dependencies such as PySCF and NumPy:
```
git clone git@github.com:BoothGroup/Vayesta.git
python -m pip install . --user
```

Install moment-conserving Dyson equation solver:
```
git clone git@github.com:BoothGroup/dyson.git
```

Clone the `momentGW` repository:
```
git clone git@github.com:BoothGroup/momentGW.git --depth 1
```

#### Usage

The `momentGW` classes behave similarly to other post-mean-field method classes in PySCF. The `examples` directory contains examples for each solver.
