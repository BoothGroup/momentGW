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

The `AGW` class behaves similarly to other post-mean-field method classes in PySCF. The file `example.py` has an example of a calculation compared to one of a standard GW implementation.
