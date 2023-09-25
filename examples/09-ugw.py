"""Example of running GW calculations with unrestricted references.
"""

from pyscf import dft, gto

from momentGW import UGW, evUGW, qsUGW, scUGW

mol = gto.M(
        atom="Be 0 0 0; H 0 0 1.6",
        basis="6-31g",
        spin=1,
        verbose=5,
)

mf = dft.UKS(mol)
mf = mf.density_fit()
mf.xc = "hf"
mf.kernel()

# G0W0
gw = UGW(mf)
gw.kernel(nmom_max=5)

# evGW
evgw = evUGW(mf)
evgw.kernel(nmom_max=3)

# qsGW
qsgw = qsUGW(mf)
qsgw.kernel(nmom_max=1)

# scGW
scgw = scUGW(mf)
scgw.kernel(nmom_max=3)
