from pyscf import gto, dft, gw
mol = gto.M(
        atom="O 0 0 0; O 0 0 1",
        basis="cc-pvdz",
        verbose=5,
)
mf = dft.RKS(mol)
mf.xc = 'pbe'
mf.kernel()


from pyscf import tdscf
nocc = mol.nelectron//2
nmo = mf.mo_energy.size
nvir = nmo-nocc
td = tdscf.TDDFT(mf)
td.nstates = nocc*nvir
td.verbose = 0
td.kernel()

gw = gw.GW(mf, freq_int='exact', tdmf=td)
gw.kernel()
print(gw.mo_energy)