"""
k-points helper utilities.
"""

import itertools

import numpy as np
import scipy.linalg
from pyscf import lib
from pyscf.agf2 import SelfEnergy, GreensFunction
from pyscf.pbc.lib import kpts_helper
from dyson import Lehmann

# TODO make sure this is rigorous


def allow_single_kpt(output_is_kpts=False):
    """
    Decorator to allow `kpts` arguments to be passed as a single
    k-point.
    """

    def decorator(func):
        def wrapper(self, kpts, *args, **kwargs):
            shape = kpts.shape
            kpts = kpts.reshape(-1, 3)
            res = func(self, kpts, *args, **kwargs)
            if output_is_kpts:
                return res.reshape(shape)
            else:
                return res

        return wrapper

    return decorator


class KPoints:
    def __init__(self, cell, kpts, tol=1e-8, wrap_around=True):
        self.cell = cell
        self.tol = tol

        if wrap_around:
            kpts = self.wrap_around(kpts)
        self._kpts = kpts

        self._kconserv = kpts_helper.get_kconserv(cell, kpts)
        self._kpts_hash = {self.hash_kpts(kpt): k for k, kpt in enumerate(self._kpts)}

    @allow_single_kpt(output_is_kpts=True)
    def get_scaled_kpts(self, kpts):
        """
        Convert absolute k-points to scaled k-points for the current
        cell.
        """
        return self.cell.get_scaled_kpts(kpts)

    @allow_single_kpt(output_is_kpts=True)
    def get_abs_kpts(self, kpts):
        """
        Convert scaled k-points to absolute k-points for the current
        cell.
        """
        return self.cell.get_abs_kpts(kpts)

    @allow_single_kpt(output_is_kpts=True)
    def wrap_around(self, kpts, window=(-0.5, 0.5)):
        """
        Handle the wrapping of k-points into the first Brillouin zone.
        """

        kpts = self.get_scaled_kpts(kpts) % 1.0
        kpts = lib.cleanse(kpts, axis=0, tol=self.tol)
        kpts = kpts.round(decimals=self.tol_decimals) % 1.0

        kpts[kpts < window[0]] += 1.0
        kpts[kpts >= window[1]] -= 1.0

        kpts = self.get_abs_kpts(kpts)

        return kpts

    @allow_single_kpt(output_is_kpts=False)
    def hash_kpts(self, kpts):
        """
        Convert k-points to a unique, hashable representation.
        """
        return tuple(np.rint(kpts / (self.tol)).ravel().astype(int))

    @property
    def tol_decimals(self):
        """Convert the tolerance into a number of decimal places."""
        return int(-np.log10(self.tol + 1e-16)) + 2

    def conserve(self, ki, kj, kk):
        """
        Get the index of the k-point that conserves momentum.
        """
        return self._kconserv[ki, kj, kk]

    def loop(self, depth):
        """
        Iterate over all combinations of k-points up to a given depth.
        """
        if depth == 1:
            yield from enumerate(self)
        else:
            yield from itertools.product(enumerate(self), repeat=depth)

    @allow_single_kpt(output_is_kpts=False)
    def is_zero(self, kpts):
        """
        Check if the k-point is zero.
        """
        return np.max(np.abs(kpts)) < self.tol

    @property
    def kmesh(self):
        """Guess the k-mesh.
        """
        kpts = self.get_scaled_kpts(self._kpts).round(self.tol_decimals)
        kmesh = [len(np.unique(kpts[:, i])) for i in range(3)]
        return kmesh

    def translation_vectors(self):
        """
        Translation vectors to construct supercell of which the gamma
        point is identical to the k-point mesh of the primitive cell.
        """

        kmesh = self.kmesh

        r_rel = [np.arange(kmesh[i]) for i in range(3)]
        r_vec_rel = lib.cartesian_prod(r_rel)
        r_vec_abs = np.dot(r_vec_rel, self.cell.lattice_vectors())

        return r_vec_abs

    def interpolate(self, other, fk):
        """
        Interpolate a function `f` from the current grid of k-points to
        those of `other`. Input must be in a localised basis, i.e. AOs.

        Parameters
        ----------
        other : KPoints
            The k-points to interpolate to.
        fk : numpy.ndarray or lis
            The function to interpolate, expressed on the current
            k-point grid. Can be a matrix-valued array expressed in
            k-space, a list of `SelfEnergy` or `GreensFunction` objects
            from `pyscf.agf2`, or a list of `dyson.Lehmann` objects.
            Matrix values or couplings *must be in a localised basis*.
        """

        if len(other) % len(self):
            raise ValueError("Size of destination k-point mesh must be divisible by the size of the source k-point mesh for interpolation.")
        nimg = len(other) // len(self)

        r_vec_abs = self.translation_vectors()
        kR = np.exp(1.0j * np.dot(self._kpts, r_vec_abs.T)) / np.sqrt(len(r_vec_abs))

        r_vec_abs = other.translation_vectors()
        kL = np.exp(1.0j * np.dot(other._kpts, r_vec_abs.T)) / np.sqrt(len(r_vec_abs))

        if isinstance(fk, np.ndarray):
            nao = fk.shape[-1]

            # k -> bvk
            fg = lib.einsum("kR,kij,kS->RiSj", kR, fk, kR.conj())
            if np.max(np.abs(fg.imag)) > 1e-6:
                raise ValueError("Interpolated function has non-zero imaginary part.")
            fg = fg.real
            fg = fg.reshape(len(self)*nao, len(self)*nao)

            # tile in bvk
            fg = scipy.linalg.block_diag(*[fg for i in range(nimg)])

            # bvk -> k
            fg = fg.reshape(len(other), nao, len(other), nao)
            fl = lib.einsum("kR,RiSj,kS->kij", kL.conj(), fg, kL)

        else:
            assert all(isinstance(f, (SelfEnergy, GreensFunction, Lehmann)) for f in fk)
            assert len({type(f) for f in fk}) == 1
            ek = np.array([f.energies if isinstance(f, Lehmann) else f.energy for f in fk])
            vk = np.array([f.couplings if isinstance(f, Lehmann) else f.coupling for f in fk])

            # k -> bvk
            eg = ek
            vg = lib.einsum("kR,kpx->Rpx", kR, vk)

            # tile in bvk
            eg = np.concatenate([eg] * nimg, axis=0)
            vg = np.concatenate([vg] * nimg, axis=0)

            # bvk -> k
            el = eg
            vl = lib.einsum("kR,Rpx->kpx", kL.conj(), vg)  # TODO correct conjugation?
            fl = [fk[0].__class__(e, v) for e, v in zip(el, vl)]

        return fl

    def member(self, kpt):
        """
        Find the index of the k-point in the k-point list.
        """
        if kpt not in self:
            raise ValueError(f"{kpt} is not in list")
        return self._kpts_hash[self.hash_kpts(kpt)]

    index = member

    def __contains__(self, kpt):
        """
        Check if the k-point is in the k-point list.
        """
        return self.hash_kpts(kpt) in self._kpts_hash

    def __getitem__(self, index):
        """
        Get the k-point at the given index.
        """
        return self._kpts[index]

    def __len__(self):
        """
        Get the number of k-points.
        """
        return len(self._kpts)

    def __iter__(self):
        """
        Iterate over the k-points.
        """
        return iter(self._kpts)

    def __repr__(self):
        """
        Get a string representation of the k-points.
        """
        return repr(self._kpts)

    def __str__(self):
        """
        Get a string representation of the k-points.
        """
        return str(self._kpts)

    def __array__(self):
        """
        Get the k-points as a numpy array.
        """
        return np.asarray(self._kpts)


if __name__ == "__main__":
    from pyscf.pbc import gto, scf
    from pyscf.agf2 import chempot
    from momentGW import KGW

    nmom_max = 5
    r = 1.0
    vac = 25.0

    cell = gto.Cell()
    cell.atom = "H 0 0 0; H 0 0 %.6f" % r
    cell.a = np.array([[vac, 0 ,0], [0, vac, 0], [0, 0, r*2]])
    cell.basis = "sto6g"
    cell.max_memory = 1e10
    cell.build()

    kmesh1 = [1, 1, 3]
    kmesh2 = [1, 1, 6]
    kpts1 = cell.make_kpts(kmesh1)
    kpts2 = cell.make_kpts(kmesh2)

    mf1 = scf.KRHF(cell, kpts1)
    mf1 = mf1.density_fit(auxbasis="weigend")
    mf1.exxdiv = None
    mf1.conv_tol = 1e-10
    mf1.kernel()

    mf2 = scf.KRHF(cell, kpts2)
    mf2 = mf2.density_fit(mf1.with_df.auxbasis)
    mf2.exxdiv = mf1.exxdiv
    mf2.conv_tol = mf1.conv_tol
    mf2.kernel()

    gw1 = KGW(mf1)
    gw1.polarizability = "dtda"
    gw1.compression_tol = 1e-100
    gw1.kernel(nmom_max)
    gf1 = gw1.gf
    se1 = gw1.se

    gw2 = KGW(mf2)
    gw2.polarizability = "dtda"
    gw2.compression_tol = 1e-100

    kpts1 = KPoints(cell, kpts1)
    kpts2 = KPoints(cell, kpts2)

    # Interpolate via the auxiliaries
    for k in range(len(kpts1)):
        se1[k].coupling = np.dot(mf1.mo_coeff[k], se1[k].coupling)
    se2a = kpts1.interpolate(kpts2, se1)
    sc = lib.einsum("kpq,kqi->kpi", np.array(mf1.get_ovlp()), np.array(mf1.mo_coeff))
    for k in range(len(kpts1)):
        se1[k].coupling = np.dot(sc[k].T.conj(), se1[k].coupling)
    sc = lib.einsum("kpq,kqi->kpi", np.array(mf2.get_ovlp()), np.array(mf2.mo_coeff))
    for k in range(len(kpts2)):
        se2a[k].coupling = np.dot(sc[k].T.conj(), se2a[k].coupling)
    gf2a = [s.get_greens_function(f) for s, f in zip(se2a, gw2.build_se_static())]
    for k in range(len(kpts2)):
        cpt, error = chempot.binsearch_chempot(
            (gf2a[k].energy, gf2a[k].coupling),
            gf2a[k].nphys,
            gw2.nocc[k] * 2,
        )
        gf2a[k].chempot = cpt

    # Interpolate via the moments
    np.set_printoptions(edgeitems=1000, linewidth=1000, precision=4)
    th1 = np.array([s.get_occupied().moment(range(nmom_max+1)) for s in se1])
    tp1 = np.array([s.get_virtual().moment(range(nmom_max+1)) for s in se1])
    print(th1[1, 1].real, "\n")
    th1 = lib.einsum("knij,kpi,kqj->nkpq", th1, np.array(mf1.mo_coeff), np.array(mf1.mo_coeff).conj())
    tp1 = lib.einsum("knij,kpi,kqj->nkpq", tp1, np.array(mf1.mo_coeff), np.array(mf1.mo_coeff).conj())
    print(th1[1, 1].real, "\n")
    th2 = np.array([kpts1.interpolate(kpts2, t) for t in th1])
    tp2 = np.array([kpts1.interpolate(kpts2, t) for t in tp1])
    print(th2[1, kpts2.index(kpts1[1])].real, "\n")
    sc = lib.einsum("kpq,kqi->kpi", np.array(mf2.get_ovlp()), np.array(mf2.mo_coeff))
    th2 = lib.einsum("nkpq,kpi,kqj->knij", th2, sc.conj(), sc)
    tp2 = lib.einsum("nkpq,kpi,kqj->knij", tp2, sc.conj(), sc)
    print(th2[kpts2.index(kpts1[1]), 1].real, "\n")
    gf2b, se2b = gw2.solve_dyson(th2, tp2, gw2.build_se_static())

    for g in gf1:
        g.remove_uncoupled(tol=0.5)
    for g in gf2a:
        g.remove_uncoupled(tol=0.5)
    for g in gf2b:
        g.remove_uncoupled(tol=0.5)
    print(gf1[0].energy)
    print(gf2a[0].energy)
    print(gf2b[0].energy)
    assert len({len(g.energy) for g in gf1}) == 1
    assert len({len(g.energy) for g in gf2a}) == 1
    assert len({len(g.energy) for g in gf2b}) == 1

    print("%8s %12s %12s %12s" % ("k-point", "original", "via aux", "via moms"))
    for k in range(len(kpts2)):
        if kpts2[k] in kpts1:
            k1 = kpts1.index(kpts2[k])
            gaps = [
                gf1[k1].get_virtual().energy[0] - gf1[k1].get_occupied().energy[-1],
                gf2a[k].get_virtual().energy[0] - gf2a[k].get_occupied().energy[-1],
                gf2b[k].get_virtual().energy[0] - gf2b[k].get_occupied().energy[-1],
            ]
            print("%8d %12.6f %12.6f %12.6f" % (k, *gaps))
        else:
            gaps = [
                gf2a[k].get_virtual().energy[0] - gf2a[k].get_occupied().energy[-1],
                gf2b[k].get_virtual().energy[0] - gf2b[k].get_occupied().energy[-1],
            ]
            print("%8d %12s %12.6f %12.6f" % (k, "", *gaps))

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(kpts1[:, 2], np.array([g.energy for g in gf1]), "C0o", label="original")
    plt.plot(kpts2[:, 2], np.array([g.energy for g in gf2a]), "C1o", label="via aux")
    plt.plot(kpts2[:, 2], np.array([g.energy for g in gf2b]), "C2o", label="via moments")
    plt.show()
