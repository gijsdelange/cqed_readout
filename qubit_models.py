import numpy as np
from scipy.misc import factorial
from scipy.integrate import quad, simps
from scipy.special import hermite, eval_hermite
from scipy.sparse.linalg import eigsh
import itertools

sx = np.array([[0, 1], [1, 0]], complex)
sy = np.array([[0 , -1j], [1j, 0]])
sz = np.array([[1, 0], [0, -1]], complex)
s0 = np.array([[1, 0], [0, 1]], complex)

def inner(v1, v2):
    return np.dot(np.conj(v1), v2)

def dispersion(spectrum, k=0):
    if spectrum.ndim == 1:
        return np.ptp(spectrum)
    else:
        return np.ptp(spectrum[k], axis = -1)

def average(spectrum, k=0):
    if spectrum.ndim == 1:
        return np.average(spectrum)
    else:
        return np.average(spectrum[k], axis = -1)

def difference(spectrum, i, j):
    return spectrum[j] - spectrum[i]

def row_indices(N, M):
    return np.outer(np.arange(N), np.ones(M)).astype(int)

def column_indices(N, M):
    return row_indices(N, M).T

def hermite_norm(n):
    return (np.pi**0.25 * np.sqrt(2.**n * factorial(n)))

def optimize(qubit, K, tol=1e-10):
    # Finds optimal size of basis for convergence of first K eigenvalues
    qubit.N = K
    evals = np.linalg.eigvalsh(qubit.hamiltonian())[:K]
    error = 1
    count = 0
    while error > tol:
        count += 1
        qubit.N += 1
        new_evals = np.linalg.eigvalsh(qubit.hamiltonian())[:K]
        error = np.linalg.norm(new_evals - evals)
        evals = new_evals
    qubit.K = K
    qubit.optimized = True
    qubit.tol = tol


class Fluxonium:
    def __init__(self, parameters):
        for (key, value) in parameters.items():
            setattr(self, key, value)
        self.optimized = False
        self.D = 1000
        self.phimax = 10
        if not hasattr(self, 'majoranas'):
            self.majoranas = False


    def omega_LC(self):
        return np.sqrt(8 * self.Ec * self.El)

    def rescaling_factor(self):
        return (8 * self.Ec / self.El)**0.25

    def potential_energy(self, phi):
        return self.potential(phi + self.flux) + 0.5 * self.El * phi**2

    def number_operator(self):
        sqrts = np.sqrt(np.arange(1, self.N))
        s = self.rescaling_factor()
        nop = (np.sqrt(2) * 1j / s) * (np.diag(sqrts, k=1) - np.diag(sqrts, k=-1))
        return np.kron(nop, s0) if self.majoranas else nop

    def phase_operator(self):
        sqrts = np.sqrt(np.arange(1, self.N))
        s = self.rescaling_factor()
        phase_op = (s / np.sqrt(2)) * (np.diag(sqrts, k=1) + np.diag(sqrts, k=-1))
        return np.kron(phase_op, s0) if self.majoranas else phase_op

    def potential_operator(self, potential):
        grid = np.linspace(-self.phimax, self.phimax, self.D)
        r = np.tile(row_indices(self.N, self.N), (self.D, 1, 1))
        c = np.tile(column_indices(self.N, self.N), (self.D, 1, 1))
        phis = np.rollaxis(np.tile(grid, (self.N, self.N, 1)), 2)
        fs = np.exp(-phis**2) * eval_hermite(r, phis) * eval_hermite(c, phis)
        fs *= potential(self.rescaling_factor() * phis + self.flux)
        fs /= hermite_norm(r) * hermite_norm(c)
        vop = simps(fs, np.linspace(-self.phimax, self.phimax, self.D), axis=0)
        return vop

    def hamiltonian(self):
        ham = self.omega_LC() * np.diag(np.arange(self.N) + 0.5)
        ham += self.potential_operator(self.potential)
        if self.majoranas:
            ham = np.kron(ham, s0)
            vx, vy, vz = self.majorana_couplings()
            ham += vx * np.kron(np.eye(self.N), sx)
            ham += np.kron(self.potential_operator(vy), sy)
            ham += np.kron(self.potential_operator(vz), sz)
        return ham

    def majorana_couplings(self):
        P = -1. if self.parity =='odd' else 1.
        E = self.couplings
        S = self.shifts
        vx = - (E[0,1] + P * E[2, 3])
        vy = lambda phi: (E[0,2] * np.cos((phi - S[0,2])/2)
                                + P * E[1,3] * np.cos((phi - S[1,3])/2))
        vz = lambda phi: - (E[1,2] * np.cos((phi - S[1,2])/2)
                                + P * E[0,3] * np.cos((phi - S[0,3])/2))
        return vx, vy, vz

    def energies(self, return_evecs=False):
        if return_evecs:
            evals, evecs = np.linalg.eigh(self.hamiltonian())
            return evals, evecs
        else:
            evals = np.linalg.eigvalsh(self.hamiltonian())
            return evals

    def energies_vs_flux(self, fluxes, subtract_minimum=True):
        energies = []
        for flux in fluxes:
            self.flux = flux
            energies.append(self.energies())
        energies = np.vstack(energies).T
        if subtract_minimum == True:
            energies -= np.min(energies)
        return energies

    def number_matrix_elements(self, evecs):
        nop = self.number_operator()
        return inner(evecs.T, nop.dot(evecs))

    def phase_matrix_elements(self, evecs):
        phase = self.phase_operator()
        return inner(evecs.T, phase.dot(evecs))


class ChargeQubit:
    def __init__(self, parameters):
        for (key, value) in parameters.items():
            setattr(self, key, value)
        self.D = 1001 # number of samples for numerical integration with simps

    def size(self):
        return 2 * self.N + 1

    def numbers(self):
        return np.arange(-self.N, self.N + 1, 1, dtype=int)

    def number_operator(self):
        nop = np.diag(self.numbers()) + 0j
        return nop

    def josephson_potential_coefficients(self, tol=1e-10):
        grid = np.linspace(0, 2*np.pi, self.D)
        S = np.ceil((self.N+1)/2).astype(int) # Number of coefficients we take into account
        ns = np.tile(np.arange(S), (self.D, 1)).T
        phis = np.tile(grid, (S, 1))
        re = simps(self.potential(phis) * np.cos(ns * phis) / (2*np.pi),
                   grid, axis=1)
        im = simps(self.potential(phis) * np.sin(ns * phis) / (2*np.pi),
                   grid, axis=1)
        re[np.abs(re) < tol] = 0
        im[np.abs(im) < tol] = 0
        return re - 1j * im

    def josephson_potential_operator(self):
        cs = self.josephson_potential_coefficients()
        vop = np.diag(np.full(self.size(), cs[0]))
        js = np.nonzero(cs[1:])[0] + 1
        for j in js:
            vop += cs[j] * np.eye(self.size(), k=-2*j)
            vop += np.conj(cs[j]) * np.eye(self.size(), k=2*j)
        return vop

    def majorana_hamiltonian(self):
        P, E, S = self.parity, self.couplings, self.shifts
        C = E * np.exp(1j * S)
        signs = (-1)**np.abs(self.numbers())
        HM = 0.5 * (E[0,1] + P * E[2,3]) * np.diag(signs) + 0j
        HM -= 0.5 * (C[1,2] + P * C[0,3]) * np.eye(self.size(), k=-1)
        HM -= 0.5 * (C[1,2] + P * C[0,3]) * np.eye(self.size(), k=1)
        signs = (-1)**np.abs(self.numbers()[1:])
        HM -=  0.5 * 1j * (C[0,2] + P * C[1,3]) * np.diag(signs, k=-1)
        HM +=  0.5 * 1j * (C[0,2] + P * C[1,3]) * np.diag(signs, k=1)
        return HM

    def hamiltonian(self):
        N = self.number_operator() - self.ng * np.eye(self.size())
        V = self.josephson_potential_operator()
        HM = self.majorana_hamiltonian() if self.majoranas else 0
        return self.Ec * N.dot(N) + V + HM

    def dipole_matrix_elements(self, evecs):
        nop = self.number_operator()
        return inner(evecs.T, nop.dot(evecs))

    def use_cosine_potential(self):
        self.potential = lambda phi: self.Ej * (1 - np.cos(phi))

    def use_abs_potential(self):
        def abs_potential(phi):
            V = 0
            for T in self.transmission:
                V += self.gap * (1 - np.sqrt(1 - T * np.sin(phi/2)**2))
            return V
        self.potential = abs_potential

    def energies(self, branch='none', return_evecs=False):
        if return_evecs:
            evals, evecs = np.linalg.eigh(self.hamiltonian())
        else:
            evals = np.linalg.eigvalsh(self.hamiltonian())
        if branch is not 'none':
            ns = np.arange(self.size())
            es = np.where(ns[::2] % 4 != 0, ns[::2] + 1, ns[::2])[:-1]
            os = ns[np.isin(ns, es, invert=True)][:-1]
            ngmin = 0 if branch == 'even' else 1
            evals = evals[es] if np.rint(self.ng) % 2 == ngmin else evals[os]
            if return_evecs:
                evecs = evecs[es] if np.rint(self.ng) % 2 == ngmin else evecs[os]
        return (evals, evecs) if return_evecs else evals


    def energies_vs_ng(self, ngs, branch='none', subtract_minimum=True):
        energies = []
        for (i, ng) in enumerate(ngs):
            self.ng = ng
            energies.append(self.energies(branch))
        energies = np.vstack(energies).T
        if subtract_minimum == True:
            energies -= np.min(energies)
        return energies
    
    
    def energies_vs_flux(self, fluxes, branch='none', subtract_minimum=True):
        energies = []
        for (i, flux) in enumerate(fluxes):
            self.flux = flux
            if self.majoranas == True:
                self.shifts[0,3] = flux/2
            energies.append(self.energies(branch))
        energies = np.vstack(energies).T
        if subtract_minimum == True:
            energies -= np.min(energies)
        return energies


    def frequency_spectrum(self, ngs, freqs, initial_states=[0],
                           kappa=0.1, branch='none'):
        """
        Returns an image of the frequency spectrum of the system
        as a function of ng.

        Parameters:
        -----------
            ngs:
                list of induced charges to be included in the plot.
            freqs:
                list of frequencies to be included in the plot.
            initial_states:
                list of initial states to be included in the frequency spectrum.
                For instance, if [0,1] the plot includes all transitions starting
                from ground and excited states.
            kappa:
                Linewidth (same for every transition, for now
                it's just a convenience parameter with no physical input).
        """
        spectrum = np.zeros((len(ngs), len(freqs)))
        for (n, ng) in enumerate(ngs):
            self.ng = ng
            evals, evecs = self.energies(branch, return_evecs=True)
            g = self.dipole_matrix_elements(evecs)
            for i in initial_states:
                omegas = evals[i+1:] - evals[i]
                for (j, omega) in enumerate(omegas):
                    spectrum[n] += (0.25 * np.abs(g[i,i+j+1])**2 * kappa**2
                                    / ((freqs-omega)**2 + 0.25 * kappa**2))
        return spectrum
    
    def frequency_spectrum_vs_flux(self, fluxes, freqs, initial_states=[0],
                                   kappa=0.1, branch='none'):
        spectrum = np.zeros((len(fluxes), len(freqs)))
        for (n, flux) in enumerate(fluxes):
            self.flux = flux
            self.shifts[0,3] = flux/2
            evals, evecs = self.energies(branch, return_evecs=True)
            g = self.dipole_matrix_elements(evecs)
            for i in initial_states:
                omegas = evals[i+1:] - evals[i]
                for (j, omega) in enumerate(omegas):
                    spectrum[n] += (0.25 * np.abs(g[i,i+j+1])**2 * kappa**2
                                    / ((freqs-omega)**2 + 0.25 * kappa**2))
        return spectrum
