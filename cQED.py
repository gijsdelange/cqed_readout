import numpy as np
import qutip as qt
import sys
#import cStringIO

# Hilbert space truncation used to compute transmon levels
NMAX = 20


# Class to pass parameters to solvers.
class SimpleNamespace(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


# Transmon spectrum
def split_jj(Ej, phi, d):
    return np.abs(Ej * np.cos(np.pi*phi) *
                  np.sqrt(1 + (d*np.tan(np.pi*phi))**2))


def transmon_levels(p):
    if p.flux is not None:
        Ej = split_jj(p.Ej, p.flux, p.d)
    else:
        Ej = p.Ej

    H = (np.diag(4 * p.Ec * (np.arange(-NMAX, NMAX+1) - p.ng)**2) +
         0.5 * Ej * (np.diag(-np.ones(2*NMAX), 1) +
         np.diag(-np.ones(2*NMAX), -1)))

    en = qt.Qobj(H).eigenenergies()[:p.J]
    return en-en[0]


# Some useful operators
def a(p):
    return qt.tensor(qt.destroy(p.N), qt.qeye(p.J))


def Jm(p):
    j_destroy = np.diag(np.sqrt(np.arange(1, p.J)), 1)
    return qt.tensor(qt.qeye(p.N), qt.Qobj(j_destroy))


def Jp(p):
    return Jm(p).dag()


# list of interesting observables to be computed
def op_list(p):
    return [a(p).dag()*a(p), Jp(p)*Jm(p), a(p) + a(p).dag(), a(p)]


# Jaynes-Cummings Hamiltonian
def jaynes_cummings(p, driven=True):

    xi = (p.xi if driven else 0.)
    wd = (p.wd if driven else 0.)

    wjs = transmon_levels(p) - wd * np.arange(0, p.J)
    wr = p.wr - wd

    cavity = wr * a(p).dag() * a(p) + xi * (a(p).dag() + a(p))
    transmon = qt.tensor(qt.qeye(p.N), qt.Qobj(np.diag(wjs)))
    coupling = p.g * (a(p) * Jp(p) + a(p).dag() * Jm(p))

    return cavity + transmon + coupling


# Obtain frequencies from the bare spectrum.
def transition_frequencies(p):
    evals, evecs = jaynes_cummings(p, driven=False).eigenstates()
    j = qt.expect(Jp(p) * Jm(p), evecs)[1:]
    n = qt.expect(a(p).dag() * a(p), evecs)[1:]
    freqs = (evals[1:]-evals[0])/(n+j)
    return zip(np.around(n+j), freqs)


# Decoherence terms.
def betas(p):
    en = transmon_levels(p)
    return np.array([2 * i / en[1] for i in en])


def decoherence(p):
    jumps = []
    if p.kappa:
        jumps.append(np.sqrt(p.kappa) * a(p))
    if p.gamma:
        jumps.append(np.sqrt(p.gamma) * Jm(p))
    if p.gamma_phi:
        jumps.append(np.sqrt(p.gamma_phi) * qt.Qobj(np.diag(betas(p))))
    return jumps


# Functions for the cluster.
# Direct solver.
def steady_state(Ej, Ec, ng, g, wr, flux, d,
                 kappa, gamma, gamma_phi, N, J, xi, wd,
                 method='direct'):
    values = locals()
    p = SimpleNamespace()
    p.__dict__.update(values)
    rho_ss = qt.steadystate(jaynes_cummings(p), decoherence(p),
                            method=method)
    res = np.array([qt.expect(i, rho_ss) for i in op_list(p)])
    res[-1] = np.abs(res[-1])
    return tuple(np.real(res))


# Montecarlo solver.
def steady_state_mc(Ej, Ec, ng, g, wr, kappa, gamma, gamma_phi, N, J, xi, wd,
                    ntraj, ti, tf, tsteps, cpus=None):
    values = locals()
    p = SimpleNamespace()
    p.__dict__.update(values)
    times = np.linspace(p.ti, p.tf, p.tsteps)
    ham = jaynes_cummings(p)
    psi0 = jaynes_cummings(p, driven=False).groundstate()[1]

    opts = qt.Odeoptions()
    if cpus:
        opts.num_cpus = cpus

    actualstdout = sys.stdout
    sys.stdout = cStringIO.StringIO()  # avoids prints to stdout
    t_evol = qt.mcsolve(ham, psi0, times, decoherence(p), op_list(p), p.ntraj,
                        options=opts)
    sys.stdout = actualstdout  # reactivates stdout

    result = np.array([i[-1] for i in t_evol.expect])
    result[-1] = np.abs(result[-1])
    return tuple(np.real(result))


if __name__ == "__main__":
    print("Start...")
    print(steady_state(Ej=57.3, Ec=0.32, ng=0., g=0.34, wr=11.06,
                       kappa=0.34, gamma=0.1, gamma_phi=None,
                       N=5, J=2, xi=0.01, wd=11.78))
    print ("...Stop")
