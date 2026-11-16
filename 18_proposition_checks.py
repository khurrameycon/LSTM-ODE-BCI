"""Numerical checks for Proposition 1 (decision invariance) and Proposition 2 (identifiability)."""
import json
import numpy as np

R = 'outputs/results/'
k = json.load(open(R + 'ode_results.json'))['parameters']


def K_of(kk):
    return np.array([
        [-(kk['k_ap'] + kk['k_af']), kk['k_pa'], kk['k_fa']],
        [kk['k_ap'], -(kk['k_pa'] + kk['k_pf']), kk['k_fp']],
        [kk['k_af'], kk['k_pf'], -(kk['k_fa'] + kk['k_fp'])]])


def expm_apply(K, t, s0):
    w, V = np.linalg.eig(K)
    return np.real(V @ np.diag(np.exp(w * t)) @ np.linalg.solve(V, s0))


def g(p, alpha, H=20.0):
    po = 1 - p
    P0 = 0.5 * min(po, p)
    s0 = np.array([(1 - P0) * po, P0, (1 - P0) * p])
    kk = dict(k)
    kk['k_af'] *= 1 + alpha * p
    kk['k_pf'] *= 1 + alpha * p
    kk['k_fa'] *= 1 + alpha * po
    kk['k_pa'] *= 1 + alpha * po
    return expm_apply(K_of(kk), H, s0)


grid = np.linspace(0, 1, 10001)
for alpha in [0, 0.25, 0.5, 0.75, 1.0]:
    F = np.array([g(p, alpha)[2] for p in grid])
    mono = np.all(np.diff(F) > 0)
    # threshold by bisection
    lo, hi = 0.4, 0.6
    for _ in range(60):
        mid = (lo + hi) / 2
        if g(mid, alpha)[2] > 0.5:
            hi = mid
        else:
            lo = mid
    Pres = g(hi, alpha)[1]
    print(f'alpha {alpha}: strictly increasing {mono}, c_alpha {hi:.5f}, Passive at H {Pres:.5f}, min dF {np.diff(F).min():.2e}')

d = json.load(open(R + 'all_model_results.json'))
b = np.array(d['LSTM-ODE']['y_prob'])
print('windows with p in (0.5, 0.5009]:', int(((b > 0.5) & (b <= 0.5009)).sum()))

# Proposition 2: loss sensitivity to Passive outflows vs observed outflows, as eps -> 0
ts = np.linspace(0, 30, 20)


def loss(kk):
    L = 0
    for s0, j in [(np.array([1., 0, 0]), 0), (np.array([0., 0, 1]), 2)]:
        K = K_of(kk)
        L += np.mean([(expm_apply(K, t, s0)[j] - 1) ** 2 for t in ts])
    return L / 2


for eps in [0.01, 0.003, 0.001, 0.0003]:
    kk = dict(k_ap=eps, k_af=eps, k_fa=eps, k_fp=eps, k_pa=k['k_pa'], k_pf=k['k_pf'])
    def dL(name, h=1e-6):
        a = dict(kk); a[name] += h
        b_ = dict(kk); b_[name] -= h
        return (loss(a) - loss(b_)) / (2 * h)
    print(f'eps {eps}: dL/dk_af {dL("k_af"):.3e}  dL/dk_pa {dL("k_pa"):.3e}  ratio {dL("k_pa")/dL("k_af"):.3e}')
