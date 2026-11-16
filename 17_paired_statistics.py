import json, itertools, math
import numpy as np

R = 'outputs/results/'
s = json.load(open(R + 'subject_tests_vs_lstm.json'))
T7 = 2.364624251592785  # t(0.975, df=7)
rng = np.random.default_rng(42)
out = {}
for name, c in s['comparisons'].items():
    d = np.array([x['difference'] for x in c['subject_accuracies']])  # reference - other
    n = len(d)
    m, sd = d.mean(), d.std(ddof=1)
    tci = (m - T7 * sd / math.sqrt(n), m + T7 * sd / math.sqrt(n))
    # exact paired permutation (sign-flip) test, two-sided, on mean difference
    obs = abs(m)
    cnt = 0
    for signs in itertools.product([1, -1], repeat=n):
        if abs((d * np.array(signs)).mean()) >= obs - 1e-12:
            cnt += 1
    p_perm = cnt / 2 ** n
    # subject bootstrap CI of mean difference
    bs = np.array([d[rng.integers(0, n, n)].mean() for _ in range(10000)])
    bci = (np.percentile(bs, 2.5), np.percentile(bs, 97.5))
    out[name] = dict(mean=m, sd=sd, t_ci=tci, boot_ci=bci, p_perm=p_perm,
                     p_sign=c['p_value'], wins=c['subjects_a_better'],
                     losses=c['subjects_b_better'], dz=c['cohens_dz'])
    print(f"{name:35s} mean {m:+.3f} sd {sd:.3f} tCI [{tci[0]:+.3f},{tci[1]:+.3f}] "
          f"bootCI [{bci[0]:+.3f},{bci[1]:+.3f}] p_perm {p_perm:.3f} p_sign {c['p_value']:.3f} "
          f"W/L {c['subjects_a_better']}/{c['subjects_b_better']} dz {c['cohens_dz']:+.2f}")
json.dump(out, open(R + 'paired_statistics.json', 'w'), indent=1, default=float)
# MDD with sd of differences
sds = [v['sd'] for v in out.values()]
print('sd range', min(sds), max(sds), 'median', np.median(sds))
for sd in [min(sds), np.median(sds), max(sds)]:
    print('MDD z (alpha .05, power .8)', (1.959964 + 0.841621) * sd / math.sqrt(8))
