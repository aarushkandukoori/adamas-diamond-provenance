"""
ADAMAS evaluation harness (final).

Stage A -- alignment-free invariant descriptor -> learned projection -> bits.
           Screening / indexing layer plus the natural-vs-CVD gate.
Stage B -- canonical-frame hypothesis search + ICP registration -> lattice
           quantisation -> fuzzy vault over the inclusion point set.

Physical note: re-polishing removes material from the surface, so near-surface
inclusions disappear, but the surviving inclusions stay rigid with respect to
one another. The reference frame moves; the constellation does not.
"""

import argparse, json, math, itertools, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _parse_args():
    p = argparse.ArgumentParser(description="ADAMAS evaluation harness")
    root = Path(__file__).resolve().parents[1]
    p.add_argument("--outdir", type=Path, default=root / "results",
                   help="directory for results.json (default: <repo>/results)")
    p.add_argument("--figdir", type=Path, default=None,
                   help="directory for figure PDFs (default: <outdir>/figures)")
    return p.parse_args()

_ARGS = _parse_args()
OUTDIR = _ARGS.outdir.resolve()
FIGDIR = (_ARGS.figdir if _ARGS.figdir is not None else OUTDIR / "figures").resolve()
OUTDIR.mkdir(parents=True, exist_ok=True)
FIGDIR.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(20260721)

R_UM, JITTER_UM = 2500.0, 8.0
P_MISS_BASE, SPUR_RATE, SIZE_NOISE = 0.06, 0.6, 0.12
AXES = np.array([1.00, 0.84, 0.63])
N_TRAIN, N_TEST = 500, 500
R_ENROLL, R_PROBE = 7, 5
GRID = 6


# ============================================================ stone and scan
def sample_stone(rng, lab_grown=False, depth=0):
    K = int(np.clip(rng.poisson(18), 6, 45))
    n_pl = int(round(0.4 * K)); n_bk = K - n_pl
    pts = []
    u = rng.normal(size=(n_bk, 3)); u /= np.linalg.norm(u, axis=1, keepdims=True)
    pts.append(u * (R_UM * rng.random(n_bk) ** (1 / 3))[:, None] * AXES)
    if n_pl > 0:
        for j in range(int(rng.integers(1, 3))):
            m = n_pl // 2 + (n_pl % 2 if j == 0 else 0)
            if m <= 0: continue
            nrm = rng.normal(size=3); nrm /= np.linalg.norm(nrm)
            b1 = np.cross(nrm, rng.normal(size=3)); b1 /= np.linalg.norm(b1)
            b2 = np.cross(nrm, b1)
            off = rng.normal(0, .35 * R_UM)
            uv = rng.normal(0, .45 * R_UM, size=(m, 2))
            th = rng.normal(0, .06 * R_UM, size=m)
            pts.append(uv[:, 0:1] * b1 + uv[:, 1:2] * b2 + (off + th)[:, None] * nrm)
    X = np.vstack(pts); X = X[np.linalg.norm(X / AXES, axis=1) < R_UM]
    if len(X) < 8:
        return sample_stone(rng, lab_grown, depth + 1) if depth < 6 else None
    s = np.exp(rng.normal(np.log(30.), .8, size=len(X)))
    pl = rng.normal(0, 1., size=12)
    if lab_grown: pl[0] += 6.; pl[1] += 4.; pl[2] -= 2.
    return dict(X=X, sizes=s, pl=pl, lab_grown=lab_grown)


def scan(st, rng, recut=False, pose=True):
    X, s = st["X"].copy(), st["sizes"].copy()
    if recut:                                  # surface material removed only
        k = np.linalg.norm(X / AXES, axis=1) < .88 * R_UM
        if k.sum() >= 6: X, s = X[k], s[k]
    pm = np.clip(P_MISS_BASE * (30. / np.clip(s, 5, None)) ** .5, 0, .6)
    d = rng.random(len(X)) > pm
    if d.sum() < 5: d[:] = True
    X, s = X[d], s[d]
    X = X + rng.normal(0, JITTER_UM, size=X.shape)
    s = s * np.exp(rng.normal(0, SIZE_NOISE, size=len(s)))
    ns = rng.poisson(SPUR_RATE)
    if ns:
        u = rng.normal(size=(ns, 3)); u /= np.linalg.norm(u, axis=1, keepdims=True)
        X = np.vstack([X, u * (R_UM * rng.random(ns) ** (1 / 3))[:, None] * AXES])
        s = np.concatenate([s, np.exp(rng.normal(np.log(12.), .5, ns))])
    if pose:
        Q, Rq = np.linalg.qr(rng.normal(size=(3, 3)))
        Q = Q * np.sign(np.diag(Rq))
        if np.linalg.det(Q) < 0: Q[:, 0] *= -1
        X = X @ Q.T + rng.normal(0, 500, size=3)
    pl = st["pl"] + rng.normal(0, .35, size=12) + rng.normal(0, .20)
    return dict(X=X, sizes=s, pl=pl)


# ================================================== Stage A: invariant descriptor
D2_C, RAD_C = np.linspace(.05, 2.3, 28), np.linspace(.05, 1.7, 14)
_g = (np.arange(GRID) + .5) / GRID * 2 - 1
_VOX = np.stack(np.meshgrid(_g, _g, _g, indexing="ij"), -1).reshape(-1, 3)
_VS = .9 / GRID


def canon(X, w):
    c = (w[:, None] * X).sum(0); Xc = X - c
    C = (w[:, None] * Xc).T @ Xc
    ev, V = np.linalg.eigh(C); o = np.argsort(ev)[::-1]; ev, V = ev[o], V[:, o]
    P = Xc @ V
    for k in range(3):
        if (w * P[:, k] ** 3).sum() < 0: V[:, k] *= -1; P[:, k] *= -1
    if np.linalg.det(V) < 0: V[:, 2] *= -1; P[:, 2] *= -1
    return P, ev / (ev.sum() + 1e-12)


def descriptor(sc_):
    X, w = sc_["X"], sc_["sizes"] / sc_["sizes"].sum()
    K = len(X); f = []
    P, evn = canon(X, w)
    sc0 = math.sqrt((w * (P ** 2).sum(1)).sum()) + 1e-9
    d2 = ((P[:, None, :] / sc0 - _VOX[None]) ** 2).sum(-1)
    v = (w[:, None] * np.exp(-.5 * d2 / _VS ** 2)).sum(0)
    f += [v / (v.sum() + 1e-12), evn]
    D = np.linalg.norm(X[:, None] - X[None], axis=-1)
    iu = np.triu_indices(K, 1); dij, wij = D[iu], np.outer(w, w)[iu]
    med = max(float(np.median(dij)), 1e-6); dn = dij / med
    for bw in (.10, .22):
        h = (np.exp(-.5 * ((dn[:, None] - D2_C[None]) / bw) ** 2) * wij[:, None]).sum(0)
        f.append(h / (h.sum() + 1e-12))
    c = (w[:, None] * X).sum(0); rr = np.linalg.norm(X - c, axis=1) / med
    hr = (np.exp(-.5 * ((rr[:, None] - RAD_C[None]) / .14) ** 2) * w[:, None]).sum(0)
    f.append(hr / (hr.sum() + 1e-12))
    for bw in (.35, .80):
        G = np.outer(w, w) * np.exp(-.5 * (D / (bw * med)) ** 2)
        e = np.sort(np.linalg.eigvalsh(G))[::-1][:10]
        f.append(np.pad(e, (0, max(0, 10 - len(e)))) / (np.abs(e).sum() + 1e-12))
    f.append(np.array([np.log(K), np.log(med), np.log(sc_["sizes"].sum()),
                       np.log(np.median(sc_["sizes"])), float(np.std(np.log(sc_["sizes"])))]))
    f.append(sc_["pl"])
    return np.concatenate(f)


def build(n, rng, lab_frac=0.):
    stones, S = [], []
    while len(stones) < n:
        st = sample_stone(rng, lab_grown=(rng.random() < lab_frac))
        if st is None: continue
        stones.append(st)
        S.append([descriptor(scan(st, rng)) for _ in range(R_ENROLL + R_PROBE)])
    return stones, np.array(S)


t0 = time.time()
print("generating population ...")
train_stones, TR = build(N_TRAIN, RNG)
test_stones, TE = build(N_TEST, RNG)
d_raw = TR.shape[2]
mu = TR.reshape(-1, d_raw).mean(0); sd = TR.reshape(-1, d_raw).std(0) + 1e-9
Z = (TR - mu) / sd
print(f"  d_raw = {d_raw}   ({time.time()-t0:.0f}s)")


def inv_sqrt(M, eps):
    w, V = np.linalg.eigh(M); return V @ np.diag(np.clip(w, eps, None) ** -.5) @ V.T


Ct = np.cov(Z.reshape(-1, d_raw).T)
ev, V = np.linalg.eigh(Ct); o = np.argsort(ev)[::-1]; ev, V = ev[o], V[:, o]
d_pca = int(min((ev > 1e-3 * ev[0]).sum(), 300))
Wp = (V[:, :d_pca] / np.sqrt(ev[:d_pca])).T
Yp = Z @ Wp.T
Sw = np.cov((Yp - Yp.mean(1, keepdims=True)).reshape(-1, d_pca).T)
A = inv_sqrt(Sw, 1e-3 * np.trace(Sw) / d_pca)
Mb = (Yp.mean(1) - Yp.mean(1).mean(0)) @ A.T
evb, Vb = np.linalg.eigh(np.cov(Mb.T)); o = np.argsort(evb)[::-1]
W_LRN, W_PCA = (Vb[:, o].T @ A) @ Wp, Wp
print(f"  d_pca = {d_pca}")


def snr_of(W):
    Y = Z @ W.T
    return Y.mean(1).std(0) / ((Y - Y.mean(1, keepdims=True)).reshape(-1, W.shape[0]).std(0) + 1e-12)


S_L, S_P = snr_of(W_LRN), snr_of(W_PCA)
gray = lambda v: v ^ (v >> 1)


def codebook(W, snr, thr=(1.0, 2.0, 4.0)):
    k = sum((snr > t).astype(int) for t in thr)
    Y = (Z @ W.T).reshape(-1, W.shape[0])
    return k, [None if kj == 0 else np.quantile(Y[:, j], np.linspace(0, 1, 2 ** kj + 1)[1:-1])
               for j, kj in enumerate(k)]


def encode(S, W, k, cuts):
    Y = ((S - mu) / sd) @ W.T; out = []
    for j, kj in enumerate(k):
        if kj == 0: continue
        g = gray(np.searchsorted(cuts[j], Y[..., j]).astype(np.int64))
        out += [((g >> b) & 1).astype(np.uint8) for b in range(kj)]
    return np.stack(out, -1)


def stageA(W, snr, tag, n_list):
    k, cuts = codebook(W, snr)
    Btr, Bte = encode(TR, W, k, cuts), encode(TE, W, k, cuts)
    ntot = Btr.shape[-1]
    enr = Btr[:, :R_ENROLL]
    flip = np.abs(enr - np.round(enr.mean(1))[:, None]).mean((0, 1))
    bias = np.abs(Btr.reshape(-1, ntot).mean(0) - .5)
    rank = np.argsort(flip + .6 * bias)
    res = {}
    for n in sorted(set(min(x, ntot) for x in n_list)):
        idx = rank[:n]
        T = (Bte[:, :R_ENROLL][:, :, idx].mean(1) > .5).astype(np.uint8)
        Pr = Bte[:, R_ENROLL:][:, :, idx]
        intra = (T[:, None] != Pr).mean(-1).ravel()
        inter = np.concatenate([(T[RNG.choice(np.delete(np.arange(len(T)), i), 60, False)]
                                 != Pr[i, 0][None]).mean(-1) for i in range(len(T))])
        rc = np.array([descriptor(scan(test_stones[i], RNG, recut=True)) for i in range(len(T))])
        recut = (T != encode(rc[:, None, :], W, k, cuts)[:, 0][:, idx]).mean(-1)
        p1 = T.mean(0)
        hb = float(np.sum(-np.log2(np.clip(np.maximum(p1, 1 - p1), 1e-9, 1))))
        Cc = np.nan_to_num(np.corrcoef(T.astype(float).T), nan=0.)
        e = np.clip(np.linalg.eigvalsh(Cc), 1e-9, None); pe = e / e.sum()
        eff = float(np.exp(-(pe * np.log(pe)).sum()))
        res[n] = dict(n=n, ntot=int(ntot), intra_mean=float(intra.mean()),
                      inter_mean=float(inter.mean()), inter_std=float(inter.std()),
                      recut_mean=float(recut.mean()), h_corr=hb * eff / n,
                      rho=hb * eff / n / n, intra=intra, inter=inter, recut=recut)
        print(f"  [{tag} n={n:4d}/{ntot}] BER={intra.mean():.4f} inter={inter.mean():.4f}"
              f"±{inter.std():.4f} recut={recut.mean():.4f} H={hb*eff/n:6.1f}b rho={hb*eff/n/n:.3f}")
    return res, (k, cuts, rank)


NS = [64, 96, 128, 192]
print("\nStage A -- unsupervised (PCA only):"); resP, _ = stageA(W_PCA, S_P, "PCA", NS)
print("Stage A -- learned (PCA->WCCN->LDA):"); resL, (kL, cL, rkL) = stageA(W_LRN, S_L, "LRN", NS)
NA = max(resL.keys()); rl, rp = resL[NA], resP[max(resP.keys())]


def det(r):
    n = r["n"]; ts = np.arange(0, int(.5 * n) + 1)
    frr = np.array([(r["intra"] * n > t).mean() for t in ts])
    far = np.array([(r["inter"] * n <= t).mean() for t in ts])
    j = int(np.argmin(np.abs(frr - far)))
    return ts, frr, far, int(ts[j]), float(.5 * (frr[j] + far[j]))


tsL, frrL, farL, teL, eerL = det(rl); tsP, frrP, farP, teP, eerP = det(rp)
ok = np.where(frrL <= 1e-2)[0]
T_A = int(tsL[ok[0]]) if len(ok) else int(tsL[-1])
FAR_A = float(farL[list(tsL).index(T_A)])
print(f"\nStage A: EER learned {eerL:.3e} vs PCA {eerP:.3e}; "
      f"screening t={T_A}/{NA} -> FAR {FAR_A:.3e} at FRR 1e-2")

lab_stones, LAB = build(200, RNG, lab_frac=1.)
idxA = rkL[:NA]
Bte = encode(TE, W_LRN, kL, cL)
TMPL_A = (Bte[:, :R_ENROLL][:, :, idxA].mean(1) > .5).astype(np.uint8)
Blab = encode(LAB, W_LRN, kL, cL)[:, 0][:, idxA]
lab_hd = np.array([(TMPL_A != Blab[i][None]).mean(-1).min() for i in range(len(Blab))])
# PL sub-block on its own is the material-class gate
pl_dims = np.arange(d_raw - 12, d_raw)
pl_nat = TR[:, :, pl_dims].reshape(-1, 12)[:, 0]
pl_lab = LAB[:, :, pl_dims].reshape(-1, 12)[:, 0]
thr_pl = (pl_nat.mean() + pl_lab.mean()) / 2
pl_acc = float(((pl_nat < thr_pl).mean() + (pl_lab > thr_pl).mean()) / 2)
print(f"Stage A lab-grown: nearest-template HD {lab_hd.mean():.4f}±{lab_hd.std():.4f} "
      f"(min {lab_hd.min():.4f}); admitted at t={T_A}: {(lab_hd*NA<=T_A).sum()}/{len(lab_hd)}; "
      f"PL-gate accuracy {pl_acc:.4f}")


# ============================================ Stage B: registration + vault
def proper_signed_perms():
    out = []
    for p in itertools.permutations(range(3)):
        for s in itertools.product((1, -1), repeat=3):
            M = np.zeros((3, 3))
            for i, pi in enumerate(p): M[i, pi] = s[i]
            if np.linalg.det(M) > 0: out.append(M)
    return np.array(out)


PERMS = proper_signed_perms()          # 24 proper rotations


def _icp(Y, T, gates):
    for g in gates:
        D = np.linalg.norm(Y[:, None] - T[None], axis=-1)
        j = D.argmin(1); m = D.min(1) < g
        if m.sum() < 4: break
        Aa, Bb = Y[m], T[j[m]]
        ca, cb = Aa.mean(0), Bb.mean(0)
        U, S_, Vh = np.linalg.svd((Aa - ca).T @ (Bb - cb))
        Rr = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vh))]) @ Vh
        sc = S_.sum() / (((Aa - ca) ** 2).sum() + 1e-9)
        sc = float(np.clip(sc, .9, 1.1))
        Y = sc * (Y - ca) @ Rr + cb
    return Y


def align(Xp, wp, Xt, wt):
    """24-hypothesis canonical-frame search + annealed ICP; returns best fit."""
    Pp, _ = canon(Xp, wp); Pt, _ = canon(Xt, wt)
    best, bestin, bestres = None, -1, np.inf
    for M in PERMS:
        Y = _icp(Pp @ M.T, Pt, (600., 400., 250., 160., 120.))
        D = np.linalg.norm(Y[:, None] - Pt[None], axis=-1)
        nin = int((D.min(1) < 120.).sum())
        res = float(np.mean(np.sort(D.min(1))[:max(4, nin)]))
        if (nin, -res) > (bestin, -bestres):
            best, bestin, bestres = Y, nin, res
    return best, Pt, bestres


def enroll_template(st, rng, R=R_ENROLL):
    scans = [scan(st, rng) for _ in range(R)]
    base = scans[0]
    T, _ = canon(base["X"], base["sizes"] / base["sizes"].sum())
    acc, resid = [[p] for p in T], []
    for sc_ in scans[1:]:
        Y, Tt, r = align(sc_["X"], sc_["sizes"] / sc_["sizes"].sum(),
                         base["X"], base["sizes"] / base["sizes"].sum())
        resid.append(r)
        D = np.linalg.norm(Y[:, None] - T[None], axis=-1)
        for a in range(len(Y)):
            b = int(D[a].argmin())
            if D[a, b] < 120.: acc[b].append(Y[a])
    cnt = np.array([len(v) for v in acc])
    keep = cnt >= max(3, int(.55 * R))
    return np.array([np.mean(acc[i], 0) for i in np.where(keep)[0]]), base, float(np.mean(resid))


def probe_match(base, T, src, rng, recut=False):
    sc_ = scan(src, rng, recut=recut)
    Y, _, r = align(sc_["X"], sc_["sizes"] / sc_["sizes"].sum(),
                    base["X"], base["sizes"] / base["sizes"].sum())
    D = np.linalg.norm(Y[:, None] - T[None], axis=-1)
    return int((D.min(0) < 120.).sum()), len(Y), r


print("\nStage B -- registration and point recovery ...")
NB = 200
G_list, rec, rec_rc, imp, resid_all, nprobe = [], [], [], [], [], []
for i in range(NB):
    T, base, r0 = enroll_template(test_stones[i], RNG)
    if len(T) < 6: continue
    G_list.append(len(T)); resid_all.append(r0)
    h = [probe_match(base, T, test_stones[i], RNG) for _ in range(R_PROBE)]
    rec.append(np.mean([x[0] for x in h]) / len(T)); nprobe.append(np.mean([x[1] for x in h]))
    rec_rc.append(probe_match(base, T, test_stones[i], RNG, recut=True)[0] / len(T))
    imp.append(probe_match(base, T, test_stones[(i + 7) % NB], RNG)[0] / len(T))
    if i == 0: TEMPL0, BASE0 = T, base
G_arr = np.array(G_list); rec = np.array(rec); rec_rc = np.array(rec_rc); imp = np.array(imp)
print(f"  |G| = {G_arr.mean():.1f} ± {G_arr.std():.1f}   "
      f"registration residual = {np.mean(resid_all):.1f} µm")
print(f"  genuine recovery  = {rec.mean():.4f} ± {rec.std():.4f}")
print(f"  after re-polish   = {rec_rc.mean():.4f} ± {rec_rc.std():.4f}")
print(f"  impostor recovery = {imp.mean():.4f} ± {imp.std():.4f}")
print(f"  probe points/scan = {np.mean(nprobe):.1f}   ({time.time()-t0:.0f}s)")

LATTICE_UM = max(80.0, 4 * np.mean(resid_all))
N_CELLS = (2 * R_UM / LATTICE_UM) ** 3
print(f"  lattice {LATTICE_UM:.0f} µm -> {N_CELLS:.0f} cells")


def logC(n, k):
    return -np.inf if (k < 0 or k > n) else (math.lgamma(n + 1) - math.lgamma(k + 1)
                                             - math.lgamma(n - k + 1))


def sec_bits(G, C, D):
    a, b = logC(G + C, D + 1), logC(G, D + 1)
    return np.inf if b == -np.inf else (a - b) / math.log(2)


def pmf_binom(n, p):
    return np.array([math.exp(logC(n, i) + i * math.log(max(p, 1e-15))
                              + (n - i) * math.log(max(1 - p, 1e-15))) for i in range(n + 1)])


def p_unlock(G, r, D, P, C, cells):
    """RS decoding succeeds iff genuine matches g >= D+1+s, s = chaff collisions."""
    pg = pmf_binom(G, r)
    ps = pmf_binom(int(round(P)), min(C / cells, .95))
    tot = 0.
    for g, a in enumerate(pg):
        if a < 1e-18: continue
        for s, b in enumerate(ps):
            if b < 1e-18: continue
            if g >= D + 1 + s: tot += a * b
    return float(tot)


Gb, rb, Pb = int(round(G_arr.mean())), float(rec.mean()), float(np.mean(nprobe))
rows = []
for C in (2000, 10000, 50000, 200000):
    for D in (6, 8, 10, 12):
        if C > N_CELLS * .9: continue
        rows.append(dict(C=C, D=D, sec=sec_bits(Gb, C, D),
                         unlock=p_unlock(Gb, rb, D, Pb, C, N_CELLS),
                         unlock_recut=p_unlock(Gb, float(rec_rc.mean()), D, Pb, C, N_CELLS),
                         false=p_unlock(Gb, float(imp.mean()), D, Pb, C, N_CELLS)))
print("\n   chaff    deg   security(b)   P(unlock)   P(unlock|recut)   P(false unlock)")
for r_ in rows:
    print(f"  {r_['C']:7d} {r_['D']:5d} {r_['sec']:12.1f} {r_['unlock']:11.4f} "
          f"{r_['unlock_recut']:16.4f} {r_['false']:16.2e}")
cands = [r_ for r_ in rows if r_["sec"] >= 128 and r_["unlock"] >= .95]
CHOICE = min(cands, key=lambda r_: r_["C"]) if cands else max(rows, key=lambda r_: r_["sec"] if r_["unlock"] >= .95 else -1)
print(f"\n  selected operating point: {CHOICE}")


def vault_attack(frac, m=120):
    out = []
    for i in range(m):
        st = test_stones[i]
        keep = RNG.random(len(st["X"])) < frac
        if keep.sum() < 5: keep[:5] = True
        nm = int((~keep).sum())
        u = RNG.normal(size=(nm, 3)); u /= (np.linalg.norm(u, axis=1, keepdims=True) + 1e-9)
        fake = dict(X=np.vstack([st["X"][keep], u * (R_UM * RNG.random(nm) ** (1 / 3))[:, None] * AXES]),
                    sizes=np.concatenate([st["sizes"][keep], np.exp(RNG.normal(np.log(30), .8, nm))]),
                    pl=st["pl"])
        T, base, _ = enroll_template(st, RNG, R=3)
        if len(T) < 6: continue
        out.append(probe_match(base, T, fake, RNG)[0] / len(T))
    return np.array(out)


vatk = {f: vault_attack(f) for f in (.25, .50, .75, .90)}
print(f"\n  partial-knowledge attack (D={CHOICE['D']}, C={CHOICE['C']}):")
for f, v in vatk.items():
    print(f"    {int(f*100):3d}% of map known -> recovery {v.mean():.3f}±{v.std():.3f}, "
          f"P(unlock) = {p_unlock(Gb, float(v.mean()), CHOICE['D'], Pb, CHOICE['C'], N_CELLS):.2e}")


# =============================================== Proposition 1 feasibility
def h2(p):
    p = np.clip(p, 1e-12, 1 - 1e-12); return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


ELL, EPSB, GAM = 128, 128, 1.2
feas = []
for p in (.05, .10, .15, .20):
    row = [p, float(h2(p))]
    for rho in (.5, .7, .9):
        den = rho - GAM * h2(p)
        row.append(int(np.ceil((ELL + EPSB) / den)) if den > 0 else None)
    feas.append(row)
print("\nProposition 1 requirements:")
for row in feas:
    print(f"  BER={row[0]:.2f} h2={row[1]:.3f} n_req(rho=.5,.7,.9)={row[2:]}")
bestA = max(((n, resL[n]["h_corr"] - GAM * resL[n]["n"] * h2(resL[n]["intra_mean"]) - EPSB)
             for n in resL), key=lambda x: x[1])
print(f"\nStage A best extractable key: {bestA[1]:.1f} bits (n={bestA[0]}) "
      f"-> below 128, which is why Stage B exists")

# ==================================================================== figures
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.linewidth": .6,
                     "figure.dpi": 200})
C1, C2, C3, C4 = "#1b3a5c", "#b03a2e", "#6b7d2f", "#7d3c98"

fig, ax = plt.subplots(figsize=(3.3, 2.05))
ax.hist(rl["inter"], bins=70, density=True, color=C1, alpha=.75, label="distinct stones")
ax.hist(rl["intra"], bins=45, density=True, color=C2, alpha=.85, label="same stone, re-scan")
ax.hist(rl["recut"], bins=45, density=True, color=C3, alpha=.55, label="same stone, re-polished")
ax.axvline(T_A / NA, color="k", ls="--", lw=.8)
ax.set_xlabel("fractional Hamming distance"); ax.set_ylabel("density")
ax.legend(frameon=False, fontsize=6.3, loc="upper center"); ax.set_xlim(0, .62)
fig.tight_layout(); fig.savefig(FIGDIR / "fig_hamming.pdf")

fig, ax = plt.subplots(figsize=(3.3, 2.05))
ax.semilogy(tsL / NA, np.clip(frrL, 3e-6, 1), color=C2, lw=1.2, label="FRR, learned")
ax.semilogy(tsL / NA, np.clip(farL, 3e-6, 1), color=C1, lw=1.2, label="FAR, learned")
ax.semilogy(tsP / rp["n"], np.clip(frrP, 3e-6, 1), color=C2, lw=.9, ls=":", label="FRR, PCA only")
ax.semilogy(tsP / rp["n"], np.clip(farP, 3e-6, 1), color=C1, lw=.9, ls=":", label="FAR, PCA only")
ax.axvline(T_A / NA, color="k", ls="--", lw=.8)
ax.set_xlabel(r"screening radius $t/n$"); ax.set_ylabel("rate")
ax.set_xlim(0, .5); ax.set_ylim(3e-6, 1.6); ax.legend(frameon=False, fontsize=6.3, loc="lower right")
fig.tight_layout(); fig.savefig(FIGDIR / "fig_det.pdf")

fig, ax = plt.subplots(figsize=(3.3, 2.05))
bins = np.linspace(0, 1.02, 42)
ax.hist(rec, bins=bins, density=True, color=C2, alpha=.85, label="genuine re-scan")
ax.hist(rec_rc, bins=bins, density=True, color=C3, alpha=.55, label="after re-polish")
ax.hist(imp, bins=bins, density=True, color=C1, alpha=.8, label="impostor stone")
ax.hist(vatk[.90], bins=bins, density=True, color=C4, alpha=.45, label="adversary, 90% of map")
ax.set_xlabel("fraction of enrolled inclusions recovered"); ax.set_ylabel("density")
ax.legend(frameon=False, fontsize=6.0, loc="upper center")
fig.tight_layout(); fig.savefig(FIGDIR / "fig_recovery.pdf")

fig, ax = plt.subplots(figsize=(3.3, 2.05))
for D, col, mk in ((6, C3, "o"), (8, C1, "s"), (10, C2, "^"), (12, C4, "d")):
    xs = [r_["C"] for r_ in rows if r_["D"] == D]; ys = [r_["sec"] for r_ in rows if r_["D"] == D]
    if xs: ax.semilogx(xs, ys, mk + "-", color=col, ms=3.2, lw=1.1, label=rf"$D={D}$")
ax.axhline(128, color="k", ls="--", lw=.8)
ax.set_xlabel("chaff points $C$"); ax.set_ylabel("vault security (bits)")
ax.legend(frameon=False, fontsize=6.3, loc="lower right")
fig.tight_layout(); fig.savefig(FIGDIR / "fig_vault.pdf")

fig, ax = plt.subplots(figsize=(3.3, 2.05))
pp = np.linspace(.01, .30, 400)
for rho, col, ls in ((.5, C3, ":"), (.7, C1, "-"), (.9, C2, "--")):
    den = rho - GAM * h2(pp)
    ax.semilogy(pp, np.where(den > 0, (ELL + EPSB) / np.maximum(den, 1e-9), np.nan),
                ls, color=col, lw=1.2, label=rf"$\rho={rho}$")
ax.scatter([rl["intra_mean"]], [NA], marker="*", s=80, color="k", zorder=5)
ax.annotate("Stage A (sim.)", (rl["intra_mean"], NA), textcoords="offset points",
            xytext=(6, -10), fontsize=6.3)
ax.set_xlabel("bit error rate $p$"); ax.set_ylabel(r"$n$ required for a 128-bit key")
ax.set_ylim(50, 1e5); ax.legend(frameon=False, fontsize=6.3)
fig.tight_layout(); fig.savefig(FIGDIR / "fig_feasible.pdf")

strip = lambda d: {str(k): {kk: vv for kk, vv in v.items() if not isinstance(vv, np.ndarray)}
                   for k, v in d.items()}
json.dump(dict(
    d_raw=int(d_raw), d_pca=int(d_pca), n_train=N_TRAIN, n_test=N_TEST,
    r_enroll=R_ENROLL, r_probe=R_PROBE, n_stageA=int(NA),
    stageA_pca=strip(resP), stageA_learned=strip(resL),
    eer_learned=eerL, eer_pca=eerP, t_screen=int(T_A), far_screen=FAR_A,
    lab_hd_mean=float(lab_hd.mean()), lab_hd_min=float(lab_hd.min()),
    lab_admitted=int((lab_hd * NA <= T_A).sum()), lab_n=int(len(lab_hd)), pl_gate_acc=pl_acc,
    stageA_best_key_bits=float(bestA[1]), stageA_best_n=int(bestA[0]),
    G_mean=float(G_arr.mean()), G_std=float(G_arr.std()),
    resid_um=float(np.mean(resid_all)), lattice_um=float(LATTICE_UM), n_cells=float(N_CELLS),
    probe_pts=float(np.mean(nprobe)),
    rec_mean=float(rec.mean()), rec_std=float(rec.std()),
    rec_recut_mean=float(rec_rc.mean()), rec_recut_std=float(rec_rc.std()),
    imp_mean=float(imp.mean()), imp_std=float(imp.std()),
    vault=rows, vault_choice=CHOICE,
    vault_attack={str(f): dict(mean=float(v.mean()), std=float(v.std()),
                               punlock=p_unlock(Gb, float(v.mean()), CHOICE["D"], Pb,
                                                CHOICE["C"], N_CELLS))
                  for f, v in vatk.items()},
    feasibility=feas, nb=int(len(G_arr)),
), open(OUTDIR / "results.json", "w"), indent=2)
print(f"\nwrote {OUTDIR / 'results.json'} + 5 figures   ({time.time()-t0:.0f}s total)")

# ============================================================================
# 11. Sensitivity of achievable security to the number of mapped inclusions.
#     This is the analysis that converts the security level into an instrument
#     specification (Finding 3 in the paper).
# ============================================================================

def best_security(G, r, Pscale, cells, target_unlock=0.95):
    best, arg = 0.0, None
    for C in (2000, 5000, 10000, 20000, 50000, 100000):
        if C > 0.35 * cells:
            continue
        for D in range(4, G):
            if p_unlock(G, r, D, Pscale, C, cells) >= target_unlock:
                sv = sec_bits(G, C, D)
                if sv > best:
                    best, arg = sv, (C, D)
    return best, arg


print("\nSensitivity of achievable security to template size:")
sens = []
for G in (10, 14, 18, 22, 26, 30, 36, 42):
    sv, arg = best_security(G, rb, Pb * G / max(Gb, 1), N_CELLS)
    sens.append(dict(G=G, sec=float(sv), C=(arg[0] if arg else None), D=(arg[1] if arg else None)))
    print(f"  |G|={G:3d}  max security at 95% unlock: {sv:6.1f} bits  (C,D)={arg}")
G_for_128 = next((d["G"] for d in sens if d["sec"] >= 128), None)
print(f"\n  smallest |G| reaching 128-bit security: {G_for_128}")

fig, ax = plt.subplots(figsize=(3.3, 2.05))
ax.plot([d["G"] for d in sens], [d["sec"] for d in sens], "o-", color=C1, ms=3.4, lw=1.2)
ax.axhline(128, color="k", ls="--", lw=.8)
ax.text(10.5, 133, "128-bit target", fontsize=6.3)
ax.axvline(G_arr.mean(), color=C2, ls=":", lw=1.0)
ax.text(G_arr.mean() + .6, 20, "simulated\nmapper", fontsize=6.3, color=C2)
ax.set_xlabel(r"reliably mapped inclusions $|G|$")
ax.set_ylabel("achievable security (bits)")
fig.tight_layout(); fig.savefig(FIGDIR / "fig_sens.pdf")

_r = json.load(open(OUTDIR / "results.json"))
_r["sensitivity"] = sens
_r["G_for_128"] = G_for_128
json.dump(_r, open(OUTDIR / "results.json", "w"), indent=2)
print(f"wrote {FIGDIR / 'fig_sens.pdf'} and updated {OUTDIR / 'results.json'}")
