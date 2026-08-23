"""Demonstrate the EOF tip/tilt predictor on (1) a single sinusoidal vibration
and (2) a sum of sinusoidal vibrations, in the spirit of Guyon & Males (2017) §3."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eof_predict import (build_training_matrices, eof_filter, apply_filter,
                         make_vibrations, rms)

# ---- common AO-loop parameters (cf. paper §3) ----------------------------- #
FS      = 1000.0     # 1 kHz sampling
LAG     = 3          # 3-step (3 ms) loop delay
N_ORDER = 200        # filter order: 0.2 s look-back (~3 periods of 15 Hz)
NOISE   = 0.16       # per-axis measurement noise [mas]  (mH ~ 9 star in paper)
TRAIN_S = 30.0       # training-set length [s]
EVAL_S  = 4.0        # evaluation length [s]


def run_case(name, freqs, amps, axes, phases, k_modes, seed=1):
    # one continuous stream; train on the first TRAIN_S, evaluate on the rest
    truth, meas = make_vibrations(freqs, amps, axes, phases, FS,
                                  TRAIN_S + EVAL_S, NOISE, seed=seed)
    n_tr = int(TRAIN_S * FS)

    D, P = build_training_matrices(meas[:n_tr], truth[:n_tr], N_ORDER, LAG)
    F, _ = eof_filter(D, P, k_modes=k_modes)

    # full (untruncated) spectrum, for an honest panel (d)
    _, sv_full, _ = np.linalg.svd(D)

    out = apply_filter(meas[n_tr - N_ORDER - LAG:], truth[n_tr - N_ORDER - LAG:],
                       F, N_ORDER, LAG)
    pred, true, latest = out["pred"], out["true"], out["latest"]

    # residuals at the prediction-target time
    r_none = true                          # apply nothing
    r_last = true - latest                 # best non-predictive (lag + noise)
    r_pred = true - pred                   # EOF predictive

    print(f"\n=== {name} ===  (D: {D.shape[0]}x{D.shape[1]}, kept modes: {k_modes})")
    print(f"  no correction         RMS = {rms(r_none):7.3f} mas/axis")
    print(f"  last measurement      RMS = {rms(r_last)/np.sqrt(2):7.3f} mas/axis "
          f"(lag + {NOISE} mas noise)")
    print(f"  EOF predictive        RMS = {rms(r_pred)/np.sqrt(2):7.3f} mas/axis")
    return truth, meas, out, (r_none, r_last, r_pred), sv_full


def make_figure(name, fname, out, resid, sv_full, k_modes, show_modes, annot):
    pred, true, latest, tidx = out["pred"], out["true"], out["latest"], out["t_idx"]
    r_none, r_last, r_pred = resid
    t = tidx / FS

    fig = plt.figure(figsize=(13, 9))
    fig.suptitle(name, fontsize=13, y=0.98)

    # (a) X time series: true / last-available-measurement / predicted
    ax = fig.add_subplot(2, 2, 1)
    w = (t >= t[0]) & (t <= t[0] + 0.12)            # 120 ms window
    ax.plot(t[w], true[w, 0], "r.-", ms=4, lw=0.8, label="true X (t+lag)")
    ax.plot(t[w], latest[w, 0], "g.", ms=4, alpha=0.7,
            label=f"measurement (lag {LAG} ms)")
    ax.plot(t[w], pred[w, 0], "bo", ms=4, mfc="none", label="EOF prediction")
    ax.set_xlabel("time [s]"); ax.set_ylabel("X tip [mas]")
    ax.legend(fontsize=8, loc="upper right"); ax.set_title("(a) X-axis tracking")

    # (b) 2D track
    ax = fig.add_subplot(2, 2, 2)
    ax.plot(latest[:, 0], latest[:, 1], "g.", ms=2, alpha=0.4, label="measured")
    ax.plot(true[:, 0], true[:, 1], "r-", lw=0.6, alpha=0.7, label="true")
    ax.plot(pred[:, 0], pred[:, 1], "b-", lw=0.6, alpha=0.7, label="predicted")
    ax.set_xlabel("X [mas]"); ax.set_ylabel("Y [mas]"); ax.set_aspect("equal")
    ax.legend(fontsize=8); ax.set_title("(b) 2-D tip/tilt track")

    # (c) residual scatter
    ax = fig.add_subplot(2, 2, 3)
    ax.plot(r_none[:, 0], r_none[:, 1], "r.", ms=2, alpha=0.3,
            label=f"none ({rms(r_none)/np.sqrt(2):.2f})")
    ax.plot(r_last[:, 0], r_last[:, 1], "g.", ms=2, alpha=0.3,
            label=f"last meas ({rms(r_last)/np.sqrt(2):.2f})")
    ax.plot(r_pred[:, 0], r_pred[:, 1], "b.", ms=3,
            label=f"predictive ({rms(r_pred)/np.sqrt(2):.2f})")
    lim = 1.1 * np.max(np.abs(r_none))
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_xlabel("X residual [mas]"); ax.set_ylabel("Y residual [mas]")
    ax.legend(fontsize=8, title="RMS/axis [mas]", title_fontsize=8)
    ax.set_title("(c) residual error")

    # (d) EOF singular-value spectrum -- full spectrum, zoomed to the modes
    #     that matter, with the signal/noise structure called out.
    ax = fig.add_subplot(2, 2, 4)
    s = sv_full[:show_modes] / sv_full[0]
    idx = np.arange(s.size)
    ax.semilogy(idx, s, "k.-", ms=4, lw=0.6)
    if k_modes <= show_modes:
        ax.axvline(k_modes, color="b", ls="--", lw=1.0,
                   label=f"modes kept = {k_modes}")
        ax.legend(fontsize=8, loc="upper right")
    ax.annotate(annot["text"], xy=annot["xy"], xytext=annot["xytext"],
                fontsize=9, ha="center",
                arrowprops=dict(arrowstyle="->", color="0.3", lw=0.8))
    ax.set_xlim(-1, show_modes)
    ax.set_xlabel("EOF index"); ax.set_ylabel("normalised singular value")
    ax.set_title("(d) EOF spectrum of data matrix D")
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(fname, dpi=170)
    plt.close(fig)
    print(f"  -> wrote {fname}")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # ---- Demo 1: a single 2-D sinusoidal vibration (one line in X-Y) ------- #
    truth, meas, out, resid, sv_full = run_case(
        "Demo 1 - single 37 Hz tip/tilt vibration",
        freqs=[37.0], amps=[3.0], axes=[np.deg2rad(35.0)], phases=[0.4],
        k_modes=20)
    make_figure(
        "EOF prediction - single 37 Hz vibration", "fig1_single_sine.png",
        out, resid, sv_full, k_modes=20, show_modes=24,
        annot=dict(text="2 dominant EOFs\n(sin + cos of one tone)",
                   xy=(1, 0.9), xytext=(9, 0.35)))

    # ---- Demo 2: sum of 20 vibrations, 15-92 Hz (paper-like) --------------- #
    rng = np.random.default_rng(7)
    K = 20
    freqs  = rng.uniform(15, 92, K)
    amps   = rng.uniform(0.3, 1.2, K)
    axes   = rng.uniform(0, np.pi, K)
    phases = rng.uniform(0, 2 * np.pi, K)
    truth, meas, out, resid, sv_full = run_case(
        "Demo 2 - sum of 20 vibrations (15-92 Hz)",
        freqs, amps, axes, phases, k_modes=90)
    make_figure(
        "EOF prediction - sum of 20 vibrations (15-92 Hz)", "fig2_sum_sines.png",
        out, resid, sv_full, k_modes=90, show_modes=100,
        annot=dict(text="knee ~ 40\n(2 modes per vibration)",
                   xy=(40, 0.03), xytext=(62, 0.18)))