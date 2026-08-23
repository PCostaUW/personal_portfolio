"""
Empirical Orthogonal Functions (EOF) predictive filter for tip/tilt sensing.

Implements the optimal-least-squares multivariate auto-regressive predictor of
Guyon & Males (2017), "Adaptive Optics Predictive Control with Empirical
Orthogonal Functions (EOFs)", Section 2.  Control is NOT implemented -- this
is purely the *sensing/prediction* part: given the last n noisy measurements,
estimate the wavefront (here: 2D tip/tilt position) delta-t steps into the future.

Notation maps to the paper:
  m    = number of variables per measurement   (tip/tilt => m = 2: X, Y)
  n    = temporal order of the filter (number of look-back steps)
  l/L  = number of training history vectors
  lag  = delta_t / dt, the loop delay in samples
  h(t) = length n*m "history" vector, eq. (1), most-recent-first
  D    = (n*m) x L data matrix,            eq. (6)
  P~   = m x L matrix of a-posteriori (future) values, eqs. (7),(19)
  F    = m x (n*m) predictive filter,      eqs. (10),(18)
  pred = F h(t),                           eqs. (3),(20)

The filter is obtained from an SVD of D (eq. 11-13).  The columns of the left
singular matrix U are the EOFs (principal components) of the telemetry; keeping
only the dominant ones performs the dimensionality reduction / noise filtering
of eq. (14).  Tikhonov regularisation (eqs. 16-17) is also available.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


# --------------------------------------------------------------------------- #
#  Training-matrix assembly  (eqs. 1, 6, 7)
# --------------------------------------------------------------------------- #
def build_training_matrices(meas, truth, n_order, lag):
    """Assemble the data matrix D and target matrix P~ from a telemetry stream.

    Parameters
    ----------
    meas  : (T, m) noisy measurements used to build the history vectors h(t).
    truth : (T, m) values the filter should reproduce at t+lag.  For the
            least-squares argument of eqs. (4)-(5) this can simply be `meas`
            itself (uncorrelated noise -> same optimal filter); using the
            noise-free signal here just gives a cleaner training target.
    n_order : temporal filter order n.
    lag     : prediction horizon delta_t in samples.

    Returns
    -------
    D : (n*m, L) data matrix, columns = history vectors h, most-recent-first.
    P : (m, L)   a-posteriori targets w~(t+lag).
    """
    T, m = meas.shape
    # windows[i] covers samples meas[i : i+n], windows[i][:, :, j] = meas[i+j]
    w = sliding_window_view(meas, n_order, axis=0)          # (T-n+1, m, n)
    # most-recent-first ordering -> reverse the window axis, put time outermost
    H = w[:, :, ::-1].transpose(0, 2, 1)                    # (T-n+1, n, m)
    H = H.reshape(H.shape[0], n_order * m)                  # (T-n+1, n*m)

    # history vector ending at sample k = i + n - 1 predicts truth[k + lag]
    n_win = T - n_order - lag + 1                           # keep valid targets only
    D = H[:n_win].T                                         # (n*m, L)
    tgt_idx = np.arange(n_win) + n_order - 1 + lag
    P = truth[tgt_idx].T                                    # (m, L)
    return D, P


# --------------------------------------------------------------------------- #
#  Filter derivation  (eqs. 10-14, 16-18)
# --------------------------------------------------------------------------- #
def eof_filter(D, P, k_modes=None, tikhonov=0.0):
    """Optimal least-squares predictive filter F minimising ||F D - P||^2.

    F = P (D^T)^+  computed via SVD of D = U S V^T, so (D^T)^+ = V S^+ U^T and
    F = P V S^+ U^T.  The columns of U are the EOFs.

    k_modes  : keep only the k largest singular values / EOFs (None => all).
    tikhonov : regularisation parameter lambda (eq. 16); damps small modes via
               s/(s^2 + lambda^2) instead of a hard truncation.  Use one or the
               other; both can be combined.
    """
    U, s, Vt = np.linalg.svd(D, full_matrices=False)        # U:(nm,r) s:(r,) Vt:(r,L)
    if k_modes is not None: #if passed k_modes, keep only the k strongest EOFs
        s = s.copy()
        s[k_modes:] = 0.0
    if tikhonov > 0.0:
        s_inv = s / (s ** 2 + tikhonov ** 2)
    else:
        s_inv = np.zeros_like(s)
        nz = s > 0
        s_inv[nz] = 1.0 / s[nz]
    # F = P @ V @ diag(s_inv) @ U^T
    F = (P @ Vt.T) * s_inv @ U.T                            # (m, n*m)
    return F, s


# --------------------------------------------------------------------------- #
#  Causal application of the filter  (eqs. 3, 20)
# --------------------------------------------------------------------------- #
def apply_filter(meas, truth, F, n_order, lag):
    """Run the filter causally over a stream, predicting `lag` steps ahead.

    Returns dict aligned on the prediction-target time index:
        pred   : (N, m) predicted position at t+lag
        true   : (N, m) noise-free position at t+lag
        latest : (N, m) newest available measurement meas(t)  -> non-predictive
                        "last measurement" correction (still carries lag+noise)
        t_idx  : (N,)   target sample indices
    """
    #same first few lines to set up the history matrix H
    T, m = meas.shape
    w = sliding_window_view(meas, n_order, axis=0)
    H = w[:, :, ::-1].transpose(0, 2, 1).reshape(w.shape[0], n_order * m)
    n_win = T - n_order - lag + 1
    H = H[:n_win]                                           # (N, n*m)

    #apply the filter to get the prediction
    #the filter is a block of weights, learned from the training
    #the real data follows the same physics
    #therefore the same filter is applicable
    pred = H @ F.T                                          # (N, m)

    #the now
    end_k = np.arange(n_win) + n_order - 1                  # newest-sample index
    
    #the target we want to predict is at t+lag, so we need to add lag to the index
    t_idx = end_k + lag

    #return the prediction, the true value, the latest measurement, and the target indices
    return dict(pred=pred, true=truth[t_idx], latest=meas[end_k], t_idx=t_idx)


# --------------------------------------------------------------------------- #
#  Tip/tilt vibration generator
# --------------------------------------------------------------------------- #
def make_vibrations(freqs, amps, axes, phases, fs, duration, noise, seed=0):
    """Sum of 2D sinusoidal vibrations -> (truth, meas), both (T, 2) in mas.

    Each vibration k oscillates along a line in the X-Y plane at position angle
    axes[k] (rad), i.e.  pos(t) = amps[k]*sin(2*pi*freqs[k]*t + phases[k]) *
    [cos(axis), sin(axis)].  `noise` is the per-axis Gaussian measurement sigma.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(round(duration * fs))) / fs
    truth = np.zeros((t.size, 2))
    for f, A, ax, ph in zip(freqs, amps, axes, phases):
        s = A * np.sin(2 * np.pi * f * t + ph)
        truth[:, 0] += s * np.cos(ax)
        truth[:, 1] += s * np.sin(ax)
    meas = truth + rng.normal(0.0, noise, truth.shape)
    return truth, meas


def rms(a):
    return float(np.sqrt(np.mean(np.square(a))))