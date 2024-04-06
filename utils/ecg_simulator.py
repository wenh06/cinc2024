r"""
The EKF2 model for ECG denoising is described by the following system of ODEs

.. math::

    \begin{equation}
    \label{ode_system}
    \begin{cases}
        \dot{r} = r(1-r) \\
        \dot{\theta} = \omega \\
        \dot{z} = \sum\limits_{i\in\{P,Q,R,S,T\}} \dfrac{\alpha_i\omega}{b_i^2} \Delta\theta_i \exp{\left(-\dfrac{(\Delta\theta_i)^2}{2b_i^2}\right)}
    \end{cases}
    \end{equation}

where $r$ is the RR interval, $\theta$ is the phase, $z$ is the ECG data, $\omega$ is the angular frequency, $\alpha_i$ is the amplitude, $b_i$ is the standard deviation, and $\Delta\theta_i = \theta-\theta_i$.

The discrete form is

.. math::

    \begin{equation}
    \label{discrete_form}
    \begin{cases}
        \theta_{k+1} = (\theta_k+\omega\delta) \mod{2\pi} \\
        z_{k+1} = -\sum\limits_{i\in\{P,Q,R,S,T\}} \delta \dfrac{\alpha_i\omega}{b_i^2} \Delta\theta_i \exp{\left(-\dfrac{(\Delta\theta_i)^2}{2b_i^2}\right)} + z_k + \eta
    \end{cases}
    \end{equation}

The ODE system \eqref{ode_system} has solution

.. math::

    \begin{equation}
    \label{ode_solution}
    z(t) = \sum\limits_{i\in\{P,Q,R,S,T\}} \alpha_i \exp{\left(-\dfrac{(\Delta\theta_i(t))^2}{2b_i^2}\right)} + const
    \end{equation}

Let the loss (objective) function for the curve fitting be

.. math::

    \begin{equation}
    \label{loss_function}
    L = \sqrt{\sum\limits_t (s(t)-z(t))^2}
    \end{equation}

where $s(t)$ is the measurement value, $z(t)$ is the evovled value, at time $t$. Write $\Delta(t) = z(t)-s(t)$, then

.. math::

    \begin{equation}
    \label{jacobian_loss_function}
    \begin{cases}
    \dfrac{\partial L}{\partial \alpha_i} = \dfrac{\sum\limits_t \Delta(t) \exp{\left( -\dfrac{(\Delta\theta_i)^2}{2b_i^2} \right)}}{\sqrt{\sum\limits_t \Delta(t)^2}} \\
    \dfrac{\partial L}{\partial b_i} = \dfrac{\sum\limits_t \Delta(t) \left( \dfrac{\alpha_i(\Delta\theta_i)^2}{b_i^3} \right) \exp{\left( -\dfrac{(\Delta\theta_i)^2}{2b_i^2} \right)}}{\sqrt{\sum\limits_t \Delta(t)^2}}
    \end{cases}
    \end{equation}

The EKF2 model can also be used to simulate the ECG data.

References
----------
[1] Sameni R, Shamsollahi M B, Jutten C, et al. A nonlinear Bayesian filtering framework for ECG denoising[J]. IEEE Transactions on Biomedical Engineering, 2007, 54(12): 2172-2185.
[2] Clifford G D, Shoeb A, McSharry P E, et al. Model-based filtering, compression and classification of the ECG[J]. International Journal of Bioelectromagnetism, 2005, 7(1): 158-161.

"""

from numbers import Real
from typing import Dict, Optional

import numpy as np
from scipy.ndimage import median_filter

from .utils_stats import modulo

__all__ = ["evolve_ecg", "evolve_standard_12_lead_ecg", "generate_rr_interval"]


ALL_WAVES = ["p", "q", "r", "s", "t"]

DIM_STATE = 2

DEFAULT_SINGLE_LEAD_PARAMS = {
    "bpm_std": 0.1,  # ratio
    "lf_hf": 0.5,
    "alpha": {"p": 0.16 * 1000, "q": -0.1 * 1000, "r": 1.5 * 1000, "s": -0.2 * 1000, "t": 0.5 * 1000},
    "b": {"p": 0.15, "q": 0.06, "r": 0.08, "s": 0.08, "t": 0.35},
    "theta": {"p": -0.22 * 2 * np.pi, "q": -0.05 * 2 * np.pi, "r": 0, "s": 0.05 * 2 * np.pi, "t": 0.36 * 2 * np.pi},
    "omega": 2 * np.pi / 1,
    "eta": 0,
}

DEFAULT_12_LEAD_PARAMS = {
    "bpm_std": 0.1,  # ratio
    "lf_hf": 0.5,
    "alpha": {
        # ordering: lead I, II, V1, V2, V3, V4, V5, V6
        # NOTE: lead III, aVR, aVL, aVF can be derived from lead I, II
        "p": 1000 * np.array([0.09, 0.16, 0.07, 0.03, 0.04, 0.06, 0.06, 0.06]),
        "q": 1000 * np.array([-0.05, -0.1, 0.02, 0.02, -0.02, -0.03, -0.05, -0.08]),
        "r": 1000 * np.array([0.9, 1.5, 0.25, 0.4, 0.75, 1.2, 1.3, 1.4]),
        "s": 1000 * np.array([-0.11, -0.2, -1.3, -1.7, -0.8, -0.2, -0.02, -0.02]),
        "t": 1000 * np.array([0.3, 0.5, 0.3, 0.85, 0.6, 0.4, 0.3, 0.35]),
    },
    "b": {"p": 0.15, "q": 0.06, "r": 0.08, "s": 0.08, "t": 0.35},
    "theta": {"p": -0.22 * 2 * np.pi, "q": -0.05 * 2 * np.pi, "r": 0, "s": 0.05 * 2 * np.pi, "t": 0.36 * 2 * np.pi},
    "omega": 2 * np.pi / 1,
    "eta": 0,
}


DEFAULT_NOISE_RATIO = {
    "alpha": {w: 0.023 for w in ALL_WAVES},
    "b": {w: 0.015 for w in ALL_WAVES},
    "theta": {w: 0.015 for w in ALL_WAVES},
    "eta": 0.015,
    "omega": 0.015,
}

DEFAULT_INIT_VALS = np.array([0, 0], dtype=float)


def evolve_ecg(
    t: float,
    fs: int,
    bpm: Real,
    params: Optional[dict] = None,
    init_vals: Optional[np.ndarray] = None,
    noise_ratio: Optional[dict] = None,
    remove_baseline: float = 0.66,
    return_phase: bool = False,
    return_format: str = "channel_first",
    verbose: int = 0,
) -> Dict[str, np.ndarray]:
    """Evolve single-lead ecg model using the state transition function with duration t.

    Parameters
    ----------
    t : float
        Time duration to evolve.
    fs : int
        Sampling frequency of the ecg data.
    params : dict, optional
        Parameters of the ecg model, consisting of
        - "bpm_std": the standard deviation of the bpm divided by the bpm
        - "lf_hf": the ratio of lf energy to hf energy
        - "alpha": the parameters of the Gaussian function for each wave
        - "b": the parameters of the Gaussian function for each wave
        - "theta": the parameters of the Gaussian function for each wave
        - "omega": the angular frequency
        - "eta": the noise of the ecg data
    init_vals : `array_like`, optional
        Initial values to start evolving, of shape ``(DIM_STATE,)``.
    noise_ratio : dict, optional
        Noise ratio of each state parameters.
    remove_baseline : float, default 0.66
        Proportion of the baseline to be removed.
    return_phase : bool, default False
        Whether to return the phase of the evolved ecg data.
    return_format : {"channel_first", "channel_last", "lead_first", "lead_last", "flat"}, default "channel_first"
        The format of the returned ecg data,
        either "channel_first" (alias "lead_first") or "channel_last" (alias "lead_last") for multi-lead ECG,
        or "flat" for single-lead ECG.
    verbose : int, default 1
        Verbosity level.

    Returns
    -------
    dict
        The evolved ecg data, consisting of
        - "ecg": the evolved ecg data
        - "rr_intervals": rr intervals of the evolved ecg data
        - "r_peak_indices": indices of r peaks of the evolved ecg data

        If `params["alpha"]` indicates single lead ECG,
        then "ecg" is of shape ``(len_pts, DIM_STATE)`` if return_phase is True, otherwise ``(len_pts,)``.
        If `params["alpha"]` indicates multi-lead ECG,
        then "ecg" is of shape ``(len_pts, DIM_STATE)`` if return_phase is True, otherwise ``(len_pts, nb_leads)``.

    References
    ----------
    [1] Sameni R, Shamsollahi M B, Jutten C, et al. A nonlinear Bayesian filtering framework for ECG denoising[J]. IEEE Transactions on Biomedical Engineering, 2007, 54(12): 2172-2185.
    [2] Clifford G D, Shoeb A, McSharry P E, et al. Model-based filtering, compression and classification of the ECG[J]. International Journal of Bioelectromagnetism, 2005, 7(1): 158-161.

    """
    assert t > 0, "please provide a positive time duration"
    assert fs > 0, "please provide a positive sampling frequency"
    assert 30 <= bpm <= 300, "bpm should be in the range of [30, 300]"
    assert 0 <= remove_baseline <= 1, "proportion of the baseline to be removed should be in the range of [0, 1]"
    if params is None:
        params = DEFAULT_SINGLE_LEAD_PARAMS.copy()
    else:
        params = {k: v for k, v in DEFAULT_SINGLE_LEAD_PARAMS.items() if k not in params} | params
    if init_vals is None:
        init_vals = DEFAULT_INIT_VALS.copy()
    if noise_ratio is None:
        noise_ratio = DEFAULT_NOISE_RATIO.copy()
    else:
        noise_ratio = {k: v for k, v in DEFAULT_NOISE_RATIO.items() if k not in noise_ratio} | noise_ratio
    if verbose >= 1:
        print(f"{params =}")
        print(f"{init_vals =}")
        print(f"{noise_ratio =}")

    # if "alpha" is an array for all waves, then they must have the same length
    if isinstance(params["alpha"][ALL_WAVES[0]], (list, tuple, np.ndarray)):
        nb_leads = len(params["alpha"][ALL_WAVES[0]])
        assert all(
            len(params["alpha"][w]) == nb_leads for w in ALL_WAVES
        ), f"all waves should have the same length (={nb_leads}) for alpha"
        # convert to numpy array if necessary
        for w in ALL_WAVES:
            params["alpha"][w] = np.array(params["alpha"][w], dtype=float)
    else:
        nb_leads = 1
    if nb_leads > 1 and len(init_vals) == DIM_STATE:
        init_vals = np.array([init_vals[0]] + [init_vals[1]] * nb_leads, dtype=float)
    else:
        init_vals = np.array(init_vals, dtype=float)

    spacing = 1 / fs
    # len_pts = int(t / params["delta"]) + 1
    len_pts = int(t * fs) + 1
    rr_mean = 60 / bpm
    nb_beats = int(np.ceil(t / rr_mean)) + 5  # +5 to ensure that beats are enouph

    rr_intervals = generate_rr_interval(nb_beats=nb_beats, bpm_mean=bpm, bpm_std=params["bpm_std"], lf_hf=params["lf_hf"])

    # make rr intervals to be multiples of spacing,
    # very important, otherwise distortion would occur
    rr_intervals_nb_pts = np.vectorize(lambda n: int(round(n * fs)))(rr_intervals)
    r_peak_indices = [0] + np.cumsum(rr_intervals_nb_pts).tolist()
    # one has to add nb_r_peaks by 1
    nb_r_peaks = len([r for r in r_peak_indices if r < len_pts]) + 1
    r_peak_indices = r_peak_indices[:nb_r_peaks]
    rr_intervals = rr_intervals_nb_pts[:nb_r_peaks] * spacing
    r_peak_t = [0] + np.cumsum(rr_intervals).tolist()

    if verbose >= 1:
        print(f"{rr_intervals =}")
        print(f"{r_peak_t =}")
        print(f"{r_peak_indices =}")

    noise_info = {"alpha": {}, "b": {}, "theta": {}, "eta": [], "omega": []}
    for k in ["alpha", "b", "theta"]:
        noise_info[k] = {
            w: (
                np.random.normal(0, np.abs(noise_ratio[k][w] * params[k][w]), nb_beats)
                if isinstance(params[k][w], (Real, np.generic))
                else np.random.normal(0, np.abs(noise_ratio[k][w] * params[k][w]), (nb_beats, len(params[k][w])))
            )
            for w in ALL_WAVES
        }
    noise_info["omega"] = np.random.normal(0, np.abs(noise_ratio["eta"] * params["omega"]), nb_beats)
    if isinstance(params["alpha"]["r"], (Real, np.generic)):
        noise_info["eta"] = np.random.normal(0, np.abs(noise_ratio["eta"] * params["alpha"]["r"]), len_pts)
    else:
        noise_info["eta"] = np.random.normal(0, np.abs(noise_ratio["eta"] * params["alpha"]["r"]), (len_pts, nb_leads))

    if verbose >= 2:
        print(f"{noise_info =}")

    # start evolving
    init_theta = init_vals[0]
    init_z = init_vals[1:]
    synthetic_ecg = [[init_theta] + init_z.tolist()]
    theta, z = init_theta, init_z
    beat_no = 0
    idx = 0
    current_beat_idx = 0
    while idx < len_pts - 1:
        current_params = {
            "eta": params["eta"] + noise_info["eta"][idx],
            "delta": spacing,
        }
        current_params["omega"] = 2 * np.pi / rr_intervals[beat_no] + noise_info["omega"][beat_no]
        for k in ["alpha", "b", "theta"]:
            current_params[k] = {w: params[k][w] + noise_info[k][w][beat_no] for w in ALL_WAVES}

        new_state = _state_transition_func(
            fs=fs,
            state_vec=np.array([[theta] + z.tolist()]),
            state_params=current_params,
        )
        theta = new_state[0]
        z = new_state[1:]

        idx += 1
        current_beat_idx += 1
        if current_beat_idx >= rr_intervals_nb_pts[beat_no]:
            if verbose >= 2:
                print(
                    f"at the beat-split point, the {idx}-th point, and the {current_beat_idx}-th point of the current_beat, "
                    f"current state vector is theta = {theta} (= {theta / np.pi}pi), z = {z}"
                )
            current_beat_idx = 0
            beat_no += 1
        synthetic_ecg.append([theta] + z.tolist())

    synthetic_ecg = np.array(synthetic_ecg, dtype=float)
    if remove_baseline:
        synthetic_ecg[:, 1:] = remove_base_line(synthetic_ecg[:, 1:], fs, proportion=remove_baseline)
    if not return_phase:
        synthetic_ecg = synthetic_ecg[:, 1:]

    if return_format in ["channel_first", "lead_first"]:
        synthetic_ecg = synthetic_ecg.T
    elif return_format in ["flat"]:
        assert nb_leads == 1, "only single lead ECG can be flattened"
        synthetic_ecg = synthetic_ecg.flatten()
    elif return_format in ["channel_last", "lead_last"]:
        pass
    else:
        raise ValueError(
            "return_format should be one of {'channel_first', 'channel_last', 'lead_first', 'lead_last', 'flat'} "
            f"but got {return_format}"
        )

    ret = {
        "ecg": synthetic_ecg,
        "rr_intervals": np.array(rr_intervals[:-1], dtype=float),
        "r_peak_indices": np.array(r_peak_indices[:-1], dtype=int),
    }
    return ret


def evolve_standard_12_lead_ecg(
    t: float,
    fs: int,
    bpm: Real,
    params: Optional[dict] = None,
    init_vals: Optional[np.ndarray] = None,
    noise_ratio: Optional[dict] = None,
    remove_baseline: float = 0.66,
    return_phase: bool = False,
    return_format: str = "channel_first",
    verbose: int = 0,
) -> Dict[str, np.ndarray]:
    """Evolve single-lead ecg model using the state transition function with duration t.

    Parameters
    ----------
    t : float
        Time duration to evolve.
    fs : int
        Sampling frequency of the ecg data.
    params : dict, optional
        Parameters of the ecg model, consisting of
        - "bpm_std": the standard deviation of the bpm derived by the bpm
        - "lf_hf": the ratio of lf energy to hf energy
        - "alpha": the parameters of the Gaussian function for each wave
        - "b": the parameters of the Gaussian function for each wave
        - "theta": the parameters of the Gaussian function for each wave
        - "omega": the angular frequency
        - "eta": the noise of the ecg data
    init_vals : `array_like`, optional
        Initial values to start evolving, of shape ``(DIM_STATE,)``.
    noise_ratio : dict, optional
        Noise ratio of each state parameters.
    remove_baseline : float, default 0.66
        Proportion of the baseline to be removed.
    return_phase : bool, default False
        Whether to return the phase of the evolved ecg data.
    return_format : {"channel_first", "channel_last", "lead_first", "lead_last", "flat"}, default "channel_first"
        The format of the returned ecg data,
        either "channel_first" (alias "lead_first") or "channel_last" (alias "lead_last") for multi-lead ECG,
        or "flat" for single-lead ECG.
    verbose : int, default 1
        Verbosity level.

    Returns
    -------
    dict
        The evolved ecg data, consisting of
        - "ecg": the evolved ecg data
        - "rr_intervals": rr intervals of the evolved ecg data
        - "r_peak_indices": indices of r peaks of the evolved ecg data

        "ecg" is of shape ``(len_pts, 13)`` if return_phase is True, otherwise ``(len_pts, 12)``.

    .. note::

        params["alpha"] should be a dictionary of arrays which are amplitude of each wave for each lead.
        The arrays should have shape ``(8,)``, where the amplitudes are for leads I, II, V1, V2, V3, V4, V5, V6.
        Values for leads III, aVR, aVL, aVF will be derived from leads I, II.

    References
    ----------
    [1] Sameni R, Shamsollahi M B, Jutten C, et al. A nonlinear Bayesian filtering framework for ECG denoising[J]. IEEE Transactions on Biomedical Engineering, 2007, 54(12): 2172-2185.
    [2] Clifford G D, Shoeb A, McSharry P E, et al. Model-based filtering, compression and classification of the ECG[J]. International Journal of Bioelectromagnetism, 2005, 7(1): 158-161.
    [3] https://ecglibrary.com/norm.php
    [4] https://www.bem.fi/book/15/15.htm

    """
    if params is None:
        params = DEFAULT_12_LEAD_PARAMS.copy()
    else:
        params = {k: v for k, v in DEFAULT_12_LEAD_PARAMS.items() if k not in params} | params
    simulated_ecg = evolve_ecg(
        t=t,
        fs=fs,
        bpm=bpm,
        params=params,
        init_vals=init_vals,
        noise_ratio=noise_ratio,
        remove_baseline=remove_baseline,
        return_phase=True,
        return_format="lead_last",
        verbose=verbose,
    )
    # relation mat: by Einthoven’s Law and Goldberger’s Law
    # ref. https://www.bem.fi/book/15/15.htm
    relation_mat = np.array([[-1, -0.5, 1, -0.5], [1, -0.5, -0.5, 1]])  # shape (2, 4)
    # derive leads III, aVR, aVL, aVF from leads I, II
    derived_leads = np.matmul(simulated_ecg["ecg"][:, 1:3], relation_mat)  # shape (len_pts, 4)
    simulated_ecg["ecg"] = np.concatenate([simulated_ecg["ecg"][:, :3], derived_leads, simulated_ecg["ecg"][:, 3:]], axis=1)
    if not return_phase:
        simulated_ecg["ecg"] = simulated_ecg["ecg"][:, 1:]
    if return_format in ["channel_first", "lead_first"]:
        simulated_ecg["ecg"] = simulated_ecg["ecg"].T
    return simulated_ecg


def _state_transition_func(
    fs: int, state_vec: np.ndarray, state_params: Optional[dict] = None, nb_leads: int = 1
) -> np.ndarray:
    """The state transition function, given by Gaussian functions in EKF2 model,
    ordering of the state variables in state_vec: theta, z.

    Parameters
    ----------
    fs : int
        Sampling frequency of the ecg data.
    state_vec : `array_like`
        Vector of current state, of shape ``(DIM_STATE,)`` if single lead,
        otherwise ``(nb_leads + 1,)``.
    state_params : dict, optional
        Parameters for the state transition function.
    nb_leads : int, default 1
        Number of leads.

    Returns
    -------
    numpy.ndarray
        The evolved state vector, of shape ``(DIM_STATE,)`` if single lead,
        otherwise ``(nb_leads + 1,)``.

    """
    spacing = 1 / fs
    theta = state_vec.ravel()[0]
    z = state_vec.ravel()[1:]

    new_theta = theta + spacing * state_params["omega"]
    delta_theta = {w: modulo(theta - state_params["theta"][w], 2 * np.pi, -np.pi) for w in ALL_WAVES}
    summands = np.array(
        [
            (state_params["alpha"][w] * delta_theta[w] / pow(state_params["b"][w], 2))
            * np.exp(-pow(delta_theta[w], 2) / (2 * pow(state_params["b"][w], 2)))
            for w in ALL_WAVES
        ]
    )  # of shape (nb_waves,) or (nb_waves, nb_leads)
    new_z = z + state_params["eta"] - state_params["omega"] * spacing * np.sum(summands, axis=0)

    return np.concatenate([[new_theta], new_z])


def generate_rr_interval(
    nb_beats: int,
    bpm_mean: Real,
    bpm_std: Real,
    lf_hf: float = 0.5,
    lf_freq: float = 0.1,
    hf_freq: float = 0.25,
    lf_std: float = 0.01,
    hf_std: float = 0.01,
    verbose: int = 0,
) -> np.ndarray:
    """Generate synthetic rr intervals with given bpm and lf hf information.

    Parameters
    ----------
    nb_beats : int
        Number of beats
    bpm_mean : numbers.Real
        Mean bpm of the rr intervals to be generated
    bpm_std : numbers.Real
        Standard deviation of the rr intervals to be generated / bpm_mean
    lf_hf : float, default 0.5
        lf energey / hf energy,
        normal range: 2.2 ± 3.4 (mean ± std)
    lf_freq : float, default 0.1
        Center (mean) of the (gaussian) distribution of lf
    hf_freq : float, default 0.25
        Center (mean) of the (gaussian) distribution of hf
    lf_std : float, default 0.01
        Standard deviation of the (gaussian) distribution of lf
    hf_std : float, default 0.01
        Standard deviation of the (gaussian) distribution of hf

    Returns
    -------
    numpy.ndarray
        The generated rr interval, of shape ``(nb_beats,)``.

    """
    expected_rr_mean = 60 / bpm_mean
    bpm_std = bpm_mean * bpm_std
    expected_rr_std = 60 * bpm_std / (bpm_mean * bpm_mean)

    lf = lf_hf * np.random.normal(loc=lf_freq, scale=lf_std, size=nb_beats)  # lf power spectum
    hf = np.random.normal(loc=hf_freq, scale=hf_std, size=nb_beats)  # hf power spectum
    rr_power_spectrum = np.sqrt(lf + hf)

    # random (uniformly distributed in [0,2pi]) phases
    phases = np.vectorize(lambda theta: np.exp(2 * 1j * np.pi * theta))(
        np.random.uniform(low=0.0, high=2 * np.pi, size=nb_beats)
    )
    # real part of inverse FFT of complex spectrum
    raw_rr = np.real(np.fft.ifft(rr_power_spectrum * phases)) / nb_beats
    raw_rr_std = np.std(raw_rr)
    ratio = expected_rr_std / raw_rr_std
    rr = (raw_rr * ratio) + expected_rr_mean

    return rr


def remove_base_line(curve: np.ndarray, fs: int, proportion: float = 0.66) -> np.ndarray:
    """Remove the baseline of the ecg curve using 200 ms median filter and 600 ms median filter.

    Parameters
    ----------
    curve : numpy.ndarray
        The ecg curve to remove baseline, of shape ``(len_pts, nb_leads)``.
    fs : int
        Sampling frequency of the ecg data.
    proportion : float, default 0.66
        Proportion of the baseline to be removed.

    Returns
    -------
    numpy.ndarray
        The ecg curve with baseline removed.

    """
    # apply median filter
    wind_1 = int(0.2 * fs)  # 200 ms window
    wind_1 = wind_1 if wind_1 % 2 == 1 else wind_1 + 1  # window size must be odd
    wind_2 = int(0.6 * fs)  # 600 ms window
    wind_2 = wind_2 if wind_2 % 2 == 1 else wind_2 + 1
    baseline = median_filter(curve, size=wind_1, mode="nearest", axes=0)
    baseline = median_filter(baseline, size=wind_2, mode="nearest", axes=0)
    return curve - baseline * proportion
