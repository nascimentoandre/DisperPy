import numpy as np
import os
from obspy import read
from obspy.io.sac.util import SacIOError
from obspy.io.sac import SACTrace
from obspy.signal.invsim import cosine_taper
from obspy.signal.filter import bandpass
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN

class DispersionPicker(object):
    """
    Class to extract group velocity dispersion curve from waveform data in the
    SAC format using unsupervised machine learning algorithms.

    Parameters
    ----------
    file: str
        Path to the SAC file relative to the script. Example:
        `2022/CX.PB03.20220213_082150..HHZ.sac`.

    fname: str
        Name of the SAC file after excluding the root folder. Example:
        `CX.PB03.20220213_082150..HHZ.sac`

    wave: str
        Type of wave whose dispersion is intended to be measured. Must be
        `rayleigh` (default) or `love`.

    measure_type: str, default = raw         
        Refers to the type of measurement being performed and can be assigned
        either `raw` (a first measurement before the application of a
        phase-matched filter) or `clean` (if the measurement is being performed
        on a clean trace).

    Tmin: int, default = 8
        Minimum period in seconds being analyzed.
    
    Tmax: int, default = 310
        Maximum period in seconds being analyzed.

    vmin: float, default = 2.0
        Minimum group velocity value allowed in km/s.
    
    Tmax: float, default = 5.0
        Maximum group velocity value allowed in km/s.

    alpha: int or NoneType, default = None
        Value for the alpha parameter used in the frequency-time analysis. If
        not set by the user, an appropriate value will be chosen based on the
        epicentral distance after function call :meth:`set_alpha`.

    n_filts: int, default = 100
        Number of filters used in the frequency-time analysis. `n_filt` filter
        frequencies will be set to be evenly spaced on a log scale between
        1/`Tmax` and 1/`Tmin`.

    Tmin_short_dist: int, default = 5
        Minimum period (in s) in case the epicentral distance is shorter than
        `short_dist_value`.

    short_dist_value: float, default = 2500
        Epicentral distance threshold below which the minimum period being
        analyzed can be reduced.

    snr: bool
        Whether or not to perform filtering of group velocity/period based
        signal-to-noise ratio criteria. Default is `True`.

    noise_window_offset: int, default = 900
        Offset in seconds between the end of the signal window and the start of
        the noise window used in the signal-to-noise ratio analysis.

    noise_window_size: int, default = 500
        Size in seconds of the noise window.

    keep_no_snr: bool
        Whether or not to keep measurements where the signal-to-noise ratio
        could not be calculated. Default is to discard (`False`).

    short_dist: bool
        Whether or not to decrease `Tmin` if the epicentral distance is below
        `short_dist_value`. Default is `True`.

    min_period_range: int, default = 5
        The minimum period range allowed in the final measured dispersion
        curve. Curves below this threshold will be discarded. 

    n_clusters_raw: int, default = 8
        Number of clusters for the K-means analysis if the trace is raw.

    n_clusters_clean: int, default = 3
        Number of clusters for the K-means analysis if the trace is clean
        (phase-matched filtered). 

    eps_raw: float
        Maximum distance between samples in the DBSCAN algorithm for them to be
        considered in the vicinity of each other. In DisperPy, smaller values
        of eps correspond to more conservative picks, while greater values
        might pick lower frequencies but also noisy points. A value that works
        well for earthquake data is 0.10 (Default).

    eps_clean: float
        Maximum distance between samples in the DBSCAN algorithm for them to be
        considered in the vicinity of each other. For clean traces, larger
        values are recommended compared to raw traces, and the default is
        `0.25`.

    save_residual: bool
        Whether or not to save residual traces after the application of the
        phase-matched filter, i.e., traces without the fundamental mode surface
        wave signal. Default is `False`.

    Attributes
    ----------

    tr: obspy.core.trace.Trace
        Obspy object containing a seismic trace. Available after function call
        :meth:`load_file`.
    
    metadata: dict
        Dictionary containing the following metadata information: origin time,
        number of sample points, sampling rate, distance between samples and
        epicentral distance (in km). Available after function call
        :meth:`load_file`.

    x: numpy.ndarray
        Waveform samples in the numpy array format. Available after function call
        :meth:`load_file`.

    x_filt: numpy.ndarray
        Waveform samples in the numpy array format after the application of the
        phase-matched filter. Available after function call
        :meth:`phase_matched_filter`.
 
    t: numpy.ndarray
        Time vector of the waveform in the numpy array format. Available after function call
        :meth:`load_file`.

    max_period: float
        Maximum period allowed by the epicentral distance threshold (dist/12)
        as established by Bensen et al. (2007). Available after function call
        :meth:`load_file`.

    Tc: list
        Vector containing central periods of the dispersion spectrogram.
        Available after function call :meth:`FTAN`.

    T: list
        Vector containing instantaneous periods of the dispersion spectrogram.
        Available after function call :meth:`FTAN`.

    Vg: list
        Vector containing group velocities of the dispersion spectrogram.
        Available after function call :meth:`FTAN`.

    A: list
        Vector containing amplitudes of the dispersion spectrogram. Available
        after function call :meth:`FTAN`.

    X_train: numpy.ndarray
        Matrix containing instantaneous periods, group velocities and
        normalized amplitudes of the dispersion spectrogram. Available after
        function call :meth:`clusters`.

    labels: numpy.ndarray
        Array containing a K-means label for each (period, group velocity,
        normalized amplitude) triplet. Available after function call
        :meth:`clusters`.

    X_train_filt: numpy.ndarray
        Filtered matrix by Gaussian mixtures containing instantaneous periods,
        group velocities and normalized amplitudes of the dispersion
        spectrogram. Available after function call :meth:`measure_trial_curve`.

    curve: numpy.ndarray of size (n, 2)
        Preliminary array containing n picks of period and group velocity. Some
        noise points might be picked at this stage. Available after function
        call :meth:`measure_trial_curve`.

    curve_filt: numpy.ndarray of size (n, 2)
        Dispersion curve after cleaning using the function
        :meth:`filter_curve`.

    References
    ----------
    Bensen, G. D., Ritzwoller, M. H., Barmin, M. P., Levshin, A. L., Lin, F.,
    Moschetti, M. P., ... and Yang, Y. (2007). Processing seismic ambient noise
    data to obtain reliable broad-band surface wave dispersion measurements.
    Geophysical journal international, 169(3), 1239-1260
        
    """
    def __init__(self, file, wave="rayleigh", measure_type="raw", Tmin=8, Tmax=310, vmin=2.0, vmax=5.0, alpha=None, n_filts=100, Tmin_short_dist=5, snr=True, noise_window_offset=900, noise_window_size=500, keep_no_snr=False, short_dist=True, short_dist_value=2500, min_period_range=5, n_clusters_raw=8, n_clusters_clean=3, eps_raw=0.10, eps_clean=0.25, save_residual=False):

        assert measure_type == "raw" or measure_type == "clean"
        assert wave == "rayleigh" or wave == "love"
        assert 0 < vmin < vmax
        assert 0 < Tmin < Tmax

        self.file = file
        self.fname = file.split(os.sep)[-1]
        self.measure_type = measure_type
        self.wave = wave
        self.Tmin = Tmin
        self.Tmax = Tmax
        self.vmin = vmin
        self.vmax = vmax
        self.alpha = alpha
        self.n_filts = n_filts
        self.short_dist = short_dist
        self.snr = snr
        self.Tmin_short_dist = Tmin_short_dist
        self.noise_window_offset = noise_window_offset
        self.noise_window_size = noise_window_size
        self.keep_no_snr = keep_no_snr
        self.short_dist_value = short_dist_value
        self.min_period_range = min_period_range
        self.n_clusters_raw = n_clusters_raw
        self.n_clusters_clean = n_clusters_clean
        self.save_residual = save_residual
        self.eps_raw = eps_raw
        self.eps_clean = eps_clean

    def load_file(self):
        """
        Loads the SAC file. If preset samples are included, this function also
        trims them out and ensures that the first sample correspond to the
        origin time. If the origin time sample is not included in the trace,
        the loading operation fails.
        """
        self.x = None
        self.metadata = None

        try:
            st = read(self.file)
        except:
            print("Could not read file %s, skipping."%self.file)
            return

        tr = st[0]
        self.tr = tr

        self.metadata = {}

        # if preset samples are present, let's exclude them
        if tr.stats.sac["o"] > 0.0:
            tstart = tr.stats.starttime
            torig = tstart + tr.stats.sac["o"]
            self.metadata["ot"] = torig
            self.metadata["preset"] = True
            tend = tr.stats.endtime
            try:
                tr = tr.trim(torig, tend)
            except ValueError:
                print("Could not trim file %s, skipping it."%self.file)
                self.x = None
                return
        elif tr.stats.sac["o"] == 0.0:
            self.metadata["preset"] = False
        else:
            print("The origin time sample is not included in the time series, skipping %s."%self.file)
            self.x = None
            return

        self.metadata["npts"] = len(tr)
        self.metadata["sampling_rate"] = tr.stats.sampling_rate
        self.metadata["dt"] = tr.stats.delta
        self.metadata["dist"] = tr.stats.sac["dist"]

        # set Tmin to a lower value if dist is below a certain threshold
        if self.metadata["dist"] <= self.short_dist_value:
            self.Tmin = self.Tmin_short_dist

        self.x = tr.data
        self.t = np.arange(0, self.metadata["npts"]/self.metadata["sampling_rate"], self.metadata["dt"])

        # Defining a maximum period constraint given by Tmax = dist/12 (~ 3 wavelengths) (Bensen et al., 2007)
        self.max_period = self.metadata["dist"] / 12

    def set_alpha(self, dist):
        """
        Sets the alpha parameter necessary for frequency-time analysis to a
        value based on the epicentral distance in km.
        """
        if dist < 200:
            alpha = 10
        elif 200 <= dist < 400:
            alpha = 15
        elif 400 <= dist < 1000:
            alpha = 25
        elif 1000 <= dist < 2000:
            alpha = 50
        elif 2000 <= dist < 4000:
            alpha = 100
        elif 4000 <= dist < 8000:
            alpha = 200
        elif 8000 <= dist < 12000:
            alpha = 250
        else:
            alpha = 300
        return alpha

    def FTAN(self):
        """
        Frequency-time analysis of a seismic waveform (Dziewonski et al., 1969;
        Levshin et al., 1992). This function sets an alpha value by calling
        :meth:`set_alpha` (if not given by the user), translates the signal to
        the frequency domain via Fourier transform, computes its analytical
        signal, applies Gaussian bandpass filtering, translates the filtered
        analytical signal back to the time domain, and finally stores vectors
        containing group velocities, periods, and amplitudes.

        References 
        ---------- 
        Dziewonski, A., Bloch, S., and Landisman, M.
        (1969). A technique for the analysis of transient seismic signals.
        Bulletin of the Seismological Society of America, 59(1), 427-444.

        Levshin, A., Ratnikova, L., and Berger, J. O. N. (1992). Peculiarities
        of surface-wave propagation across central Eurasia. Bulletin of the
        Seismological Society of America, 82(6), 2464-2493.
        """
        self.T, self.Tc, self.Vg, self.A = [], [], [], []

        if not self.alpha:
            self.alpha = self.set_alpha(self.metadata["dist"])

        # fourier transform of the time series
        X = np.fft.fft(self.x)

        # frenquency vector
        freq_vec = np.fft.fftfreq(len(X), d=self.metadata["dt"])

        # analytical signal
        X[freq_vec < 0] = 0.0
        X[freq_vec > 0] *= 2.0

        # define gaussian filters
        filter_freqs = np.logspace(np.log10(1/self.Tmax), np.log10(1/self.Tmin), self.n_filts)

        # filtering the analytical signal
        for cf in filter_freqs:
            X_filt = X * np.e**(-self.alpha*((freq_vec-cf)/cf)**2)
            x_filt = np.fft.ifft(X_filt)
            # envelope, instantaneous phase, instantaneous frequency
            E = np.abs(x_filt)
            iphase = np.unwrap(np.angle(x_filt))
            try:
                ifreq = (np.gradient(iphase)/(2.0*np.pi*self.metadata["dt"]))[np.argmax(E)-1]
            except ValueError:
                print("The shape of the array is too small to calculate the gradient, skipping %s."%self.file)
                return

            Tc = np.round(1/cf, 2)
            iT = np.round(1/ifreq, 2) if ifreq != 0 else 0

            # saving the results in attributes
            for n, A in enumerate(E.real):
                if (self.t[n] > 0) and (self.vmin < self.metadata["dist"]/self.t[n] < self.vmax):
                    group_vel = self.metadata["dist"]/self.t[n]
                    self.Tc.append(Tc)
                    self.T.append(iT)
                    self.Vg.append(group_vel)
                    self.A.append(A)

    def phase_matched_filter(self):
        """
        This function uses a raw dispersion curve to construct and apply a
        phase-matched filter to a waveform, aiming to filter out segments that
        do not correspond to the fundamental mode surface wave energy, thus
        maximizing signal-to-noise ratio and smoothing the final dispersion
        curve. Here we follow the theoretical framework by Levshin et al.
        (1992) and Levshin and Ritzwoller (2001).

        This function needs the `curve_filt` attribute to be computed, so it
        should only be called after :meth:`filter_curve`.

        References
        ----------
        Levshin, A., Ratnikova, L., and Berger, J. O. N. (1992). Peculiarities
        of surface-wave propagation across central Eurasia. Bulletin of the
        Seismological Society of America, 82(6), 2464-2493.

        Levshin, A. L., and Ritzwoller, M. H. (2001). Automated detection,
        extraction, and measurement of regional surface waves. Monitoring the
        comprehensive nuclear-test-ban treaty: Surface waves, 1531-1545.
        """
        def window_size(ts, tshift, dt):
            # auxiliary function to calc the size of the time window
            # around which to taper the compressed signal
            avg = np.mean(np.abs(ts))
            std = np.std(np.abs(ts))
            ts = np.abs(ts)
            n_samples = 0
            for i in ts[int(tshift/dt)-1:]:
                if i >= avg:
                    n_samples += 1
                else:
                    break
            return n_samples

        T, gv = self.curve_filt[:, 0], self.curve_filt[:, 1]
        fs = np.flip(1/T)
        gs = np.flip(1/gv)

        # shift Rayleigh waves to the center
        tshift = (len(self.x) * self.metadata["dt"]) / 2

        # calc analytical signal
        X = np.fft.fft(self.x)
        freq_vec = np.fft.fftfreq(len(self.x), d=self.metadata["dt"])
        X[freq_vec < 0] = 0.0
        X[freq_vec > 0] *= 2.0

        # calc phase correction
        k = np.zeros(len(fs))
        k[1:] = 2*np.pi*cumulative_trapezoid(y=gs, x=fs)
        psi = k * self.metadata["dist"]
        interp_func = interp1d(x=fs, y=psi)

        # compress the signal (undisperse it)
        mask = (freq_vec >= fs.min()) & (freq_vec <= fs.max())
        psi_int = interp_func(freq_vec[mask])
        X[mask] = X[mask] * np.exp(1j * psi_int) * np.exp(-1j*tshift*2*np.pi*freq_vec[mask])
        X[~mask] = 0.0

        # return the compressed signal to the time domain and taper it
        compressed_x = np.fft.ifft(X)
        s = window_size(compressed_x, tshift, self.metadata["dt"])
        window = (self.t <= tshift + s) & (self.t >= tshift - s)
        compressed_x[window] *= cosine_taper(npts=window.sum(), p=0.05)
        compressed_x[~window] = 0.0

        # now return to the frequency domain and redisperse the signal
        X = np.fft.fft(compressed_x)
        X[mask] = X[mask] * np.exp(-1j * psi_int) * np.exp(1j*tshift*2*np.pi*freq_vec[mask])
        X[~mask] = 0.0

        # return to the time domain and save
        self.x_filt = np.real(np.fft.ifft(X))
        residual = self.x - self.x_filt

        if self.metadata["preset"]:
            header = self.update_sac_header(self.tr.stats.sac)
            sac = SACTrace(data=self.x_filt, **header)
            sacr = SACTrace(data=residual, **header)
            sac.write("results/clean_waveforms/%ss"%self.fname)
            if self.save_residual:
                sacr.write("results/residual_waveforms/%sr"%self.fname)
        else:
            self.tr.data = self.x_filt
            residual_tr = self.tr.copy()
            residual_tr.data = residual
            self.tr.write("results/clean_waveforms/%ss"%self.fname, format="SAC")
            if self.save_residual:
                residual_tr.write("results/residual_waveforms/%sr"%self.fname, format="SAC")

    def update_sac_header(self, header):
        """
        Updates SAC header metadata in the case where preset samples are present.
        """
        h_dic = dict(header)
        for info in ("depmin", "depmax", "b", "e", "a", "depmen", "iftype", "idep", "iztype"):
            if info in h_dic.keys():
                del h_dic[info]
        h_dic['o'] = 0.0
        h_dic['nzyear'] = self.metadata["ot"].year
        h_dic['nzjday'] = self.metadata["ot"].julday
        h_dic['nzhour'] = self.metadata["ot"].hour
        h_dic['nzmin'] = self.metadata["ot"].minute
        h_dic['nzsec'] = self.metadata["ot"].second
        h_dic['nzmsec'] = round(self.metadata["ot"].microsecond / 1000)
        h_dic["iftype"] = "itime"
        return h_dic

    @staticmethod
    def minmax_normalization(vec):
        "Min max normalization"
        return (vec - np.min(vec))/(np.max(vec) - np.min(vec))

    def clusters(self):
        """
        Generates a set of clusters containing the dispersion energy of the
        waveform. To do so, it first isolates the dispersion energy from
        background noise energy using Gaussian mixtures, and then divides it
        into `n_clusters_raw` (or `n_clusters_clean`) using K-means.

        Upon calling it, the `X_train` and `labels` attributes become available.
        """
        # creating a matrix with periods, group velocities and normalized amplitudes
        self.X_train = np.array([self.T, self.Vg, self.minmax_normalization(self.A)]).T

        # using GaussianMixture to isolate low frequency noise
        try:
            gmm = GaussianMixture(n_components=2, covariance_type="full", init_params="random_from_data", random_state=42).fit(self.X_train)
        except ValueError:
            self.labels = []
            return

        gmm_labels = gmm.predict(self.X_train)
        l0 = np.max(self.X_train[gmm_labels == 0][:, 2])
        l1 = np.max(self.X_train[gmm_labels == 1][:, 2])
        if l1 > l0:
            self.X_train_filt = self.X_train[gmm_labels == 1]
        else:
            self.X_train_filt = self.X_train[gmm_labels == 0]

        if self.measure_type == "raw":
            n_clusters = self.n_clusters_raw
        elif self.measure_type == "clean":
            n_clusters = self.n_clusters_clean

        # now we classify our filtered X_train matrix into smaller clusters
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit(self.X_train_filt)
        self.labels = kmeans.labels_

    def measure_trial_curve(self):
        """
        Tracks the amplitude maxima across each cluster to pick dispersion
        segments, which are concatenated into an initial dispersion curve. In
        this stage, outlier points might get picked.
        """
        unique_labels = np.unique(self.labels)
        disp_dict = {}

        xn = self.X_train_filt[:, 0]
        yn = self.X_train_filt[:, 1]
        zn = self.X_train_filt[:, 2]
        peaks, _ = find_peaks(zn)

        for i in unique_labels:
            Xn = self.X_train_filt[self.labels == i]
            xn = Xn[:, 0]
            yn = Xn[:, 1]
            zn = Xn[:, 2]

            for j in np.arange(0, 3.0, 0.01):
                peaks, _ = find_peaks(zn, prominence=j)
                xmax = xn[peaks]
                ymax = yn[peaks]

                if len(xmax) == len(np.unique(xmax)):
                    if len(ymax) > 0:
                        disp_dict[i] = np.array([xmax, ymax]).T
                    break
                else:
                    continue

        xout = np.array([])
        yout = np.array([])
        for k in disp_dict.keys():
            xout = np.concatenate((xout, disp_dict[k][:, 0]))
            yout = np.concatenate((yout, disp_dict[k][:, 1]))
        p = xout.argsort()
        xout = xout[p]
        yout = yout[p]

        self.curve = np.array([xout, yout]).T

    def comparison_with_ref(self, x, y):
        """
        Compares velocities in dispersion segment (i.e., in each cluster) to a
        reference dispersion model. Values in the reference model are
        interpolated into the same periods as measured data. A variable `perc`
        is defined in such a way that shorter periods are allowed to have
        larger variations around the reference value, aiming to reflect the
        larger heterogeneities in the shallower Earth.

        Returns
        -------
        numpy.ndarray of shape len(x)
            Returns an array containing labels for each observation in the segment where:
            1 - observation lies within the area defined by reference +/- perc * reference
            0 - observation lies outside the area defined by reference +/- perc * reference
        """
        ref_file = "reference_curves/ak135_group_%s.txt"%self.wave

        xref, yref = np.loadtxt(ref_file, unpack=True)
        f = interp1d(xref, yref)
        yref_interp = f(x)
        comparison = []
        # setting a threshold for allowed velocities
        # longer periods are expected to have velocities closer to reference
        for i in range(len(x)):
            if x[i] < 30:
                perc = 0.30
            elif 30 < x[i] < 50:
                perc = 0.25
            elif 50 < x[i] <= 120:
                perc = 0.20
            elif 120 < x[i] <= 200:
                perc = 0.18
            elif 200 < x[i] < 300:
                perc = 0.15
            else:
                perc = 0.12

            comparison.append((y[i] >= (yref_interp[i] - perc*yref_interp[i])) & (y[i] <= (yref_interp[i] + perc*yref_interp[i])))

        return np.array(comparison).astype(int)

    def filter_curve(self):
        """
        This function filters possible noise points that might get during the
        initial picking phase. It first separates the picked points into
        clusters using DBSCAN. Points classified as noise are removed right
        away. Then, the remaining clusters are filtered in the following
        manner:

        1 - Each cluster gets compared to the reference curve using
        :meth:`comparison_with_ref` and the cluster is discarded if any point
        lies outside the area around the reference curve.
        2 - A main cluster is identified and the remaining clusters are
        analyzed to check whether they present large jumps, in which case they
        are discarded.
        """
        if self.measure_type == "raw":
            eps = self.eps_raw
        elif self.measure_type == "clean":
            eps = self.eps_clean

        self.curve_filt = []
        x, y = self.curve[:, 0], self.curve[:, 1]
        X_train = np.array([self.minmax_normalization(np.log10(x)), self.minmax_normalization(y)]).T
        try:
            db = DBSCAN(eps=eps, min_samples=5).fit(X_train)
        except:
            self.curve_filt = []
            return

        clusters = np.unique(db.labels_[db.labels_ != -1])

        if len(clusters) == 0:
            return

        throw = []

        largest = -1
        max_largest_pts = 0

        # filtering possible outlier clusters that stray too far
        # away from the reference curve
        for cluster in clusters:
            x1 = x[db.labels_ == cluster]
            y1 = y[db.labels_ == cluster]
            if len(x1) > max_largest_pts:
                largest = cluster
                max_largest_pts = len(x1)
            if np.mean(self.comparison_with_ref(x1, y1)) != 1:
                throw.append(cluster)
        throw = np.array(throw)
        clusters = np.setdiff1d(clusters, throw)
        if largest in throw:
            return [], [], []

        # identifying the main cluster
        main_cluster, max_pts = -1, 0
        for cluster in clusters:
            x1 = x[db.labels_ == cluster]
            if len(x1) > max_pts:
                main_cluster = cluster
                max_pts = len(x1)

        main_ind = np.where(clusters == main_cluster)[0].item()

        # filtering possible clusters around the main cluster
        if len(clusters) > 1:
            steps = []
            for i in range(len(clusters) - 1):
                x1 = np.log10(x[db.labels_ == clusters[i]])[-1]
                y1 = y[db.labels_ == clusters[i]][-1]
                x2 = np.log10(x[db.labels_ == clusters[i+1]])[0]
                y2 = y[db.labels_ == clusters[i+1]][0]

                steps.append(np.sqrt((y2-y1)**2 + (x2-x1)**2))

            step_indices = np.arange(len(steps))
            lower_bound = step_indices[step_indices < main_ind]
            upper_bound = step_indices[step_indices >= main_ind]
            keep = []
            if len(lower_bound) > 0:
                for i in range(len(lower_bound)-1, -1, -1):
                    if steps[i] < 0.12:
                        keep.append(i)
                    else:
                        break
            if len(upper_bound) > 0:
                for i in range(main_ind, len(steps)):
                    if steps[i] < 0.12:
                        keep.append(i+1)
                    else:
                        break

            keep.append(main_ind)
            keep = sorted(keep)
            clusters = clusters[keep]

        cond = np.in1d(db.labels_, clusters)
        x_filt = x[cond]
        y_filt = y[cond]

        # filtering periods longer than allowed by the distance constraint
        max_T = x_filt <= self.max_period
        x_filt = x_filt[max_T]
        y_filt = y_filt[max_T]

        # throwing measurement away if the period range is too short
        if len(x_filt) > 0:
            if np.max(x_filt) - np.min(x_filt) < self.min_period_range:
                print("Period range too short, discarding measurement.")
                x_filt, y_filt = [], []

        self.curve_filt = np.array([x_filt, y_filt]).T

    def signal_to_noise_ratio(self):
        """
        Applies a frequency-dependent signal-to-noise ratio analysis (Bensen et
        al., 2007) to the waveform to identify and discard dispersion picks
        that fall below a certain threshold. The threshold is variable
        according to the period in the following way:

        1 - SNR = 10, if period < 80;
        2 - SNR = 7, if 80 <= period < 120;
        3 - SNR = 5, if period >= 120.

        References
        ----------
        Bensen, G. D., Ritzwoller, M. H., Barmin, M. P., Levshin, A. L., Lin, F.,
        Moschetti, M. P., ... and Yang, Y. (2007). Processing seismic ambient noise
        data to obtain reliable broad-band surface wave dispersion measurements.
        Geophysical journal international, 169(3), 1239-1260
        """

        vmin = self.vmin
        vmax = self.vmax

        from collections import Counter

        # calculating signal and noise time windows
        tmin_signal = self.metadata["dist"] / vmax
        tmax_signal = self.metadata["dist"] / vmin
        tmin_noise = tmax_signal + self.noise_window_offset
        tmax_noise = tmin_noise + self.noise_window_size

        # taking the position of the sample for each time
        tmin_signal_sample = tmin_signal / self.metadata["dt"]
        tmax_signal_sample = tmax_signal / self.metadata["dt"]
        tmin_noise_sample = tmin_noise / self.metadata["dt"]
        tmax_noise_sample = tmax_noise / self.metadata["dt"]

        # trying a smaller and closer to the signal window
        # if the defaults are not available
        if tmin_noise_sample > len(self.x):
            try:
                tmin_noise = tmax_signal + self.noise_window_offset/2
                tmax_noise = tmin_noise + self.noise_window_size/2

                # taking the position of the sample for each time
                tmin_noise_sample = tmin_noise / self.metadata["dt"]
                tmax_noise_sample = tmax_noise / self.metadata["dt"]
                if tmin_noise_sample > len(self.x):
                    raise ValueError
            except:
                print("Window size too small to calculate the noise window.")
                if self.keep_no_snr:
                    print("Continuing without calculating signal-to-noise ratio.")
                    return
                else:
                    print("Skipping measurement of file %s."%self.file)
                    self.curve_filt = np.array([[], []]).T
                    return

        # defining a set of frequencies around which to bandpass the data
        f = np.logspace(np.log10(1/self.Tmax), np.log10(1/self.Tmin), 40)
        grad = np.abs(np.gradient(f))
        snr = []

        for i in range(len(f)):
            x = np.copy(self.x)
            freqmin = f[i] - grad[i]
            freqmax = f[i] + grad[i]

            x_filt = bandpass(data=x, freqmin=freqmin, freqmax=freqmax, df=self.metadata["sampling_rate"])
            window_signal = (self.t >= tmin_signal_sample) & (self.t <= tmax_signal_sample)
            window_noise = (self.t >= tmin_noise_sample) & (self.t <= tmax_noise_sample)

            peak = np.max(x_filt[window_signal])
            noise_rms = np.std(x_filt[window_noise])
            snr.append(peak/noise_rms)

        snr = np.flip(np.array(snr))
        snr_T = np.flip(1/f)

        f = interp1d(x=snr_T, y=snr, fill_value='extrapolate')

        snr_interp = f(self.curve_filt[:, 0])

        # creating a signal to noise vector as described below
        snr_vec = np.zeros(len(self.curve_filt[:, 0]))
        for i in range(len(snr_vec)):
            if self.curve_filt[:, 0][i] < 80:
                snr_vec[i] = 10
            elif 80 <= self.curve_filt[:, 0][i] < 120:
                snr_vec[i] = 7
            else:
                snr_vec[i] = 5

        snr_condition = (snr_interp >= snr_vec).astype(int)
        crossings = np.diff(snr_condition)
        n_crossings = Counter(crossings)[1] + Counter(crossings)[-1]

        # rules for filtering dispersion curves
        # 1: 0-1 crossing, >= 51 % of data above threshold SNR
        # 2: 2 crossings, >= 69 % of data above threshold SNR
        # 3: 3 crossings, >= 75 % of data above threshold SNR
        # 4: signals with < 51 % points above SNR or > 4 crossings are completely discarded

        def get_main_subset_indices(arr, n_crossings):
            indices = []
            current = 0

            for i in range(n_crossings + 1):
                indices.append([])

            indices[current].append(0)

            for i in range(1, len(arr)):
                if arr[i] != arr[i-1]:
                    indices[current].append(i-1)
                    current += 1
                    indices[current].append(i)

            indices[-1].append(len(arr)-1)

            s = 0
            main = 0
            for n, pair in enumerate(indices):
                if pair[1] - pair[0] > s:
                   s = pair[1] - pair[0]
                   main = n

            return indices[main]

        if n_crossings == 0 and np.mean(snr_condition) == 1.0:
            x, y = self.curve_filt[:, 0], self.curve_filt[:, 1]
        elif n_crossings == 1 and np.mean(snr_condition) >= .51:
            ind_range = get_main_subset_indices(snr_condition, n_crossings)
            indices = np.arange(ind_range[0], ind_range[1]+1)
            x, y = self.curve_filt[indices, 0], self.curve_filt[indices, 1]
        elif n_crossings == 2 and np.mean(snr_condition) >= .69:
            ind_range = get_main_subset_indices(snr_condition, n_crossings)
            indices = np.arange(ind_range[0], ind_range[1]+1)
            x, y = self.curve_filt[indices, 0], self.curve_filt[indices, 1]
        elif n_crossings == 3 and np.mean(snr_condition) >= .75:
            ind_range = get_main_subset_indices(snr_condition, n_crossings)
            indices = np.arange(ind_range[0], ind_range[1]+1)
            x, y = self.curve_filt[indices, 0], self.curve_filt[indices, 1]
        elif n_crossings == 4 and np.mean(snr_condition) >= .85:
            ind_range = get_main_subset_indices(snr_condition, n_crossings)
            indices = np.arange(ind_range[0], ind_range[1]+1)
            x, y = self.curve_filt[indices, 0], self.curve_filt[indices, 1]
        else:
            x, y = [], []

        self.curve_filt = np.array([x, y]).T

    def save_curve(self, name):
        """
        Saves the dispersion curve in text format.

        Parameters
        ----------
        name: str
            Name of the output text file.
        """
        np.savetxt(fname=name, X=self.curve_filt, fmt="%.2f %.4f")

    def extract_curve(self):
        """
        Helper function that wraps together all the functions necessary to
        extract the dispersion curve from waveform data.
        """
        self.clusters()

        if len(self.labels) > 0:
            self.measure_trial_curve()
            if len(self.curve) == 0:
                print("Could not extract dispersion curve from file %s, skipping it."%self.fname)
                return
        else:
            print("A problem has occurred while dividing the spectrogram from file %s into clusters, skipping it."%self.fname)
            return

        self.filter_curve()

        if len(self.curve_filt) == 0:
            print("No points left after filtering dispersion curve from file %s, skipping"%self.fname)
            return

        if self.snr:
            self.signal_to_noise_ratio()
            if len(self.curve_filt) == 0:
                print("Dispersion curve below signal-to-noise ratio threshold, discarding measurement")
                return

        if self.measure_type == "raw":
            self.phase_matched_filter()

        output_file = ".".join(self.fname.split(".")[:-1]) + ".txt"
        self.save_curve("results/%s_measurements/%s"%(self.measure_type, output_file))
