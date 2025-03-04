import numpy as np
import os
from scipy.interpolate import interp1d
import glob
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt

def interpolate(files, period_range=np.arange(5, 251)):
    """
    Interpolates dispersion files into an array of periods. Extrapolations
    might be made for a maximum of 0.5 s (i.e., the maximum and minimum periods
    are rounded).

    Parameters
    ----------
    Files: list
        List of files containing dispersion measurements to be interpolated.

    period_range: list or numpy.ndarray
        Array of periods onto which to interpolate the measurements. Default is
        numpy.arange(5, 251).
    """
    if not os.path.isdir("interp"):
        os.mkdir("interp")

    print("######## INTERPOLATING ########")
    for file in files:
        fname = file.split(os.sep)[-1]
        T, gv = np.loadtxt(file, unpack=True)
        bounds = (int(np.round(np.min(T))), int(np.round(np.max(T))))
        cond = (period_range >= bounds[0]) & (period_range <= bounds[1])
        periods = period_range[cond]
        f = interp1d(x=T, y=gv, fill_value="extrapolate")
        interp_gv = f(periods)
        if np.mean(np.isinf(interp_gv).astype(int)) > 0:
            continue
        np.savetxt(fname="interp/%s"%fname, X=np.array([periods, interp_gv]).T, fmt="%d %.4f")
        print("Done with file %s"%fname)

def generate_input(sta_file="stations.pkl", ev_file="ev_table.csv"):
    """
    Generates input files for each period in the columns sorted as (evla, evlo,
    stla, stlo, U (m/s)). These input files are in the format required to use
    in tomography packages such as SeisLib (Magrini et al., 2022). This
    function should be run after the one that interpolates data.

    Parameters
    ----------
    sta_file: str
        Pickle file containing station information. The file is loaded as a
        dictionary where the keys are given by net.sta and the values are
        tuples containing (lat, long).

    ev_file: str
        Text file containing event information as:
        latitude,longitude,depth,datetime,magnitude,event_id.

    References
    ---------
    Magrini, F., Lauro, S., Kästle, E., & Boschi, L. (2022). Surface-wave
    tomography using SeisLib: a Python package for multiscale seismic imaging.
    Geophysical Journal International, 231(2), 1011-1030.
    """
    if not os.path.isdir("interp"):
        raise Exception("You must interpolate the dispersion curves first.")

    if not os.path.isdir("input"):
        os.mkdir("input")

    files = sorted(glob.glob("interp/*txt"))
    stations = pd.read_pickle(sta_file)
    events = pd.read_csv(ev_file)

    print("######## GENERATING INPUT ########")
    for file in files:
        fname = file.split(os.sep)[-1]
        print(fname)
        station = ".".join((fname.split(".")[0], fname.split(".")[1]))
        event_id = fname.split(".")[2] + ".a"
        evla, evlo = events[events["event_id"] == event_id]["latitude"].values[0], events[events["event_id"] == event_id]["longitude"].values[0]
        try:
            stla, stlo = stations[station][0], stations[station][1]
        except:
            continue
        T, gv = np.loadtxt(file, unpack=True)
        for n, period in enumerate(T):
            out = open("input/input_%s.00s.txt"%int(period), "a")
            out.write("%s %s %s %s %.4f\n"%(evla, evlo, stla, stlo, gv[n]*1000))
            out.close()

def average_curve(periods):
    """
    Given the input folder, computes an average dispersion curve at selected
    periods.

    Parameters
    ----------
    periods: list
        List of periods to compute the average curve.
    """
    for period in periods:
        if not os.path.isfile("input/input_%d.00s.txt"%period):
            raise Exception("Could not find file for period %s."%period)

    avg_vels = []
    stds = []

    for period in periods:
        _, _, _, _, gv = np.loadtxt("input/input_%d.00s.txt"%period, unpack=True)
        avg_vels.append(np.mean(gv))
        stds.append(np.std(gv))

    return np.array(avg_vels), np.array(stds)

def plot_average_curve(periods, averages, stds, multiplier=3):
    """
    Plots the average dispersion curve.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    T, gv = np.loadtxt("reference_curves/ak135_group_rayleigh.txt", unpack=True)
    gv = gv * 1000
    gv = gv[(T <= np.max(periods)) & (T >= np.min(periods))]
    T = T[(T <= np.max(periods)) & (T >= np.min(periods))]
    ax.plot(periods, averages)
    ax.plot(T, gv)
    ax.fill_between(periods, averages - multiplier*stds, averages + multiplier*stds, color="#888888", alpha=0.2)
    ax.set_ylim([2000, 4500])
    plt.show()

def detect_outlier_curve(files, ref, multp=3):
    """
    Detects dispersion curves that lie outside the area defined by a reference
    curve +/- mult * ref.

    Parameters
    ----------
    files: list
        List of files to detect the outliers.

    ref: tuple
        Tuple containing the reference curve.

    Returns
    -------
        List of outlier files.
    """
    Tref, vref, stdref = ref[0], ref[1], ref[2]

    f_v = interp1d(x=Tref, y=vref, fill_value="extrapolate")
    f_std = interp1d(x=Tref, y=stdref, fill_value="extrapolate")

    outliers = []

    for file in files:
        T, gv = np.loadtxt(file, unpack=True)
        gv = gv[(T >= np.min(Tref)) & (T <= np.max(Tref))]
        T = T[(T >= np.min(Tref)) & (T <= np.max(Tref))]
        vref_int = f_v(T)
        std_int = f_std(T)

        for i in range(len(T)):
            lower_bound = vref_int[i] - multp*std_int[i]
            upper_bound = vref_int[i] + multp*std_int[i]
            if gv[i] < lower_bound or gv[i] > upper_bound:
                outliers.append(file)
                break
    return outliers

def remove_outliers(files, outliers):
    """
    Moves the outlier files to a separate folder.
    """
    if not os.path.isdir("outliers"):
        os.mkdir("outliers")
    for file in files:
        if file in outliers:
            print("Removing %s."%file)
            os.system("mv %s outliers"%file)

if __name__ == "__main__":
    files = glob.glob("results/clean_measurements/*txt")
    interpolate(files)
    generate_input()

    # average curve with std
    periods = [15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 110, 120]
    avgs, stds = average_curve(periods=periods)
    ref = (periods, avgs/1000, stds/1000)

    # creating new folder containing only curves within avg +/- 3 stdsfiles
    files = glob.glob("interp/*txt")
    outs = detect_outlier_curve(files, ref)
    remove_outliers(files, outs)
    os.system("rm -r input")

    generate_input()
