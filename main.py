import glob
import numpy as np
import os
from fastai.vision.all import *
from scipy.signal import find_peaks
from obspy import read
from picker import DispersionPicker
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from plotting import save_figure
from time import time
import argparse

def initialize_folders(root, classify, residual=False):
    """
    Creates folders to save results.
    """
    if not os.path.isdir("results"):
        os.mkdir("results")
        os.mkdir("results/raw_measurements")
        os.mkdir("results/clean_measurements")
        os.mkdir("results/clean_waveforms")
        os.mkdir("results/figures")
        if residual:
            os.mkdir("results/residual_waveforms")

    if classify:
        if not os.path.isdir("%s/Yes"%root):
            os.mkdir("%s/Yes"%root)
            os.mkdir("%s/No"%root)

def save_image_class(X, fname):
    """
    Saves images to use as input for the classifier.
    """
    x = X[:, 0]
    y = X[:, 1]
    z = X[:, 2]

    peaks, _ = find_peaks(z)
    xmax = x[peaks]
    ymax = y[peaks]

    x = np.unique(x)[::-1]
    y = np.unique(y)[::-1]
    xmesh, ymesh = np.meshgrid(x, y)
    zmesh = z.reshape(len(x), len(y)).T

    plt.subplots(figsize=(5, 3))

    try:
        plt.xscale("log")
        plt.ylim([2.0, 5.0])
        plt.contourf(xmesh, ymesh, zmesh, levels=40, cmap="jet")
        plt.scatter(xmax, ymax, c='black', s=6)
        plt.xticks([])
        plt.yticks([])
        plt.minorticks_off()
        plt.savefig(fname)
        plt.clf()
        plt.close()
    except:
        plt.clf()
        plt.close()
        return

def classify_and_extract_curves(files, learner):
    """
    Function to load a list of waveforms, classify them, and pick their respective
    dispersion curves if the quality is good.

    Parameters
    ----------

    files: list
        List of files to process.

    learner: fastai.learner.Learner
        Learner object containing the classifier.
    """
    for file in files:
        print(file)
        update_done_file(file)

        fname = file.split(os.sep)[-1]
        root = os.path.dirname(file)
        img_name = file[:-4] + ".png"
        dp = DispersionPicker(file)

        dp.load_file()

        if dp.x is None or dp.metadata is None:
            os.system("mv %s %s/No"%(file, root))
            print("Could not read file %s, moving it to No directory."%file)
            continue

        dp.FTAN()

        if len(dp.A) == 0:
            os.system("mv %s %s/No"%(file, root))
            print("Could not perform Frequency-Time Analysis on file %s, skipping"%file)
            continue

        try:
            save_image_class(np.array([dp.T, dp.Vg, dp.A]).T, img_name)
            pred = learner.predict(img_name)
            score = round(pred[-1][1].item(), 4)
        except:
            continue
        if score > 0.8:
            dp.extract_curve()
            os.system("mv %s %s/Yes"%(file, root))
            os.system("mv %s %s/Yes"%(img_name, root))
        else:
            os.system("mv %s %s/No"%(file, root))
            os.system("mv %s %s/No"%(img_name, root))

def extract_curves(files, tp="raw"):
    """
    Function to load a list of waveforms and pick their respective dispersion
    curves if the quality is good.

    Parameters
    ----------

    files: list
        List of files to process.

    tp: str, default = raw
        Type of measurement. Should be `raw` or `clean`.
    """
    for file in files:
        print(file)
        update_done_file(file)

        if tp == "raw":
            dp = DispersionPicker(file)
        elif tp == "clean":
            dp = DispersionPicker(file, measure_type="clean", snr=False)
        else:
            print("Type must be raw or clean.")
            break

        dp.load_file()

        if dp.x is None or dp.metadata is None:
            os.system("mv %s %s/No"%(file, root))
            print("Could not read file %s, moving it to No directory."%file)
            continue

        dp.FTAN()

        if len(dp.A) == 0:
            os.system("mv %s %s/No"%(file, root))
            print("Could not perform Frequency-Time Analysis on file %s, skipping"%file)
            continue

        dp.extract_curve()

def post_processing():
    """
    Removes possible problematic curves based on the difference between raw and
    clean curves.
    """
    os.chdir("results")
    files = [x for x in os.listdir("clean_measurements") if x.endswith(".txt")]
    for file in files:
        T0, gv0 = np.loadtxt("raw_measurements/%s"%file, unpack=True)
        T, gv = np.loadtxt("clean_measurements/%s"%file, unpack=True)
        # Removing problematic clean measurements
        if np.max(T) > np.max(T0) or np.min(T) < np.min(T0):
            print("Clean measurements should not have a broader range than raw ones, removing %s"%file)
            os.system("rm clean_measurements/%s"%file)
            continue
        # We will remove observations where the mean absolute difference between raw and clean measurements differ by more than 0.03 km/s
        f = interp1d(y=gv0, x=T0)
        y2 = f(T)
        diff = np.mean(np.abs(y2-gv))
        if diff > 0.03:
            print("The mean difference between raw and clean curves exceeds 0.03 km/s, removing %s"%file)
            os.system("rm clean_measurements/%s"%file)
            continue

        # sometimes a flat, glitchy, horizontal non-dispersive curve can bypass
        # the classification, but we can remove such measurements that
        # barely have any variation
        grad = np.gradient(gv, T)
        grad = grad[~np.isnan(grad)]
        if np.mean(np.abs(grad)) <= 0.002:
            print("Flat, (probably) non-dispersive curve detected in %s, removing."%file)
            os.system("rm clean_measurements/%s"%file)

    os.chdir("..")

def save_figures(root):
    """
    Saves the figures related to each successful measurement.

    Parameters
    ----------
    root: str
        Path of the root to the SAC files.
    """
    os.chdir("results")
    files = [x for x in os.listdir("clean_measurements") if x.endswith(".txt")]
    for file in files:
        print(file)
        fname = file.split(os.sep)[-1]
        sac_file = root + os.sep + ".".join(fname.split(".")[:-1]) + ".sac"
        T0, gv0 = np.loadtxt("raw_measurements/%s"%file, unpack=True)
        T, gv = np.loadtxt("clean_measurements/%s"%file, unpack=True)

        Tmin, Tmax = int(np.min(T0) - 5), int(np.max(T0) + 20)
        Tmin = np.max([Tmin, 5])
        dp = DispersionPicker(sac_file, Tmin=Tmin, Tmax=Tmax)
        dp.load_file()
        dp.FTAN()
        output_path = "figures" + os.sep + ".".join(dp.fname.split(".")[:-1]) + ".png"
        save_figure(np.array([dp.T, dp.Vg, dp.A]).T, T0, gv0, T, gv, output_path)
    os.chdir("..")

def get_files(path, tp="raw"):
    """
    Creates a list of waveforms within the `path` directory.

    Parameters
    ----------
    path: str
        Path to directory containing waveform data.

    tp: str, default = raw
        Type of measurement. Should be `raw` or `clean`.
    """
    files = glob.glob("%s/*%s"%(path, "sac" if tp == "raw" else "sacs"))
    done = open("DONE.txt", "a")
    done_values = []
    with open("DONE.txt", "r") as infile:
        for line in infile:
            done_values.append(line.strip())
    done_values = done_values[:-1]
    files = [x for x in files if x not in done_values]

    return files

def update_done_file(file):
    """
    Updates the list of files that were measured.

    Parameters
    ----------
    file: str
        Name of the file.
    """
    done = open("DONE.txt", "a")
    done.write(file+'\n')
    done.close()

if __name__ == "__main__":
    t1 = time()

    parser = argparse.ArgumentParser(description="Extract group velocity dispersion curves from SAC waveforms.")
    parser.add_argument("-f", type=str, metavar="", required=True, help="Path to data.")
    parser.add_argument("--classify", action=argparse.BooleanOptionalAction, default=True, help="Whether or not to classify waveforms.")
    parser.add_argument("--residual", action=argparse.BooleanOptionalAction, default=False, help="Whether or not to save residual waveforms.")

    args = parser.parse_args()

    classify = args.classify

    files = get_files(args.f)
    root = os.path.dirname(files[0])

    initialize_folders(root, classify, residual=args.residual)
    learner = load_learner("classifier.pkl")

    if args.classify:
        print("###### STARTING CLASSIFICATION AND RAW MEASUREMENTS ######")
    else:
        print("###### STARTING RAW MEASUREMENTS ######")

    if classify:
        classify_and_extract_curves(files, learner)
    else:
        extract_curves(files)

    os.system("rm DONE.txt")

    print("###### STARTING CLEAN MEASUREMENTS ######")

    files = get_files("results/clean_waveforms", tp="clean")
    extract_curves(files, tp="clean")

    print("###### REMOVING POSSIBLE BAD MEASUREMENTS ######")
    post_processing()

    print("###### SAVING FIGURES ######")
    root = os.path.abspath(root)

    if classify:
        root = os.path.join(root, "Yes")

    save_figures(root)

    os.system("rm DONE.txt")

t2 = time()
print("Time elapsed: %.3f s"%(t2 - t1))
