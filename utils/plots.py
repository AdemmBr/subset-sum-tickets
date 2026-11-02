import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from . import theory

BLUE = "#2b6ca3"
RED = "#c8553d"
GREY = "#9a9a9a"


def use_style():
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["legend.frameon"] = False


def plot_transition(runs, exponents):
    fig, ax = plt.subplots(figsize=(7, 4))
    n = np.arange(1, runs.n_max + 1)
    colors = plt.cm.Blues(np.linspace(0.4, 1, len(exponents)))
    for k, color in zip(exponents, colors):
        eps = 2.0**-k
        ax.plot(n, (runs.radii <= eps).mean(axis=0), color=color, lw=2)
        # model is useless for small n
        model = [theory.coverage_given_weights(i, eps, runs.weights).mean() for i in n[7:]]
        ax.plot(n[7:], model, color=RED, lw=1, ls="--")
        mid = np.argmax((runs.radii <= eps).mean(axis=0) >= 0.5)
        ax.text(n[mid], 1.03, rf"$2^{{-{k}}}$", ha="center", fontsize=9)
    ax.plot([], [], color=BLUE, lw=2, label="measured")
    ax.plot([], [], color=RED, lw=1, ls="--", label="Poisson model")
    ax.set(xlabel="$n$", ylabel=r"$P(\mathrm{covered\ at\ }\varepsilon)$", ylim=(0, 1.1))
    ax.legend(loc="center left")


def plot_thresholds(runs, exponents):
    fig, ax = plt.subplots(figsize=(7, 4))
    exponents = np.asarray(exponents)
    for q, color in zip((0.5, 0.9, 0.99), (BLUE, "#5c97c6", RED)):
        y = np.array([runs.threshold(2.0**-k, q) for k in exponents])
        ok = np.isfinite(y)
        ax.plot(exponents[ok], y[ok], "o-", color=color, ms=4, label=f"{q:.0%} of draws")
    first_order = exponents + np.log2(np.log(2.0**exponents))
    ax.plot(exponents, first_order, color=GREY, ls="--",
            label=r"$\log_2(1/\varepsilon) + \log_2\ln(1/\varepsilon)$")
    ax.set(xlabel=r"$\log_2(1/\varepsilon)$", ylabel=r"$N(\varepsilon)$")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend()


def plot_offset(runs, k):
    fig, ax = plt.subplots(figsize=(7, 4))
    hits = runs.first_hit(2.0**-k)
    ok = np.isfinite(hits)
    offset = runs.standardised_offset(int(np.median(hits[ok])))
    jitter = np.random.default_rng(0).uniform(-0.2, 0.2, ok.sum())
    ax.scatter(offset[ok], hits[ok] + jitter, s=6, color=BLUE, alpha=0.5, lw=0)
    ax.set(xlabel=r"$|\sum_i x_i| \,/\, \sqrt{\sum_i x_i^2}$", ylabel=rf"$N(2^{{-{k}}})$")


def plot_ticket(rows, target_accuracy):
    fig, ax = plt.subplots(figsize=(7, 4))
    pool = [r["pool_size"] for r in rows]
    accuracy = [r["accuracy"] for r in rows]
    ax.plot(pool, accuracy, "o-", color=BLUE, ms=4, label="pruned random weights")
    ax.axhline(target_accuracy, color=GREY, ls="--", label="trained target")
    ax.set(xlabel="random weights per target weight", ylabel="test accuracy")
    ax.legend(loc="lower right")


def plot_worst_case(exponents, measured):
    fig, ax = plt.subplots(figsize=(7, 4))
    exponents = np.asarray(exponents)
    slope, intercept = np.polyfit(exponents, measured, 1)
    ax.plot(exponents, measured, "o", color=RED, ms=5, label=r"weights needed, $1 - \varepsilon$ of draws")
    ax.plot(exponents, slope * exponents + intercept, color=GREY, ls="--", label=f"slope {slope:.2f}")
    ax.set(xlabel=r"$\log_2(1/\varepsilon)$", ylabel=r"$N$")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend()
