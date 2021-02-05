"""
Example script showing how to run the trough code and view a
trough trajectory.
"""

import matplotlib.pyplot as plt

from mars_troughs import Trough

# Plot formatting
plt.rc("font", size=14, family="serif")


test_acc_params = [1e-8, 5e-9]
acc_model_number = 1
test_lag_params = [1, 1e-9]
lag_model_number = 1
errorbar = 100.0
tr = Trough(
    test_acc_params,
    test_lag_params,
    acc_model_number,
    lag_model_number,
    errorbar,
)


def get_fig_and_axes():
    times = tr.ins_times
    fig, ax = plt.subplots(ncols=2, nrows=3, sharex=True)
    ax[0, 0].set_ylabel("Ins(t)")
    ax[1, 0].set_ylabel("-A(t)")
    ax[2, 0].set_ylabel(r"$y(t)$")  # = \int_0^t{\rm d}t'\ A(t')$")
    ax[2, 0].set_xlabel(r"Lookback Time (yrs)")
    ax[2, 1].set_xlabel(r"Lookback Time (yrs)")
    ax[1, 0].ticklabel_format(
        style="scientific", scilimits=(0, 0)
    )  # , useMathText=True)
    ax[2, 0].ticklabel_format(
        axis="x", style="sci", scilimits=(0, 0)
    )  # , useMathText=True)
    ax[2, 1].ticklabel_format(
        axis="x", style="sci", scilimits=(0, 0)
    )  # , useMathText=True)
    ax[0, 0].plot(times, tr.get_insolation(times))
    ax[1, 0].plot(times, tr.get_accumulation(times))
    ax[2, 0].plot(times, tr.get_yt(times))

    ax[0, 1].plot(times, tr.get_lag_at_t(times))
    ax[1, 1].plot(times, tr.retreat_model_spline(times))
    ax[2, 1].plot(times, tr.get_xt(times))
    for i in range(3):
        ax[i, 1].yaxis.tick_right()
        ax[i, 1].yaxis.set_label_position("right")
    ax[0, 1].set_ylabel(r"Lag(t) (mm)")
    ax[1, 1].set_ylabel(r"$R(t)$ (km)")
    ax[2, 1].set_ylabel(
        r"$x(t)$"
    )  # = \int_0^t{\rm d}t' \frac{R(t') + A(t')\cos(\theta)}{\sin(\theta)}$")
    ax[2, 1].ticklabel_format(
        axis="y", style="sci", scilimits=(0, 0)
    )  # , useMathText=True)

    plt.subplots_adjust(wspace=0.1)

    return fig, ax


# Plot variables vs time
fig, ax = get_fig_and_axes()

# fig.savefig("example_plot.png", dpi=300, bbox_inches="tight")
plt.show()
plt.clf()


# Compare the trajectory with data
def get_trajectory_fig():
    times = tr.ins_times
    plt.plot(tr.get_xt(times), tr.get_yt(times))
    plt.errorbar(tr.xdata, tr.ydata, yerr=tr.errorbar, c="k", marker=".", ls="")
    xn, yn = tr.get_nearest_points()
    plt.plot(xn, yn, ls="", marker="o", c="r")
    return plt.gcf()


_ = get_trajectory_fig()
plt.show()
