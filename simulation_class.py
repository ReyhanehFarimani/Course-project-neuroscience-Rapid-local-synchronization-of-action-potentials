import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from cycler import cycler
import warnings

"""Set the defaults for your plots."""
# line cyclers adapted to colourblind people
line_cycler = cycler(
    color=["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]
) + cycler(linestyle=["-", "--", "-.", ":", "-", "--", "-."])
marker_cycler = (
    cycler(
        color=[
            "#E69F00",
            "#56B4E9",
            "#009E73",
            "#0072B2",
            "#D55E00",
            "#CC79A7",
            "#F0E442",
        ]
    )
    + cycler(linestyle=["none", "none", "none", "none", "none", "none", "none"])
    + cycler(marker=["4", "2", "3", "1", "+", "x", "."])
)
plt.rc("figure", figsize=(10, 5))  # Image size
plt.rc("axes", prop_cycle=line_cycler)
plt.rc("text", usetex=True)
plt.rc(
    "text.latex",
    preamble=r"\usepackage{newpxtext}\usepackage{newpxmath}\usepackage{commath}\usepackage{mathtools}",
)
plt.rc("font", family="serif", size=12)
plt.rc("savefig", dpi=200)
plt.rc("legend", loc="best", fontsize="large", fancybox=False, framealpha=0.5)
plt.rc("lines", linewidth=2, markersize=5, markeredgewidth=2.5)
plt.rc("axes", prop_cycle=line_cycler)


# plt.rcParams.update({'font.size': 20, 'figsize':(8,6)})
SMALL_SIZE = 12
MEDIUM_SIZE = 15
BIGGER_SIZE = 18

plt.rc("axes", titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=MEDIUM_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

warnings.filterwarnings("ignore")


class simulation:
    def __init__(self, L, simulation_time, alpha, I_ext, delta_t):
        self.L = L
        self.u = np.random.random((L, L))
        self.u_dot = np.zeros((L, L))
        self.I = np.zeros((L, L)) + I_ext
        self.delta_t = delta_t
        self.simulation_time = int(simulation_time / delta_t)
        self.n_sync = np.zeros((self.simulation_time))
        self.E = np.zeros((self.simulation_time))
        self.alpha = alpha

    def _update_method_A(self, n=0):
        self.u_dot = -self.u + self.I
        self.u += self.u_dot * self.delta_t
        while self.u.max() > 1:
            n += 1
            i, j = np.unravel_index(self.u.argmax(), self.u.shape)
            self.u[i, j] -= 1
            self.u[i - 1, j] += self.alpha
            self.u[i, j - 1] += self.alpha
            self.u[(i + 1) % self.L, j] += self.alpha
            self.u[i, (j + 1) % self.L] += self.alpha
        return n

    def _update_method_B(self, n=0):
        self.u_dot = -self.u + self.I
        self.u += self.u_dot * self.delta_t
        while self.u.max() > 1:
            n += 1
            i, j = np.unravel_index(self.u.argmax(), self.u.shape)
            self.u[i, j] = 0
            self.u[i - 1, j] += self.alpha
            self.u[i, j - 1] += self.alpha
            self.u[(i + 1) % self.L, j] += self.alpha
            self.u[i, (j + 1) % self.L] += self.alpha
        return n

    def _update_method_C(self, n=0):
        self.u_dot = self.I
        self.u += self.u_dot * self.delta_t
        while self.u.max() > 1:
            n += 1
            i, j = np.unravel_index(self.u.argmax(), self.u.shape)
            self.u[i, j] -= 1
            self.u[i - 1, j] += self.alpha
            self.u[i, j - 1] += self.alpha
            self.u[(i + 1) % self.L, j] += self.alpha
            self.u[i, (j + 1) % self.L] += self.alpha
        return n

    def _update_method_D(self, n=0):
        self.u_dot = self.I
        self.u += self.u_dot * self.delta_t
        while self.u.max() > 1:
            n += 1
            i, j = np.unravel_index(self.u.argmax(), self.u.shape)
            self.u[i, j] = 0
            self.u[i - 1, j] += self.alpha
            self.u[i, j - 1] += self.alpha
            self.u[(i + 1) % self.L, j] += self.alpha
            self.u[i, (j + 1) % self.L] += self.alpha
        return n

    def _update_method_E(self, n=0):
        self.u_dot = -self.u + self.I
        self.u += self.u_dot * self.delta_t
        while self.u.max() > 1:
            n += 1
            i, j = np.unravel_index(self.u.argmax(), self.u.shape)
            self.u[i, j] -= 1
            self.u[i - 1, j] += self.alpha
            self.u[i, j - 1] += self.alpha
            if i != self.L:
                self.u[(i + 1), j] += self.alpha
            if j != self.L:
                self.u[i, (j + 1)] += self.alpha
        return n

    def render(self, enum, name):
        if enum == 0:
            for t in range(self.simulation_time):
                n = self._update_method_A()
                np.save(
                    "methodA_"
                    + str(self.alpha)
                    + "_"
                    + str(self.I)
                    + "_"
                    + str(t)
                    + ".npy",
                    self.u,
                )
                self.n_sync[t] = n
                self.E[t] = -np.sum(self.u)
            plt.title("number of syncronized neurons in model A:")
            plt.xlabel("time({0})".format(self.delta_t))
            plt.ylabel("Nsync neurons")
            plt.plot(self.n_sync)
            plt.savefig(name + "A_n.jpg")
            plt.show()
            plt.title("Time evolution of the Lyapunov function E in model A:")
            plt.xlabel("time({0})".format(self.delta_t))
            plt.ylabel("E")
            plt.plot(self.E)
            plt.savefig(name + "A_E.jpg")
            plt.show()
        if enum == 1:
            for t in range(self.simulation_time):
                n = self._update_method_B()
                self.n_sync[t] = n
                self.E[t] = -np.sum(self.u)
            plt.title("number of syncronized neurons in model B:")
            plt.xlabel("time({0})".format(self.delta_t))
            plt.ylabel("Nsync neurons")
            plt.plot(self.n_sync)
            plt.savefig(name + "B_n.jpg")
            plt.show()
            """
            plt.title("Time evolution of the Lyapunov function E in model B:")
            plt.xlabel("time({0})".format(self.delta_t))
            plt.ylabel("E")
            plt.plot(self.E)
            plt.show()
            """
        if enum == 2:
            for t in range(self.simulation_time):
                n = self._update_method_C()
                self.n_sync[t] = n
                self.E[t] = -np.sum(self.u)
            plt.title("number of syncronized neurons in model C:")
            plt.xlabel("time({0})".format(self.delta_t))
            plt.ylabel("Nsync neurons")
            plt.plot(self.n_sync)
            plt.savefig(name + "C_n.jpg")
            plt.show()
            plt.title("Time evolution of the Lyapunov function E in model C:")
            plt.xlabel("time({0})".format(self.delta_t))
            plt.ylabel("E")
            plt.plot(self.E)
            plt.savefig(name + "C_E.jpg")
            plt.show()
        if enum == 3:
            for t in range(self.simulation_time):
                n = self._update_method_D()
                self.n_sync[t] = n
                self.E[t] = -np.sum(self.u)
            plt.title("number of syncronized neurons in model D:")
            plt.xlabel("time({0})".format(self.delta_t))
            plt.ylabel("Nsync neurons")
            plt.plot(self.n_sync)
            plt.savefig(name + "n_D.jpg")
            plt.show()
            """
            plt.title("Time evolution of the Lyapunov function E in model D:")
            plt.xlabel("time({0})".format(self.delta_t))
            plt.ylabel("E")
            plt.plot(self.E)
            plt.show()
            """

    def dist_n_sync(self, start, end, name):
        d = self.n_sync[start:end]
        plt.hist(d, bins=np.linspace(1, 19))
        plt.title("syncronised neuron distribuation")
        plt.savefig(name + ".jpg")
        plt.show()

    def periodic_behavior(self, time, period):
        x = self.E[time:]

        # for local maxima
        (f,) = argrelextrema(x, np.greater)
        for t in period:
            g = []
            l = np.size(f)
            for i in range(l):
                g.append(f[(i + t) % l])
            g = np.array(g)
            plt.plot(x[g], x[f])
            plt.plot(x[f], x[f], "-r")
            plt.xlabel("E[i + " + str(t) + "]")
            plt.ylabel("E[i]")
            plt.savefig("test.jpg")
            plt.show()

    def amplitude(self, time, name):
        x = self.E[time:]
        y1 = x.max()
        y2 = x.min()
        amplitude = abs(y2 - y1)

        plt.axhline(y=y1, color="g", label="amplitude = {}".format(round(amplitude, 3)))
        plt.axhline(y=y2, color="g")
        plt.plot(self.E)
        plt.legend()
        plt.savefig(name + ".jpg")
        plt.show()
