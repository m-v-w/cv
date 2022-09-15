import numpy as np
from matplotlib import pyplot as plot

plots_lsv_call = [
    {
        "file": "data/spline_lsv_call_500.npz",
        "key": "arr_0",
        "title": "SMC"
    },
    {
        "file": "data/poly_lsv_call_500.npz",
        "key": "arr_1",
        "title": "Hermite polynomials"
    },
    {
        "file": "data/spline_lsv_call_500.npz",
        "key": "arr_1",
        "title": "Splines"
    },
    {
        "file": "data/nncv_lsv_call_500.npz",
        "key": "arr_1",
        "title": "NNCV"
    }
]
plots_lsv_call_50 = [
    {
        "file": "data/spline_lsv_call_50.npz",
        "key": "arr_0",
        "title": "SMC"
    },
    {
        "file": "data/poly_lsv_call_50.npz",
        "key": "arr_1",
        "title": "Hermite polynomials"
    },
    {
        "file": "data/spline_lsv_call_50.npz",
        "key": "arr_1",
        "title": "Splines"
    }
    #,
    #{
    #    "file": "data/nncv_lsv_call_50.npz",
    #    "key": "arr_1",
    #    "title": "NNCV"
    #}
]
plots_lsvrkhs_call_50 = [
    {
        "file": "data/spline_lsvrkhs_call_50.npz",
        "key": "arr_0",
        "title": "SMC"
    },
    {
        "file": "data/poly_lsvrkhs_call_50.npz",
        "key": "arr_1",
        "title": "Hermite polynomials"
    },
    {
        "file": "data/spline_lsvrkhs_call_50.npz",
        "key": "arr_1",
        "title": "Splines"
    },
    {
        "file": "data/mlmc_lsvrkhs_call_50.npz",
        "key": "arr_1",
        "title": "MLMC"
    }
 #   ,
 #   {
  #      "file": "data/nncv_lsv_call_50.npz",
  #      "key": "arr_1",
  #      "title": "NNCV"
  #  }
]
plots_normal_call = [
    {
        "file": "data/poly_normal_call_50.npz",
        "key": "arr_0",
        "title": "SMC"
    },
    {
        "file": "data/mlmc_normal_call_50.npz",
        "key": "arr_1",
        "title": "MLMC"
    },
    {
        "file": "data/poly_normal_call_50.npz",
        "key": "arr_1",
        "title": "Hermite polynomials"
    },
    {
        "file": "data/spline_normal_call_50.npz",
        "key": "arr_1",
        "title": "Splines"
    },
    {
        "file": "data/nncv_normal_call_50.npz",
        "key": "arr_1",
        "title": "NNCV"
    }
]
plots_normal_corr_call = [
    {
        "file": "data/smc_normal_corr_call.npz",
        "key": "arr_0",
        "title": "SMC"
    },
    {
        "file": "data/poly_normal_corr_call.npz",
        "key": "arr_1",
        "title": "Hermite polynomials"
    },
    {
        "file": "data/spline_normal_corr_call.npz",
        "key": "arr_1",
        "title": "Splines"
    },
    {
        "file": "data/nncv_normal_corr_call.npz",
        "key": "arr_1",
        "title": "NNCV"
    }
]
plots_normal_trigonometric = [
    {
        "file": "data/smc_normal_trigonometric_50.npz",
        "key": "arr_0",
        "title": "SMC"
    },
    {
        "file": "data/poly_normal_trigonometric_50.npz",
        "key": "arr_1",
        "title": "Hermite polynomials"
    },
    {
        "file": "data/spline_normal_trigonometric_50.npz",
        "key": "arr_1",
        "title": "Splines"
    },
    {
        "file": "data/nncv_normal_trigonometric_50.npz",
        "key": "arr_1",
        "title": "NNCV"
    }
]


def plot_box(list, name):
    data = []
    labels = []
    benchmark = None
    for t in list:
        np_data = np.load(t["file"])
        std = np.std(np_data[t["key"]])
        print('%s, %s: mean=%.6f std=%.6f' % (name, t["title"], np.mean(np_data[t["key"]]), std))
        if benchmark is None:
            benchmark = std
        if std <= benchmark:
            data.append(np_data[t["key"]])
            labels.append(t["title"])

    len = data.__len__()
    plot.boxplot(data)
    plot.xticks(range(1, len + 1), labels)
    plot.savefig("plots/" + name + ".eps", format='eps')
    plot.savefig("plots/" + name + ".png", format='png')
    plot.show()


plot_box(plots_normal_call, "normal_call")
plot_box(plots_lsv_call, "lsv_calls_500")
plot_box(plots_lsv_call_50, "lsv_calls_50")
plot_box(plots_lsvrkhs_call_50, "lsvrkhs_calls_50")
plot_box(plots_normal_trigonometric, "normal_trigonometric")
#plot_box(plots_normal_corr_call, "lsv_calls")

