"""Python script to plot experiments for main result."""
#%%
from experiments_causal.plot_experiment import get_results
from experiments_causal.plot_experiment_balanced import get_results as get_results_balanced
from experiments_causal.plot_experiment_causalml import get_results as get_results_causalml
from experiments_causal.plot_experiment_anticausal import get_results as get_results_anticausal
from experiments_causal.plot_experiment_causal_robust import get_results as get_results_causal_robust
from experiments_causal.plot_experiment_arguablycausal_robust import get_results as get_results_arguablycausal_robust
from experiments_causal.plot_experiment_causal_robust import dic_robust_number as dic_robust_number_causal
from experiments_causal.plot_experiment_arguablycausal_robust import dic_robust_number as dic_robust_number_arguablycausal
from experiments_causal.plot_config_colors import *
from experiments_causal.plot_config_tasks import dic_title
from scipy.spatial import ConvexHull
from paretoset import paretoset
import seaborn as sns
from matplotlib.legend_handler import HandlerBase
from matplotlib.ticker import FormatStrFormatter
import matplotlib.markers as mmark
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Set plot configurations
sns.set_context("paper")
sns.set_style("white")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 1200
list_mak = [
    mmark.MarkerStyle("s"),
    mmark.MarkerStyle("D"),
    mmark.MarkerStyle("o"),
    mmark.MarkerStyle("X"),
]
list_lab = ["All", "Arguably causal", "Causal", "Constant"]
list_color = [color_all, color_arguablycausal, color_causal, color_constant]
markers_causalml = {
                "irm": "v",
                "vrex": "^",
                "ib_irm": ">",
                "and_mask": "h",
                "causirl_mmd": "<",
                "causirl_coral": "<",}

class MarkerHandler(HandlerBase):
    def create_artists(
        self, legend, tup, xdescent, ydescent, width, height, fontsize, trans
    ):
        return [
            plt.Line2D(
                [width / 2],
                [height / 2.0],
                ls="",
                marker=tup[1],
                markersize=markersize,
                color=tup[0],
                transform=trans,
            )
        ]


# Define list of experiments to plot
experiments = [
    "brfss_diabetes",
    "acsunemployment",
    "acsincome",
    "mimic_extract_mort_hosp",
]

sns.set_style("white")
fig = plt.figure(figsize=[5.5, 7])
(subfig1, subfig2, subfig3, subfig4) = fig.subfigures(
    4, 1
)  # create 4x1 subfigures

subfigs = (subfig1, subfig2, subfig3, subfig4)


ax1 = subfig1.subplots(
    1, 2, gridspec_kw={"width_ratios": [0.5, 0.5], "top":0.85, "bottom":0.25}
)  # create 1x4 subplots on subfig1
ax2 = subfig2.subplots(
    1, 2, gridspec_kw={"width_ratios": [0.5, 0.5], "top":0.85, "bottom":0.25}
)  # create 1x4 subplots on subfig2
ax3 = subfig3.subplots(
    1, 2, gridspec_kw={"width_ratios": [0.5, 0.5], "top":0.85, "bottom":0.25}
)  # create 1x4 subplots on subfig2
ax4 = subfig4.subplots(
    1, 2, gridspec_kw={"width_ratios": [0.5, 0.5], "top":0.85, "bottom":0.25}
)  # create 1x4 subplots on subfig2
axes = (ax1, ax2, ax3, ax4)

for index, experiment_name in enumerate(experiments):
    subfig = subfigs[index]
    subfig.subplots_adjust(wspace=0.3)
    ax = axes[index]
    subfig.suptitle(dic_title[experiment_name])  # set suptitle for subfig1
    eval_all = get_results(experiment_name)
    eval_constant = eval_all[eval_all["features"] == "constant"]
    dic_shift = {}
    dic_shift_acc = {}

    ax[0].xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax[0].yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax[1].xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax[1].yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

    ax[0].set_xlabel(f"In-domain accuracy")
    ax[0].set_ylabel(f"Out-of-domain\naccuracy")
    #############################################################################
    # plot errorbars and shift gap for constant
    #############################################################################
    errors = ax[0].errorbar(
        x=eval_constant["id_test"],
        y=eval_constant["ood_test"],
        xerr=eval_constant["id_test_ub"] - eval_constant["id_test"],
        yerr=eval_constant["ood_test_ub"] - eval_constant["ood_test"],
        fmt="X",
        color=color_constant,
        ecolor=color_error,
        markersize=markersize,
        capsize=capsize,
        label="constant",
    )
    # get pareto set for shift vs accuracy
    shift_acc = eval_constant
    shift_acc["type"] = "constant"
    shift_acc["gap"] = shift_acc["id_test"] - shift_acc["ood_test"]
    dic_shift_acc["constant"] = shift_acc

    #############################################################################
    # plot errorbars and shift gap for causal features
    #############################################################################
    eval_plot = eval_all[eval_all["features"] == "causal"]
    eval_plot.sort_values("id_test", inplace=True)
    # Calculate the pareto set
    points = eval_plot[["id_test", "ood_test"]]
    mask = paretoset(points, sense=["max", "max"])
    points = points[mask]
    points = points[points["id_test"] >= (eval_constant["id_test"].values[0] - 0.01)]
    markers = eval_plot[mask]
    markers = markers[markers["id_test"] >= (eval_constant["id_test"].values[0] - 0.01)]
    errors = ax[0].errorbar(
        x=markers["id_test"],
        y=markers["ood_test"],
        xerr=markers["id_test_ub"] - markers["id_test"],
        yerr=markers["ood_test_ub"] - markers["ood_test"],
        fmt="o",
        color=color_causal,
        ecolor=color_error,
        markersize=markersize,
        capsize=capsize,
        label="causal",
    )
    # highlight bar
    shift = eval_plot[mask]
    shift = shift[shift["ood_test"] == shift["ood_test"].max()]
    shift["type"] = "causal"
    dic_shift["causal"] = shift
    # get pareto set for shift vs accuracy
    shift_acc = eval_plot[
        eval_plot["ood_test"] == eval_plot["ood_test"].max()
    ].drop_duplicates()
    shift_acc["type"] = "causal"
    shift_acc["gap"] = shift_acc["id_test"] - shift_acc["ood_test"]
    dic_shift_acc["causal"] = shift_acc
    #############################################################################
    # plot errorbars and shift gap for arguablycausal features
    #############################################################################
    if (eval_all["features"] == "arguablycausal").any():
        eval_plot = eval_all[eval_all["features"] == "arguablycausal"]
        eval_plot.sort_values("id_test", inplace=True)
        # Calculate the pareto set
        points = eval_plot[["id_test", "ood_test"]]
        mask = paretoset(points, sense=["max", "max"])
        points = points[mask]
        points = points[points["id_test"] >= (eval_constant["id_test"].values[0] - 0.01)]
        markers = eval_plot[mask]
        markers = markers[
            markers["id_test"] >= (eval_constant["id_test"].values[0] - 0.01)
        ]
        errors = ax[0].errorbar(
            x=markers["id_test"],
            y=markers["ood_test"],
            xerr=markers["id_test_ub"] - markers["id_test"],
            yerr=markers["ood_test_ub"] - markers["ood_test"],
            fmt="D",
            color=color_arguablycausal,
            ecolor=color_error,
            markersize=markersize,
            capsize=capsize,
            label="arguably\ncausal",
        )
        # highlight bar
        shift = eval_plot[mask]
        shift = shift[shift["ood_test"] == shift["ood_test"].max()]
        shift["type"] = "arguably\ncausal"
        dic_shift["arguablycausal"] = shift
        # get pareto set for shift vs accuracy
        shift_acc = eval_plot[
            eval_plot["ood_test"] == eval_plot["ood_test"].max()
        ].drop_duplicates()
        shift_acc["type"] = "arguablycausal"
        shift_acc["gap"] = shift_acc["id_test"] - shift_acc["ood_test"]
        dic_shift_acc["arguablycausal"] = shift_acc

    #############################################################################
    # plot errorbars and shift gap for all features
    #############################################################################
    eval_plot = eval_all[eval_all["features"] == "all"]
    eval_plot.sort_values("id_test", inplace=True)
    # Calculate the pareto set
    points = eval_plot[["id_test", "ood_test"]]
    mask = paretoset(points, sense=["max", "max"])
    points = points[mask]
    points = points[points["id_test"] >= (eval_constant["id_test"].values[0] - 0.01)]
    markers = eval_plot[mask]
    markers = markers[markers["id_test"] >= (eval_constant["id_test"].values[0] - 0.01)]
    errors = ax[0].errorbar(
        x=markers["id_test"],
        y=markers["ood_test"],
        xerr=markers["id_test_ub"] - markers["id_test"],
        yerr=markers["ood_test_ub"] - markers["ood_test"],
        fmt="s",
        color=color_all,
        ecolor=color_error,
        markersize=markersize,
        capsize=capsize,
        label="all",
    )
    # highlight bar
    shift = eval_plot[mask]
    shift = shift[shift["ood_test"] == shift["ood_test"].max()]
    shift["type"] = "all"
    dic_shift["all"] = shift
    # get pareto set for shift vs accuracy
    shift_acc = eval_plot[
        eval_plot["ood_test"] == eval_plot["ood_test"].max()
    ].drop_duplicates()
    shift_acc["type"] = "all"
    shift_acc["gap"] = shift_acc["id_test"] - shift_acc["ood_test"]
    dic_shift_acc["all"] = shift_acc

    #############################################################################
    # plot pareto dominated area for constant
    #############################################################################
    xmin, xmax = ax[0].set_xlim()
    ymin, ymax = ax[0].set_ylim()
    ax[0].plot(
        [xmin, eval_constant["id_test"].values[0]],
        [eval_constant["ood_test"].values[0], eval_constant["ood_test"].values[0]],
        color=color_constant,
        linestyle=(0, (1, 1)),
        linewidth=linewidth_bound,
    )
    ax[0].plot(
        [eval_constant["id_test"].values[0], eval_constant["id_test"].values[0]],
        [ymin, eval_constant["ood_test"].values[0]],
        color=color_constant,
        linestyle=(0, (1, 1)),
        linewidth=linewidth_bound,
    )
    ax[0].fill_between(
        [xmin, eval_constant["id_test"].values[0]],
        [ymin, ymin],
        [eval_constant["ood_test"].values[0], eval_constant["ood_test"].values[0]],
        color=color_constant,
        alpha=0.05,
    )

    #############################################################################
    # plot pareto dominated area for causal features
    #############################################################################
    eval_plot = eval_all[eval_all["features"] == "causal"]
    eval_plot.sort_values("id_test", inplace=True)
    # Calculate the pareto set
    points = eval_plot[["id_test", "ood_test"]]
    mask = paretoset(points, sense=["max", "max"])
    points = points[mask]
    points = points[points["id_test"] >= (eval_constant["id_test"].values[0] - 0.01)]
    markers = eval_plot[mask]
    markers = markers[markers["id_test"] >= (eval_constant["id_test"].values[0] - 0.01)]
    # get extra points for the plot
    new_row = pd.DataFrame(
        {
            "id_test": [xmin, max(points["id_test"])],
            "ood_test": [max(points["ood_test"]), ymin],
        },
    )
    points = pd.concat([points, new_row], ignore_index=True)
    points.sort_values("id_test", inplace=True)
    ax[0].plot(
        points["id_test"],
        points["ood_test"],
        color=color_causal,
        linestyle=(0, (1, 1)),
        linewidth=linewidth_bound,
    )
    new_row = pd.DataFrame(
        {"id_test": [xmin], "ood_test": [ymin]},
    )
    points = pd.concat([points, new_row], ignore_index=True)
    points = points.to_numpy()
    hull = ConvexHull(points)
    ax[0].fill(
        points[hull.vertices, 0], points[hull.vertices, 1], color=color_causal, alpha=0.05
    )

    #############################################################################
    # plot pareto dominated area for arguablycausal features
    #############################################################################
    if (eval_all["features"] == "arguablycausal").any():
        eval_plot = eval_all[eval_all["features"] == "arguablycausal"]
        eval_plot.sort_values("id_test", inplace=True)
        # Calculate the pareto set
        points = eval_plot[["id_test", "ood_test"]]
        mask = paretoset(points, sense=["max", "max"])
        points = points[mask]
        points = points[points["id_test"] >= (eval_constant["id_test"].values[0] - 0.01)]
        # get extra points for the plot
        new_row = pd.DataFrame(
            {
                "id_test": [xmin, max(points["id_test"])],
                "ood_test": [max(points["ood_test"]), ymin],
            },
        )
        points = pd.concat([points, new_row], ignore_index=True)
        points.sort_values("id_test", inplace=True)
        ax[0].plot(
            points["id_test"],
            points["ood_test"],
            color=color_arguablycausal,
            linestyle=(0, (1, 1)),
            linewidth=linewidth_bound,
        )
        new_row = pd.DataFrame(
            {"id_test": [xmin], "ood_test": [ymin]},
        )
        points = pd.concat([points, new_row], ignore_index=True)
        points = points.to_numpy()
        hull = ConvexHull(points)
        ax[0].fill(
            points[hull.vertices, 0],
            points[hull.vertices, 1],
            color=color_arguablycausal,
            alpha=0.05,
        )
    ax[0].margins(x=0,y=0)
    #############################################################################
    # plot pareto dominated area for all features
    #############################################################################
    eval_plot = eval_all[eval_all["features"] == "all"]
    eval_plot.sort_values("id_test", inplace=True)
    # Calculate the pareto set
    points = eval_plot[["id_test", "ood_test"]]
    mask = paretoset(points, sense=["max", "max"])
    points = points[mask]
    points = points[points["id_test"] >= (eval_constant["id_test"].values[0] - 0.01)]
    # get extra points for the plot
    new_row = pd.DataFrame(
        {
            "id_test": [xmin, max(points["id_test"])],
            "ood_test": [max(points["ood_test"]), ymin],
        },
    )
    points = pd.concat([points, new_row], ignore_index=True)
    points.sort_values("id_test", inplace=True)
    ax[0].plot(
        points["id_test"],
        points["ood_test"],
        color=color_all,
        linestyle=(0, (1, 1)),
        linewidth=linewidth_bound,
    )
    new_row = pd.DataFrame(
        {"id_test": [xmin], "ood_test": [ymin]},
    )
    points = pd.concat([points, new_row], ignore_index=True)
    points = points.to_numpy()
    hull = ConvexHull(points)
    ax[0].fill(
        points[hull.vertices, 0], points[hull.vertices, 1], color=color_all, alpha=0.05
    )

    #############################################################################
    # Add legend & diagonal, save plot
    #############################################################################
    # Plot the diagonal line
    start_lim = max(xmin, ymin)
    end_lim = min(xmax, ymax)
    ax[0].plot([start_lim, end_lim], [start_lim, end_lim], color="k")

    #############################################################################
    # Plot shift gap vs accuarcy
    #############################################################################
    if (eval_all["features"] == "arguablycausal").any():
        ax[1].set_xlabel("Shift gap")
        # ax[1].set_ylabel("Out-of-domain\naccuracy")
        shift_acc = pd.concat(dic_shift_acc.values(), ignore_index=True)
        markers = {
            "constant": "X",
            "causal": "o",
            "arguablycausal": "D",
            "all": "s",
        }
        for type, marker in markers.items():
            type_shift = shift_acc[shift_acc["type"] == type]
            type_shift["id_test_var"] = (
                (type_shift["id_test_ub"] - type_shift["id_test"])
            ) ** 2
            type_shift["ood_test_var"] = (
                (type_shift["ood_test_ub"] - type_shift["ood_test"])
            ) ** 2
            type_shift["gap_var"] = type_shift["id_test_var"] + type_shift["ood_test_var"]

            # Get markers
            ax[1].errorbar(
                x=-type_shift["gap"],
                y=type_shift["ood_test"],
                xerr=type_shift["gap_var"] ** 0.5,
                yerr=type_shift["ood_test_ub"] - type_shift["ood_test"],
                color=eval(f"color_{type}"),
                ecolor=color_error,
                fmt=marker,
                markersize=markersize,
                capsize=capsize,
                label="arguably\ncausal" if type == "arguablycausal" else f"{type}",
                zorder=3,
            )
        xmin, xmax = ax[1].get_xlim()
        ymin, ymax = ax[1].get_ylim()
        for type, marker in markers.items():
            type_shift = shift_acc[shift_acc["type"] == type]
            # Get 1 - shift gap
            type_shift["-gap"] = -type_shift["gap"]
            # Calculate the pareto set
            points = type_shift[["-gap", "ood_test"]]
            mask = paretoset(points, sense=["max", "max"])
            points = points[mask]
            # get extra points for the plot
            new_row = pd.DataFrame(
                {
                    "-gap": [xmin, max(points["-gap"])],
                    "ood_test": [max(points["ood_test"]), ymin],
                },
            )
            points = pd.concat([points, new_row], ignore_index=True)
            points.sort_values("-gap", inplace=True)
            ax[1].plot(
                points["-gap"],
                points["ood_test"],
                color=eval(f"color_{type}"),
                linestyle=(0, (1, 1)),
                linewidth=linewidth_bound,
            )
            new_row = pd.DataFrame(
                {"-gap": [xmin], "ood_test": [ymin]},
            )
            points = pd.concat([points, new_row], ignore_index=True)
            points = points.to_numpy()
            hull = ConvexHull(points)
            ax[1].fill(
                points[hull.vertices, 0],
                points[hull.vertices, 1],
                color=eval(f"color_{type}"),
                alpha=0.05,
            )
            ax[1].set_ylim((ymin,ymax))
            ax[1].margins(x=0,y=0)

list_mak_results = list_mak.copy()
list_mak_results.append("_")
list_lab_results = list_lab.copy()
list_lab_results.append("Diagonal")
list_color_results = list_color.copy()
list_color_results.append("black")
fig.legend(
    list(zip(list_color_results, list_mak_results)),
    list_lab_results,
    handler_map={tuple: MarkerHandler()},
    loc="upper center",
    bbox_to_anchor=(0.5, 0),
    fancybox=True,
    ncol=5,
)

fig.savefig(
    str(Path(__file__).parents[0] / f"plots_paper/plot_main_result.pdf"),
)

#%%

#############################################################################
# Next figure
#############################################################################

fig = plt.figure(figsize=[5.5, 1.5 * 2])

experiments = ["acsincome", "anes"]
subfig1, subfig2 = fig.subfigures(2, 1)  # create 4x1 subfigures

subfigs = (subfig1, subfig2)
ax1 = subfig1.subplots(
    1, 2, gridspec_kw={"width_ratios": [0.5, 0.5], "top":0.85, "bottom":0.25}
)  # create 1x4 subplots on subfig1
ax2 = subfig2.subplots(1, 2, gridspec_kw={"width_ratios": [0.5, 0.5], "top":0.85, "bottom":0.25})
axes = (ax1, ax2)

for index, experiment_name in enumerate(experiments):
    subfig = subfigs[index]
    subfig.subplots_adjust(wspace=0.3)
    ax = axes[index]
    subfig.suptitle(dic_title[experiment_name], fontsize=9)  # set suptitle for subfig1

    eval_all = get_results_causalml(experiment_name)
    eval_constant = eval_all[eval_all["features"] == "constant"]
    dic_shift = {}

    ax[0].xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax[0].yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax[1].xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax[1].yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

    ax[0].set_xlabel(f"In-domain accuracy")
    ax[0].set_ylabel(f"Out-of-domain\naccuracy")

    ##############################################################################
    # plot errorbars and shift gap for constant
    #############################################################################
    errors = ax[0].errorbar(
        x=eval_constant["id_test"],
        y=eval_constant["ood_test"],
        xerr=eval_constant["id_test_ub"] - eval_constant["id_test"],
        yerr=eval_constant["ood_test_ub"] - eval_constant["ood_test"],
        fmt="X",
        color=color_constant,
        ecolor=color_error,
        markersize=markersize,
        capsize=capsize,
        label="constant",
    )
    shift = eval_constant
    shift = shift[shift["ood_test"] == shift["ood_test"].max()]
    shift["type"] = "constant"
    dic_shift["constant"] = shift
    #############################################################################
    # plot errorbars and shift gap for all features
    #############################################################################
    eval_plot = eval_all[eval_all["features"] == "all"]
    eval_plot = eval_plot[
        (eval_plot["model"] != "irm")
        & (eval_plot["model"] != "vrex")
        & (eval_plot["model"] != "tableshift:irm")
        & (eval_plot["model"] != "tableshift:vrex")
        & (eval_plot["model"] != "ib_irm")
        & (eval_plot["model"] != "causirl_mmd")
        & (eval_plot["model"] != "causirl_coral")
        & (eval_plot["model"] != "and_mask")
    ]
    eval_plot.sort_values("id_test", inplace=True)
    # Calculate the pareto set
    points = eval_plot[["id_test", "ood_test"]]
    mask = paretoset(points, sense=["max", "max"])
    points = points[mask]
    points = points[points["id_test"] >= eval_constant["id_test"].values[0]]
    markers = eval_plot[mask]
    markers = markers[markers["id_test"] >= eval_constant["id_test"].values[0]]
    errors = ax[0].errorbar(
        x=markers["id_test"],
        y=markers["ood_test"],
        xerr=markers["id_test_ub"] - markers["id_test"],
        yerr=markers["ood_test_ub"] - markers["ood_test"],
        fmt="s",
        color=color_all,
        ecolor=color_error,
        markersize=markersize,
        capsize=capsize,
        label="all",
    )
    # highlight bar
    shift = markers
    shift = shift[shift["ood_test"] == shift["ood_test"].max()]
    shift["type"] = "all"
    dic_shift["all"] = shift

    #############################################################################
    # plot errorbars and shift gap for causal features
    #############################################################################
    eval_plot = eval_all[eval_all["features"] == "causal"]
    eval_plot = eval_plot[
        (eval_plot["model"] != "irm")
        & (eval_plot["model"] != "vrex")
        & (eval_plot["model"] != "tableshift:irm")
        & (eval_plot["model"] != "tableshift:vrex")
        & (eval_plot["model"] != "ib_irm")
        & (eval_plot["model"] != "causirl_mmd")
        & (eval_plot["model"] != "causirl_coral")
        & (eval_plot["model"] != "and_mask")
    ]
    eval_plot.sort_values("id_test", inplace=True)
    # Calculate the pareto set
    points = eval_plot[["id_test", "ood_test"]]
    mask = paretoset(points, sense=["max", "max"])
    points = points[mask]
    markers = eval_plot[mask]
    markers = markers[markers["id_test"] >= eval_constant["id_test"].values[0]]
    errors = ax[0].errorbar(
        x=markers["id_test"],
        y=markers["ood_test"],
        xerr=markers["id_test_ub"] - markers["id_test"],
        yerr=markers["ood_test_ub"] - markers["ood_test"],
        fmt="o",
        color=color_causal,
        ecolor=color_error,
        markersize=markersize,
        capsize=capsize,
        label="causal",
    )
    # highlight bar
    shift = markers
    shift = shift[shift["ood_test"] == shift["ood_test"].max()]
    shift["type"] = "causal"
    dic_shift["causal"] = shift

    #############################################################################
    # plot errorbars and shift gap for arguablycausal features
    #############################################################################
    if (eval_all["features"] == "arguablycausal").any():
        eval_plot = eval_all[eval_all["features"] == "arguablycausal"]
        eval_plot = eval_plot[
            (eval_plot["model"] != "irm")
            & (eval_plot["model"] != "vrex")
            & (eval_plot["model"] != "tableshift:irm")
            & (eval_plot["model"] != "tableshift:vrex")
            & (eval_plot["model"] != "ib_irm")
            & (eval_plot["model"] != "causirl_mmd")
            & (eval_plot["model"] != "causirl_coral")
            & (eval_plot["model"] != "and_mask")
        ]
        eval_plot.sort_values("id_test", inplace=True)
        # Calculate the pareto set
        points = eval_plot[["id_test", "ood_test"]]
        mask = paretoset(points, sense=["max", "max"])
        points = points[mask]
        points = points[
            points["id_test"] >= (eval_constant["id_test"].values[0] - 0.01)
        ]
        markers = eval_plot[mask]
        markers = markers[
            markers["id_test"] >= (eval_constant["id_test"].values[0] - 0.01)
        ]
        errors = ax[0].errorbar(
            x=markers["id_test"],
            y=markers["ood_test"],
            xerr=markers["id_test_ub"] - markers["id_test"],
            yerr=markers["ood_test_ub"] - markers["ood_test"],
            fmt="D",
            color=color_arguablycausal,
            ecolor=color_error,
            markersize=markersize,
            capsize=capsize,
            label="arguably\ncausal",
        )
        # highlight bar
        shift = markers
        shift = shift[shift["ood_test"] == shift["ood_test"].max()]
        shift["type"] = "arguablycausal"
        dic_shift["arguablycausal"] = shift

    #############################################################################
    # plot errorbars and shift gap for causal ml
    #############################################################################
    eval_plot = eval_all[eval_all["features"] == "all"]

    for causalml in ["irm", "vrex"]:
        eval_model = eval_plot[
            (eval_plot["model"] == causalml)
            | (eval_plot["model"] == f"tableshift:{causalml}")
        ]
        points = eval_model[["id_test", "ood_test"]]
        mask = paretoset(points, sense=["max", "max"])
        points = points[mask]
        points = points[
            points["id_test"] >= (eval_constant["id_test"].values[0] - 0.01)
        ]
        markers = eval_model[mask]
        markers = markers[
            markers["id_test"] >= (eval_constant["id_test"].values[0] - 0.01)
        ]
        errors = ax[0].errorbar(
            x=markers["id_test"],
            y=markers["ood_test"],
            xerr=markers["id_test_ub"] - markers["id_test"],
            yerr=markers["ood_test_ub"] - markers["ood_test"],
            fmt=markers_causalml[causalml],
            color=eval(f"color_{causalml}"),
            ecolor=color_error,
            markersize=markersize,
            capsize=capsize,
            label="causal ml",
        )
        # highlight bar

        shift = markers
        shift = shift[shift["ood_test"] == shift["ood_test"].max()]
        shift["type"] = causalml
        dic_shift[causalml] = shift

    for causalml in ["ib_irm", "causirl_mmd", "causirl_coral", "and_mask"]:
        eval_model = eval_plot[
            (eval_plot["model"] == causalml)
        ]
        points = eval_model[["id_test", "ood_test"]]
        mask = paretoset(points, sense=["max", "max"])
        points = points[mask]
        points = points[
            points["id_test"] >= (eval_constant["id_test"].values[0] - 0.01)
        ]
        markers = eval_model[mask]
        markers = markers[
            markers["id_test"] >= (eval_constant["id_test"].values[0] - 0.01)
        ]
        errors = ax[0].errorbar(
            x=markers["id_test"],
            y=markers["ood_test"],
            xerr=markers["id_test_ub"] - markers["id_test"],
            yerr=markers["ood_test_ub"] - markers["ood_test"],
            fmt=markers_causalml[causalml],
            color=eval(f"color_{causalml}"),
            ecolor=color_error,
            markersize=markersize,
            capsize=capsize,
            label="causal ml",
        )
        # highlight bar

        shift = markers
        shift = shift[shift["ood_test"] == shift["ood_test"].max()]
        shift["type"] = causalml
        dic_shift[causalml] = shift

    #############################################################################
    # plot pareto dominated area for constant
    #############################################################################
    xmin, xmax = ax[0].get_xlim()
    ymin, ymax = ax[0].get_ylim()
    ax[0].plot(
        [xmin, eval_constant["id_test"].values[0]],
        [eval_constant["ood_test"].values[0], eval_constant["ood_test"].values[0]],
        color=color_constant,
        linestyle=(0, (1, 1)),
        linewidth=linewidth_bound,
    )
    ax[0].plot(
        [eval_constant["id_test"].values[0], eval_constant["id_test"].values[0]],
        [ymin, eval_constant["ood_test"].values[0]],
        color=color_constant,
        linestyle=(0, (1, 1)),
        linewidth=linewidth_bound,
    )
    ax[0].fill_between(
        [xmin, eval_constant["id_test"].values[0]],
        [ymin, ymin],
        [eval_constant["ood_test"].values[0], eval_constant["ood_test"].values[0]],
        color=color_constant,
        alpha=0.05,
    )

    #############################################################################
    # plot pareto dominated area for all features
    #############################################################################
    eval_plot = eval_all[eval_all["features"] == "all"]
    eval_plot = eval_plot[
        (eval_plot["model"] != "irm")
        & (eval_plot["model"] != "vrex")
        & (eval_plot["model"] != "tableshift:irm")
        & (eval_plot["model"] != "tableshift:vrex")
        & (eval_plot["model"] != "ib_irm")
        & (eval_plot["model"] != "causirl_mmd")
        & (eval_plot["model"] != "causirl_coral")
        & (eval_plot["model"] != "and_mask")
    ]
    eval_plot.sort_values("id_test", inplace=True)
    # Calculate the pareto set
    points = eval_plot[["id_test", "ood_test"]]
    mask = paretoset(points, sense=["max", "max"])
    points = points[mask]
    points = points[points["id_test"] >= eval_constant["id_test"].values[0]]
    # get extra points for the plot
    new_row = pd.DataFrame(
        {
            "id_test": [xmin, max(points["id_test"])],
            "ood_test": [max(points["ood_test"]), ymin],
        },
    )
    points = pd.concat([points, new_row], ignore_index=True)
    points.sort_values("id_test", inplace=True)
    ax[0].plot(
        points["id_test"],
        points["ood_test"],
        color=color_all,
        linestyle=(0, (1, 1)),
        linewidth=linewidth_bound,
    )
    new_row = pd.DataFrame(
        {"id_test": [xmin], "ood_test": [ymin]},
    )
    points = pd.concat([points, new_row], ignore_index=True)
    points = points.to_numpy()
    hull = ConvexHull(points)
    ax[0].fill(
        points[hull.vertices, 0],
        points[hull.vertices, 1],
        color=color_all,
        alpha=0.05,
    )

    #############################################################################
    # plot pareto dominated area for causal features
    #############################################################################
    eval_plot = eval_all[eval_all["features"] == "causal"]
    eval_plot = eval_plot[
        (eval_plot["model"] != "irm")
        & (eval_plot["model"] != "vrex")
        & (eval_plot["model"] != "tableshift:irm")
        & (eval_plot["model"] != "tableshift:vrex")
        & (eval_plot["model"] != "ib_irm")
        & (eval_plot["model"] != "causirl_mmd")
        & (eval_plot["model"] != "causirl_coral")
        & (eval_plot["model"] != "and_mask")
    ]
    eval_plot.sort_values("id_test", inplace=True)
    # Calculate the pareto set
    points = eval_plot[["id_test", "ood_test"]]
    mask = paretoset(points, sense=["max", "max"])
    points = points[mask]
    points = points[points["id_test"] >= eval_constant["id_test"].values[0]]
    markers = eval_plot[mask]
    markers = markers[markers["id_test"] >= eval_constant["id_test"].values[0]]
    # get extra points for the plot
    new_row = pd.DataFrame(
        {
            "id_test": [xmin, max(points["id_test"])],
            "ood_test": [max(points["ood_test"]), ymin],
        },
    )
    points = pd.concat([points, new_row], ignore_index=True)
    points.sort_values("id_test", inplace=True)
    ax[0].plot(
        points["id_test"],
        points["ood_test"],
        color=color_causal,
        linestyle=(0, (1, 1)),
        linewidth=linewidth_bound,
    )
    new_row = pd.DataFrame(
        {"id_test": [xmin], "ood_test": [ymin]},
    )
    points = pd.concat([points, new_row], ignore_index=True)
    points = points.to_numpy()
    hull = ConvexHull(points)
    ax[0].fill(
        points[hull.vertices, 0],
        points[hull.vertices, 1],
        color=color_causal,
        alpha=0.05,
    )

    #############################################################################
    # plot pareto dominated area for arguablycausal features
    #############################################################################
    if (eval_all["features"] == "arguablycausal").any():
        eval_plot = eval_all[eval_all["features"] == "arguablycausal"]
        eval_plot = eval_plot[
            (eval_plot["model"] != "irm")
            & (eval_plot["model"] != "vrex")
            & (eval_plot["model"] != "tableshift:irm")
            & (eval_plot["model"] != "tableshift:vrex")
            & (eval_plot["model"] != "ib_irm")
            & (eval_plot["model"] != "causirl_mmd")
            & (eval_plot["model"] != "causirl_coral")
            & (eval_plot["model"] != "and_mask")
        ]
        eval_plot.sort_values("id_test", inplace=True)
        # Calculate the pareto set
        points = eval_plot[["id_test", "ood_test"]]
        mask = paretoset(points, sense=["max", "max"])
        points = points[mask]
        points = points[
            points["id_test"] >= (eval_constant["id_test"].values[0] - 0.01)
        ]
        # get extra points for the plot
        new_row = pd.DataFrame(
            {
                "id_test": [xmin, max(points["id_test"])],
                "ood_test": [max(points["ood_test"]), ymin],
            },
        )
        points = pd.concat([points, new_row], ignore_index=True)
        points.sort_values("id_test", inplace=True)
        ax[0].plot(
            points["id_test"],
            points["ood_test"],
            color=color_arguablycausal,
            linestyle=(0, (1, 1)),
            linewidth=linewidth_bound,
        )
        new_row = pd.DataFrame(
            {"id_test": [xmin], "ood_test": [ymin]},
        )
        points = pd.concat([points, new_row], ignore_index=True)
        points = points.to_numpy()
        hull = ConvexHull(points)
        ax[0].fill(
            points[hull.vertices, 0],
            points[hull.vertices, 1],
            color=color_arguablycausal,
            alpha=0.05,
        )

    #############################################################################
    # plot pareto dominated area for causalml
    #############################################################################
    # eval_plot = eval_all[eval_all["features"] == "all"]

    # for causalml in ["irm", "vrex"]:
    #     # Calculate the pareto set
    #     points = eval_plot[
    #         (eval_plot["model"] == causalml)
    #         | (eval_plot["model"] == f"tableshift:{causalml}")
    #     ][["id_test", "ood_test"]]
    #     mask = paretoset(points, sense=["max", "max"])
    #     points = points[mask]
    #     # get extra points for the plot
    #     new_row = pd.DataFrame(
    #         {
    #             "id_test": [xmin, max(points["id_test"])],
    #             "ood_test": [max(points["ood_test"]), ymin],
    #         },
    #     )
    #     points = pd.concat([points, new_row], ignore_index=True)
    #     points.sort_values("id_test", inplace=True)
    #     ax[0].plot(
    #         points["id_test"],
    #         points["ood_test"],
    #         color=eval(f"color_{causalml}"),
    #         linestyle=(0, (1, 1)),
    #         linewidth=linewidth_bound,
    #     )
    #     new_row = pd.DataFrame(
    #         {"id_test": [xmin], "ood_test": [ymin]},
    #     )
    #     points = pd.concat([points, new_row], ignore_index=True)
    #     points = points.to_numpy()
    #     hull = ConvexHull(points)
    #     ax[0].fill(
    #         points[hull.vertices, 0],
    #         points[hull.vertices, 1],
    #         color=eval(f"color_{causalml}"),
    #         alpha=0.05,
    #     )
    #############################################################################
    # Add legend & diagonal, save plot
    #############################################################################
    # Plot the diagonal line
    start_lim = max(xmin, ymin)
    end_lim = min(xmax, ymax)
    if experiment_name != "acsunemployment":
        ax[0].plot([start_lim, end_lim], [start_lim, end_lim], color="black")
    ax[0].margins(x=0,y=0)
    #############################################################################
    # Plot shift gap vs accuarcy
    #############################################################################
    if (eval_all["features"] == "arguablycausal").any():
        ax[1].set_xlabel("Shift gap")
        # ax[1].set_ylabel("Ood accuracy")
        shift_acc = pd.concat(dic_shift.values(), ignore_index=True)
        markers = {
            "constant": "X",
            "all": "s",
            "causal": "o",
            "arguablycausal": "D",
        }
        for type, marker in markers.items():
            type_shift = shift_acc[shift_acc["type"] == type]
            type_shift["gap"] = type_shift["id_test"] - type_shift["ood_test"]
            type_shift["id_test_var"] = (
                (type_shift["id_test_ub"] - type_shift["id_test"])
            ) ** 2
            type_shift["ood_test_var"] = (
                (type_shift["ood_test_ub"] - type_shift["ood_test"])
            ) ** 2
            type_shift["gap_var"] = (
                type_shift["id_test_var"] + type_shift["ood_test_var"]
            )

            # Get markers
            ax[1].errorbar(
                x=-type_shift["gap"],
                y=type_shift["ood_test"],
                xerr=type_shift["gap_var"] ** 0.5,
                yerr=type_shift["ood_test_ub"] - type_shift["ood_test"],
                color=eval(f"color_{type}"),
                ecolor=color_error,
                fmt=marker,
                markersize=markersize,
                capsize=capsize,
                label="arguably\ncausal" if type == "arguablycausal" else f"{type}",
                zorder=3,
            )
        for type, marker in markers_causalml.items():
            type_shift = shift_acc[shift_acc["type"] == type]
            type_shift["gap"] = type_shift["id_test"] - type_shift["ood_test"]
            type_shift["id_test_var"] = (
                (type_shift["id_test_ub"] - type_shift["id_test"])
            ) ** 2
            type_shift["ood_test_var"] = (
                (type_shift["ood_test_ub"] - type_shift["ood_test"])
            ) ** 2
            type_shift["gap_var"] = (
                type_shift["id_test_var"] + type_shift["ood_test_var"]
            )

            # Get markers
            ax[1].errorbar(
                x=-type_shift["gap"],
                y=type_shift["ood_test"],
                xerr=type_shift["gap_var"] ** 0.5,
                yerr=type_shift["ood_test_ub"] - type_shift["ood_test"],
                color=eval(f"color_{type}"),
                ecolor=color_error,
                fmt=marker,
                markersize=markersize,
                capsize=capsize,
                label="arguably\ncausal" if type == "arguablycausal" else f"{type}",
                zorder=3,
            )
        xmin, xmax = ax[1].get_xlim()
        ymin, ymax = ax[1].get_ylim()
        for type, marker in markers.items():
            type_shift = shift_acc[shift_acc["type"] == type]
            type_shift["gap"] = type_shift["id_test"] - type_shift["ood_test"]
            # Get 1 - shift gap
            type_shift["-gap"] = -type_shift["gap"]
            # Calculate the pareto set
            points = type_shift[["-gap", "ood_test"]]
            mask = paretoset(points, sense=["max", "max"])
            points = points[mask]
            # get extra points for the plot
            new_row = pd.DataFrame(
                {
                    "-gap": [xmin, max(points["-gap"])],
                    "ood_test": [max(points["ood_test"]), ymin],
                },
            )
            points = pd.concat([points, new_row], ignore_index=True)
            points.sort_values("-gap", inplace=True)
            ax[1].plot(
                points["-gap"],
                points["ood_test"],
                color=eval(f"color_{type}"),
                linestyle=(0, (1, 1)),
                linewidth=linewidth_bound,
            )
            new_row = pd.DataFrame(
                {"-gap": [xmin], "ood_test": [ymin]},
            )
            points = pd.concat([points, new_row], ignore_index=True)
            points = points.to_numpy()
            hull = ConvexHull(points)
            ax[1].fill(
                points[hull.vertices, 0],
                points[hull.vertices, 1],
                color=eval(f"color_{type}"),
                alpha=0.05,
            )
            ax[1].set_ylim((ymin,ymax))
            ax[1].margins(x=0,y=0)

    list_color_causalml = list_color.copy()
    # list_color_causalml.remove(color_constant)
    list_color_causalml.append(color_irm)
    list_color_causalml.append(color_vrex)
    list_color_causalml.append(color_ib_irm)
    list_color_causalml.append(color_causirl_mmd)
    list_color_causalml.append(color_and_mask)
    # list_color_causalml.append(color_constant)
    list_mak_causalml = list_mak.copy()
    # list_mak_causalml.remove(list_mak[-1])
    list_mak_causalml.append("v")
    list_mak_causalml.append("^")
    list_mak_causalml.append(">")
    list_mak_causalml.append("<")
    list_mak_causalml.append("h")
    # list_mak_causalml.append(list_mak[-1])
    list_lab_causalml = list_lab.copy()
    # list_lab_causalml.remove("Constant")
    list_lab_causalml.append("IRM")
    list_lab_causalml.append("REx")
    list_lab_causalml.append("IB-IRM")
    list_lab_causalml.append("Causal IRL")
    list_lab_causalml.append("AND-Mask")
    # list_lab_causalml.append("Constant")
    list_mak_causalml.append("_")
    list_color_causalml.append("Diagonal")
    list_color_causalml.append("black")
    fig.legend(
        list(zip(list_color_causalml, list_mak_causalml)),
        list_lab_causalml,
        handler_map={tuple: MarkerHandler()},
        loc="upper center",
        bbox_to_anchor=(0.5, -0.01),
        fancybox=True,
        ncol=5,
    )


fig.savefig(
    str(Path(__file__).parents[0] / f"plots_paper/plot_main_causalml.pdf"),
)
