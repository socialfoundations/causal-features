"""Python script to plot experiments for introduction."""
from experiments_causal.plot_experiment import get_results
from experiments_causal.plot_config_colors import *
from experiments_causal.plot_config_tasks import dic_title
import seaborn as sns
from matplotlib.legend_handler import HandlerBase
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
    "acsfoodstamps",
    "acsincome",
    "acspubcov",
    "acsunemployment",
    "anes",
    "assistments",
    "brfss_blood_pressure",
    "brfss_diabetes",
    "college_scorecard",
    "diabetes_readmission",
    "mimic_extract_mort_hosp",
    "mimic_extract_los_3",
    "nhanes_lead",
    "physionet",
    "meps",
    "sipp",
]


eval_experiments = pd.DataFrame()
for index, experiment_name in enumerate(experiments):
    eval_all = get_results(experiment_name)
    eval_all["task"] = dic_title[experiment_name]

    eval_plot = pd.DataFrame()
    for set in eval_all["features"].unique():
        eval_feature = eval_all[eval_all["features"] == set]
        eval_feature = eval_feature[
            eval_feature["ood_test"] == eval_feature["ood_test"].max()
        ]
        eval_feature.drop_duplicates(inplace=True)
        eval_plot = pd.concat([eval_plot, eval_feature])
    eval_experiments = pd.concat([eval_experiments, eval_plot])
    dic_shift = {}
    dic_shift_acc = {}


fig = plt.figure(figsize=(6.75, 1.5))
ax = fig.subplots(
    1, 2, gridspec_kw={"width_ratios": [0.5, 0.5], "wspace": 0.3}
)  # create 1x4 subplots on subfig1

ax[0].set_xlabel(f"Tasks")
ax[0].set_ylabel(f"Out-of-domain accuracy")

#############################################################################
# plot ood accuracy
#############################################################################
markers = {"constant": "X", "all": "s", "causal": "o", "arguablycausal": "D"}

sets = list(eval_experiments["features"].unique())
sets.sort()

for index, set in enumerate(sets):
    eval_plot_features = eval_experiments[eval_experiments["features"] == set]
    eval_plot_features = eval_plot_features.sort_values("ood_test")
    ax[0].errorbar(
        x=eval_plot_features["task"],
        y=eval_plot_features["ood_test"],
        yerr=eval_plot_features["ood_test_ub"] - eval_plot_features["ood_test"],
        color=eval(f"color_{set}"),
        ecolor=color_error,
        fmt=markers[set],
        markersize=markersize,
        capsize=capsize,
        label=set.capitalize() if set != "arguablycausal" else "Arguably causal",
        zorder=len(sets) - index,
    )
    # get pareto set for shift vs accuracy
    shift_acc = eval_plot_features
    shift_acc["type"] = set
    shift_acc["gap"] = shift_acc["ood_test"] - shift_acc["id_test"]
    shift_acc["id_test_var"] = ((shift_acc["id_test_ub"] - shift_acc["id_test"])) ** 2
    shift_acc["ood_test_var"] = ((shift_acc["ood_test_ub"] - shift_acc["ood_test"])) ** 2
    shift_acc["gap_var"] = shift_acc["id_test_var"] + shift_acc["ood_test_var"]
    dic_shift_acc[set] = shift_acc

ax[0].tick_params(axis="x", labelrotation=90)
ax[0].set_ylim(top=1.0)
ax[0].grid(axis="y")


ax[1].set_xlabel(f"Tasks")
ax[1].set_ylabel(f"Shift gap (higher is better)")
#############################################################################
# plot shift gap
#############################################################################
shift_acc = pd.concat(dic_shift_acc.values(), ignore_index=True)
sets = list(eval_experiments["features"].unique())
sets.sort()

for index, set in enumerate(sets):
    shift_acc_plot = shift_acc[shift_acc["features"] == set]
    shift_acc_plot = shift_acc_plot.sort_values("ood_test")
    ax[1].errorbar(
        x=shift_acc_plot["task"],
        y=shift_acc_plot["gap"],
        yerr=shift_acc_plot["gap_var"] ** 0.5,
        color=eval(f"color_{set}"),
        ecolor=color_error,
        fmt=markers[set],
        markersize=markersize,
        capsize=capsize,
        label=set.capitalize() if set != "arguablycausal" else "Arguably causal",
        zorder=len(sets) - index,
    )

ax[1].axhline(
    y=0,
    color="black",
    linestyle="--",
)
ax[1].tick_params(axis="x", labelrotation=90)

ax[1].grid(axis="y")

list_mak.append("_")
list_lab.append("Same performance")
list_color.append("black")
# plt.tight_layout()
fig.legend(
    list(zip(list_color, list_mak)),
    list_lab,
    handler_map={tuple: MarkerHandler()},
    loc="lower center",
    bbox_to_anchor=(0.5, -0.9),
    fancybox=True,
    ncol=5,
)

fig.savefig(
    str(Path(__file__).parents[0] / f"plots_paper/plot_introduction.pdf"),
    bbox_inches="tight",
)
