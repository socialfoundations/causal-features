"""Python script to plot experiments for introduction."""
#%%
from experiments_causal.plot_experiment import get_results
from experiments_causal.plot_experiment_anticausal import get_results as get_results_anticausal
from experiments_causal.plot_experiment_causalml import get_results as get_results_causalml
# from experiments_causal.plot_add_on_causal_discovery import get_results_pc_subsets
from experiments_causal.plot_config_colors import *
import seaborn as sns
from matplotlib.legend_handler import HandlerBase
from matplotlib.ticker import FormatStrFormatter
from experiments_causal.plot_config_tasks import dic_title
from scipy.spatial import ConvexHull
from paretoset import paretoset
import matplotlib.markers as mmark
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
from pathlib import Path
import warnings
import os
import json
warnings.filterwarnings("ignore")


# Reset colors for poster
color_all = "#9CBB06" # "#BAC477"
color_arguablycausal = "#0B60AC" #"#4C5D78"
color_causal = "#FB9E06" #DA944C"
color_anticausal = "#E7000E" #"#D35B50" #"#C25B32"
color_ml = "#830042"

# Set plot configurations
sns.set_context("poster")
sns.set_style("white")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 1200

def lighten_color(color, amount=0.5):
    try:
        c = mcolors.cnames[color]
    except KeyError:
        c = color
    c = mcolors.to_rgb(c)
    c = [min(1, max(0, channel + amount * (1 - channel))) for channel in c]
    return c

dic_title = {
    "acsemployment": "Employment",
    "acsfoodstamps": "Food Stamps",
    "acsincome": "Income",
    "acspubcov": "Public Coverage",
    "acsunemployment": "Unemployment",
    "anes": "Voting",
    "assistments": "ASSISTments",
    "brfss_blood_pressure": "Hypertension",
    "brfss_diabetes": "Diabetes",
    "college_scorecard": "College Scorecard",
    "diabetes_readmission": "Readmission",
    "meps": "Utilization",
    "mimic_extract_los_3": "Stay in ICU",
    "mimic_extract_mort_hosp": "Hospital Mortality",
    "nhanes_lead": "Childhood Lead",
    "physionet": "Sepsis",
    "sipp": "Poverty",
}

def get_results_pc_subsets(experiment_name: str) -> pd.DataFrame:

    """Load json files of experiments from results folder, concat them into a dataframe and save it.

    Parameters
    ----------
    experiment_name : str
        The name of the task.

    Returns
    -------
    pd.DataFrame
        Dataframe containing the results of the experiment.

    """
    experiments = [experiment_name]

    # Load all json files of experiments
    eval_all = pd.DataFrame()
    feature_selection = []
    for experiment in experiments:
        file_info = []
        try:
            RESULTS_DIR = Path(__file__).parents[0] / "add_on_results" / "causal_discovery" / experiment
            for filename in os.listdir(RESULTS_DIR):
                if filename == ".DS_Store":
                    pass
                else:
                    file_info.append(filename)

            for run in file_info:
                with open(str(RESULTS_DIR / run), "rb") as file:
                    # print(str(RESULTS_DIR / run))
                    try:
                        eval_json = json.load(file)
                        eval_pd = pd.DataFrame(
                            [
                                {
                                    "id_test": eval_json["id_test"],
                                    "id_test_lb": eval_json["id_test" + "_conf"][0],
                                    "id_test_ub": eval_json["id_test" + "_conf"][1],
                                    "ood_test": eval_json["ood_test"],
                                    "ood_test_lb": eval_json["ood_test" + "_conf"][0],
                                    "ood_test_ub": eval_json["ood_test" + "_conf"][1],
                                    "validation": eval_json["validation"],
                                    "features": f"random_{experiment.split('_')[-1]}",
                                    "model": run.split("_", 2)[0] + "_" + run.split("_", 2)[1] if run.split("_")[0] in ["ib", "and", "causirl"] else run.split("_")[0],
                                    "number": len(eval_json["features"]),
                                }
                            ]
                        )
                        eval_all = pd.concat([eval_all, eval_pd], ignore_index=True)
                    except:
                        print(str(RESULTS_DIR / run))
        except:
            print(experiment)

    # Select model with highest in-domain validation accuracy
    list_model_data = []
    for set in eval_all["features"].unique():
        eval_feature = eval_all[eval_all["features"] == set]
        for model in eval_feature["model"].unique():
            model_data = eval_feature[eval_feature["model"] == model]
            model_data = model_data[
                model_data["validation"] == model_data["validation"].max()
            ]
            model_data.drop_duplicates(inplace=True)
            list_model_data.append(model_data)
    eval_all = pd.concat(list_model_data)
    return eval_all


#%%
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
# %%

fig, ax = plt.subplots(figsize=(10, 4.8)) 

# ax.set_xlabel(f"Tasks")
ax.set_ylabel(f"Out-of-domain accuracy")

#############################################################################
# plot ood accuracy
#############################################################################
markers = {"constant": "X", "all": "s", "causal": "o", "arguablycausal": "D"}

sets = list(eval_experiments["features"].unique())
sets.sort()

for index, set in enumerate(sets):
    eval_plot_features = eval_experiments[eval_experiments["features"] == set]
    eval_plot_features = eval_plot_features.sort_values("ood_test")
    ax.errorbar(
        x=eval_plot_features["task"],
        y=eval_plot_features["ood_test"],
        yerr=eval_plot_features["ood_test_ub"] - eval_plot_features["ood_test"],
        color=eval(f"color_{set}"),
        ecolor=lighten_color(eval(f"color_{set}"),amount=0.5),
        fmt=markers[set],
        # markersize=markersize,
        # capsize=capsize,
        label=set.capitalize() if set != "arguablycausal" else "Arguably causal",
        zorder=len(sets) - index,
    )

ax.tick_params(axis="x", labelrotation=90)
ax.set_ylim(top=1.0)
ax.grid(axis="y")
# plt.tight_layout()


fig.savefig(
    str(Path(__file__).parents[0] / f"plots_poster/plot_introduction.pdf"),
    bbox_inches="tight",
)

#%%
experiments = ["acsincome", "brfss_diabetes","acsunemployment"]

for index, experiment_name in enumerate(experiments):
    fig, ax = plt.subplots() 

    eval_all = get_results_anticausal(
            experiment_name
        )
    eval_constant = eval_all[eval_all["features"] == "constant"]

    ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

    ax.set_xlabel(f"In-domain accuracy")
    ax.set_ylabel(f"Out-of-domain accuracy")
    #############################################################################
    # plot errorbars and shift gap for constant
    #############################################################################
    errors = ax.errorbar(
        x=eval_constant["id_test"],
        y=eval_constant["ood_test"],
        xerr=eval_constant["id_test_ub"] - eval_constant["id_test"],
        yerr=eval_constant["ood_test_ub"] - eval_constant["ood_test"],
        fmt="X",
        color=color_constant,
        ecolor=lighten_color(color_constant),
        # markersize=markersize,
        # capsize=capsize,
        label="constant",
    )

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
    markers = markers[
        markers["id_test"] >= (eval_constant["id_test"].values[0] - 0.01)
    ]
    errors = ax.errorbar(
        x=markers["id_test"],
        y=markers["ood_test"],
        xerr=markers["id_test_ub"] - markers["id_test"],
        yerr=markers["ood_test_ub"] - markers["ood_test"],
        fmt="o",
        color=color_causal,
        ecolor=lighten_color(color_causal),
        # markersize=markersize,
        # capsize=capsize,
        label="causal",
    )
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
        points = points[
            points["id_test"] >= (eval_constant["id_test"].values[0] - 0.01)
        ]
        markers = eval_plot[mask]
        markers = markers[
            markers["id_test"] >= (eval_constant["id_test"].values[0] - 0.01)
        ]
        errors = ax.errorbar(
            x=markers["id_test"],
            y=markers["ood_test"],
            xerr=markers["id_test_ub"] - markers["id_test"],
            yerr=markers["ood_test_ub"] - markers["ood_test"],
            fmt="D",
            color=color_arguablycausal,
            ecolor=lighten_color(color_arguablycausal),
            # markersize=markersize,
            # capsize=capsize,
            label="arguably\ncausal",
        )
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
    markers = markers[
        markers["id_test"] >= (eval_constant["id_test"].values[0] - 0.01)
    ]
    errors = ax.errorbar(
        x=markers["id_test"],
        y=markers["ood_test"],
        xerr=markers["id_test_ub"] - markers["id_test"],
        yerr=markers["ood_test_ub"] - markers["ood_test"],
        fmt="s",
        color=color_all,
        ecolor=lighten_color(color_all),
        # markersize=markersize,
        # capsize=capsize,
        label="all",
    )
    #############################################################################
    # plot errorbars and shift gap for anticausal features
    #############################################################################
    if (eval_all["features"] == "anticausal").any():
        eval_plot = eval_all[eval_all["features"] == "anticausal"]
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
        errors = ax.errorbar(
            x=markers["id_test"],
            y=markers["ood_test"],
            xerr=markers["id_test_ub"] - markers["id_test"],
            yerr=markers["ood_test_ub"] - markers["ood_test"],
            fmt="P",
            color=color_anticausal,
            ecolor=lighten_color(color_anticausal),
            # markersize=markersize,
            # capsize=capsize,
            label="anticausal",
        )

    # #############################################################################
    # # plot errorbars and shift gap for causal+anticausal features
    # #############################################################################
    # if (eval_all["features"] == "causal_anticausal").any():
    #     eval_plot = eval_all[eval_all["features"] == "causal_anticausal"]
    #     eval_plot.sort_values("id_test", inplace=True)
    #     # Calculate the pareto set
    #     points = eval_plot[["id_test", "ood_test"]]
    #     mask = paretoset(points, sense=["max", "max"])
    #     points = points[mask]
    #     points = points[
    #         points["id_test"] >= (eval_constant["id_test"].values[0] - 0.01)
    #     ]
    #     markers = eval_plot[mask]
    #     markers = markers[
    #         markers["id_test"] >= (eval_constant["id_test"].values[0] - 0.01)
    #     ]
    #     errors = ax.errorbar(
    #         x=markers["id_test"],
    #         y=markers["ood_test"],
    #         xerr=markers["id_test_ub"] - markers["id_test"],
    #         yerr=markers["ood_test_ub"] - markers["ood_test"],
    #         fmt="^",
    #         color="tab:blue",
    #         ecolor=lighten_color("tab:blue"),
    #         # markersize=markersize,
    #         # capsize=capsize,
    #         label="causal+anticausal",
    #     )
    #############################################################################
    # plot pareto dominated area for constant
    #############################################################################
    xmin, xmax = ax.set_xlim()
    ymin, ymax = ax.set_ylim()
    ax.plot(
        [xmin, eval_constant["id_test"].values[0]],
        [eval_constant["ood_test"].values[0], eval_constant["ood_test"].values[0]],
        color=color_constant,
        linestyle=(0, (1, 1)),
        linewidth=linewidth_bound,
    )
    ax.plot(
        [eval_constant["id_test"].values[0], eval_constant["id_test"].values[0]],
        [ymin, eval_constant["ood_test"].values[0]],
        color=color_constant,
        linestyle=(0, (1, 1)),
        linewidth=linewidth_bound,
    )
    ax.fill_between(
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
    ax.plot(
        points["id_test"],
        points["ood_test"],
        color=color_all,
        linestyle=(0, (1, 1)),
        # linewidth=linewidth_bound,
    )
    new_row = pd.DataFrame(
        {"id_test": [xmin], "ood_test": [ymin]},
    )
    points = pd.concat([points, new_row], ignore_index=True)
    points = points.to_numpy()
    hull = ConvexHull(points)
    ax.fill(
        points[hull.vertices, 0],
        points[hull.vertices, 1],
        color=color_all,
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
    markers = markers[
        markers["id_test"] >= (eval_constant["id_test"].values[0] - 0.01)
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
    ax.plot(
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
    ax.fill(
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
        ax.plot(
            points["id_test"],
            points["ood_test"],
            color=color_arguablycausal,
            linestyle=(0, (1, 1)),
            # linewidth=linewidth_bound,
        )
        new_row = pd.DataFrame(
            {"id_test": [xmin], "ood_test": [ymin]},
        )
        points = pd.concat([points, new_row], ignore_index=True)
        points = points.to_numpy()
        hull = ConvexHull(points)
        ax.fill(
            points[hull.vertices, 0],
            points[hull.vertices, 1],
            color=color_arguablycausal,
            alpha=0.05,
        )

    #############################################################################
    # plot pareto dominated area for anticausal features
    #############################################################################
    if (eval_all["features"] == "anticausal").any():
        eval_plot = eval_all[eval_all["features"] == "anticausal"]
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
        ax.plot(
            points["id_test"],
            points["ood_test"],
            color=color_anticausal,
            linestyle=(0, (1, 1)),
            # linewidth=linewidth_bound,
        )

        new_row = pd.DataFrame(
            {"id_test": [xmin], "ood_test": [ymin]},
        )
        points = pd.concat([points, new_row], ignore_index=True)
        points = points.to_numpy()
        hull = ConvexHull(points)
        ax.fill(
            points[hull.vertices, 0],
            points[hull.vertices, 1],
            color=color_anticausal,
            alpha=0.05,
        )

    
    fig.savefig(
    str(Path(__file__).parents[0] / f"plots_poster/plot_{experiment_name}_anticausal.pdf"),
    bbox_inches="tight",
    )

#%%
experiments = ["acsincome", "brfss_diabetes", "mimic_extract_mort_hosp"]

for index, experiment_name in enumerate(experiments):
    fig, ax = plt.subplots() 

    eval_all = get_results(
            experiment_name
        )
    eval_constant = eval_all[eval_all["features"] == "constant"]

    ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

    ax.set_xlabel(f"In-domain accuracy")
    ax.set_ylabel(f"Out-of-domain accuracy")
    #############################################################################
    # plot errorbars and shift gap for constant
    #############################################################################
    errors = ax.errorbar(
        x=eval_constant["id_test"],
        y=eval_constant["ood_test"],
        xerr=eval_constant["id_test_ub"] - eval_constant["id_test"],
        yerr=eval_constant["ood_test_ub"] - eval_constant["ood_test"],
        fmt="X",
        color=color_constant,
        ecolor=lighten_color(color_constant),
        # markersize=markersize,
        # capsize=capsize,
        label="constant",
    )

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
    markers = markers[
        markers["id_test"] >= (eval_constant["id_test"].values[0] - 0.01)
    ]
    errors = ax.errorbar(
        x=markers["id_test"],
        y=markers["ood_test"],
        xerr=markers["id_test_ub"] - markers["id_test"],
        yerr=markers["ood_test_ub"] - markers["ood_test"],
        fmt="o",
        color=color_causal,
        ecolor=lighten_color(color_causal),
        # markersize=markersize,
        # capsize=capsize,
        label="causal",
    )
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
        points = points[
            points["id_test"] >= (eval_constant["id_test"].values[0] - 0.01)
        ]
        markers = eval_plot[mask]
        markers = markers[
            markers["id_test"] >= (eval_constant["id_test"].values[0] - 0.01)
        ]
        errors = ax.errorbar(
            x=markers["id_test"],
            y=markers["ood_test"],
            xerr=markers["id_test_ub"] - markers["id_test"],
            yerr=markers["ood_test_ub"] - markers["ood_test"],
            fmt="D",
            color=color_arguablycausal,
            ecolor=lighten_color(color_arguablycausal),
            # markersize=markersize,
            # capsize=capsize,
            label="arguably\ncausal",
        )
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
    markers = markers[
        markers["id_test"] >= (eval_constant["id_test"].values[0] - 0.01)
    ]
    errors = ax.errorbar(
        x=markers["id_test"],
        y=markers["ood_test"],
        xerr=markers["id_test_ub"] - markers["id_test"],
        yerr=markers["ood_test_ub"] - markers["ood_test"],
        fmt="s",
        color=color_all,
        ecolor=lighten_color(color_all),
        # markersize=markersize,
        # capsize=capsize,
        label="all",
    )
    
    #############################################################################
    # plot pareto dominated area for constant
    #############################################################################
    xmin, xmax = ax.set_xlim()
    ymin, ymax = ax.set_ylim()
    ax.plot(
        [xmin, eval_constant["id_test"].values[0]],
        [eval_constant["ood_test"].values[0], eval_constant["ood_test"].values[0]],
        color=color_constant,
        linestyle=(0, (1, 1)),
        linewidth=linewidth_bound,
    )
    ax.plot(
        [eval_constant["id_test"].values[0], eval_constant["id_test"].values[0]],
        [ymin, eval_constant["ood_test"].values[0]],
        color=color_constant,
        linestyle=(0, (1, 1)),
        linewidth=linewidth_bound,
    )
    ax.fill_between(
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
    ax.plot(
        points["id_test"],
        points["ood_test"],
        color=color_all,
        linestyle=(0, (1, 1)),
        # linewidth=linewidth_bound,
    )
    new_row = pd.DataFrame(
        {"id_test": [xmin], "ood_test": [ymin]},
    )
    points = pd.concat([points, new_row], ignore_index=True)
    points = points.to_numpy()
    hull = ConvexHull(points)
    ax.fill(
        points[hull.vertices, 0],
        points[hull.vertices, 1],
        color=color_all,
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
    markers = markers[
        markers["id_test"] >= (eval_constant["id_test"].values[0] - 0.01)
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
    ax.plot(
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
    ax.fill(
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
        ax.plot(
            points["id_test"],
            points["ood_test"],
            color=color_arguablycausal,
            linestyle=(0, (1, 1)),
            # linewidth=linewidth_bound,
        )
        new_row = pd.DataFrame(
            {"id_test": [xmin], "ood_test": [ymin]},
        )
        points = pd.concat([points, new_row], ignore_index=True)
        points = points.to_numpy()
        hull = ConvexHull(points)
        ax.fill(
            points[hull.vertices, 0],
            points[hull.vertices, 1],
            color=color_arguablycausal,
            alpha=0.05,
        )

    fig.savefig(
    str(Path(__file__).parents[0] / f"plots_poster/plot_{experiment_name}.pdf"),
    bbox_inches="tight",
    )

#%%
markers_causalml = {
                "irm": "v",
                "vrex": "v",
                "ib_irm": "v",
                "and_mask": "v",
                "causirl_mmd": "v",
                "causirl_coral": "v",}
# color_ml = "#AB5878"

experiments = ["brfss_diabetes","acsincome","acsunemployment","mimic_extract_mort_hosp",]


for index, experiment_name in enumerate(experiments):
    fig, ax = plt.subplots()

    if experiment_name == "brfss_diabetes":
        eval_all = get_results(experiment_name)
    else:
        eval_all = get_results_causalml(experiment_name)
    eval_constant = eval_all[eval_all["features"] == "constant"]
    dic_shift = {}

    ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

    ax.set_xlabel(f"In-domain accuracy")
    ax.set_ylabel(f"Out-of-domain accuracy")

    ##############################################################################
    # plot errorbars and shift gap for constant
    #############################################################################
    errors = ax.errorbar(
        x=eval_constant["id_test"],
        y=eval_constant["ood_test"],
        xerr=eval_constant["id_test_ub"] - eval_constant["id_test"],
        yerr=eval_constant["ood_test_ub"] - eval_constant["ood_test"],
        fmt="X",
        color=color_constant,
        ecolor=lighten_color(color_constant, amount=0.5),
        # markersize=markersize,
        # capsize=capsize,
        label="constant",
    )

        

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
        errors = ax.errorbar(
            x=markers["id_test"],
            y=markers["ood_test"],
            xerr=markers["id_test_ub"] - markers["id_test"],
            yerr=markers["ood_test_ub"] - markers["ood_test"],
            fmt=markers_causalml[causalml],
            color=color_ml,
            ecolor=lighten_color(color_ml, amount=0.5),
            # markersize=markersize,
            # capsize=capsize,
            label="causal ml",
        )

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
        errors = ax.errorbar(
            x=markers["id_test"],
            y=markers["ood_test"],
            xerr=markers["id_test_ub"] - markers["id_test"],
            yerr=markers["ood_test_ub"] - markers["ood_test"],
            fmt=markers_causalml[causalml],
            color=color_ml,
            ecolor=lighten_color(color_ml, amount=0.5),
            # markersize=markersize,
            # capsize=capsize,
            label="causal ml",
        )

    if experiment_name == "acsunemployment":
        #############################################################################
        # plot errorbars for causal discovery features
        #############################################################################
        for discovery in ["pc","icp"]:
            eval_all_causal_discovery = get_results_pc_subsets(f"{experiment_name}_{discovery}")

            for set in eval_all_causal_discovery["features"].unique():
                eval_plot = eval_all_causal_discovery[eval_all_causal_discovery["features"] == set]
                eval_plot.sort_values("id_test", inplace=True)
                # Calculate the pareto set
                points = eval_plot[["id_test", "ood_test"]]
                mask = paretoset(points, sense=["max", "max"])
                markers = eval_plot[mask]
                errors = ax.errorbar(
                    x=markers["id_test"],
                    y=markers["ood_test"],
                    xerr=markers["id_test_ub"] - markers["id_test"],
                    yerr=markers["ood_test_ub"] - markers["ood_test"],
                    fmt="P",
                    color="#48B3FE",
                    ecolor=lighten_color("#48B3FE"),
                )

    if experiment_name == "acsincome":
        #############################################################################
        # plot errorbars for causal discovery features
        #############################################################################
        for discovery in ["pc"]:
            eval_all_causal_discovery = get_results_pc_subsets(f"{experiment_name}_{discovery}")

            for set in eval_all_causal_discovery["features"].unique():
                eval_plot = eval_all_causal_discovery[eval_all_causal_discovery["features"] == set]
                eval_plot.sort_values("id_test", inplace=True)
                # Calculate the pareto set
                points = eval_plot[["id_test", "ood_test"]]
                mask = paretoset(points, sense=["max", "max"])
                markers = eval_plot[mask]
                errors = ax.errorbar(
                    x=markers["id_test"],
                    y=markers["ood_test"],
                    xerr=markers["id_test_ub"] - markers["id_test"],
                    yerr=markers["ood_test_ub"] - markers["ood_test"],
                    fmt="P",
                    color="#48B3FE",
                    ecolor=lighten_color("#48B3FE"),
                )
    if experiment_name == "brfss_diabetes":
        #############################################################################
        # plot errorbars for causal discovery features
        #############################################################################
        for discovery in ["pc","fci"]:
            eval_all_causal_discovery = get_results_pc_subsets(f"{experiment_name}_{discovery}")

            for set in eval_all_causal_discovery["features"].unique():
                eval_plot = eval_all_causal_discovery[eval_all_causal_discovery["features"] == set]
                eval_plot.sort_values("id_test", inplace=True)
                # Calculate the pareto set
                points = eval_plot[["id_test", "ood_test"]]
                mask = paretoset(points, sense=["max", "max"])
                markers = eval_plot[mask]
                errors = ax.errorbar(
                    x=markers["id_test"],
                    y=markers["ood_test"],
                    xerr=markers["id_test_ub"] - markers["id_test"],
                    yerr=markers["ood_test_ub"] - markers["ood_test"],
                    fmt="P",
                    color="#48B3FE",
                    ecolor=lighten_color("#48B3FE"),
                )
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
    errors = ax.errorbar(
        x=markers["id_test"],
        y=markers["ood_test"],
        xerr=markers["id_test_ub"] - markers["id_test"],
        yerr=markers["ood_test_ub"] - markers["ood_test"],
        fmt="s",
        color=color_all,
        ecolor=lighten_color(color_all, amount=0.5),
        # markersize=markersize,
        # capsize=capsize,
        label="all",
    )


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
    errors = ax.errorbar(
        x=markers["id_test"],
        y=markers["ood_test"],
        xerr=markers["id_test_ub"] - markers["id_test"],
        yerr=markers["ood_test_ub"] - markers["ood_test"],
        fmt="o",
        color=color_causal,
        ecolor=lighten_color(color_causal, amount=0.5),
        # markersize=markersize,
        # capsize=capsize,
        label="causal",
    )
    

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
        errors = ax.errorbar(
            x=markers["id_test"],
            y=markers["ood_test"],
            xerr=markers["id_test_ub"] - markers["id_test"],
            yerr=markers["ood_test_ub"] - markers["ood_test"],
            fmt="D",
            color=color_arguablycausal,
            ecolor=lighten_color(color_arguablycausal, amount=0.5),
            # markersize=markersize,
            # capsize=capsize,
            label="arguably\ncausal",
        )

    #############################################################################
    # plot pareto dominated area for constant
    #############################################################################
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    ax.plot(
        [xmin, eval_constant["id_test"].values[0]],
        [eval_constant["ood_test"].values[0], eval_constant["ood_test"].values[0]],
        color=color_constant,
        linestyle=(0, (1, 1)),
        linewidth=linewidth_bound,
    )
    ax.plot(
        [eval_constant["id_test"].values[0], eval_constant["id_test"].values[0]],
        [ymin, eval_constant["ood_test"].values[0]],
        color=color_constant,
        linestyle=(0, (1, 1)),
        linewidth=linewidth_bound,
    )
    ax.fill_between(
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
    ax.plot(
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
    ax.fill(
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
    ax.plot(
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
    ax.fill(
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
        ax.plot(
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
        ax.fill(
            points[hull.vertices, 0],
            points[hull.vertices, 1],
            color=color_arguablycausal,
            alpha=0.05,
        )
    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))

    fig.savefig(
    str(Path(__file__).parents[0] / f"plots_poster/plot_{experiment_name}_causalml.pdf"),
    bbox_inches="tight",
    )
    
# %%
