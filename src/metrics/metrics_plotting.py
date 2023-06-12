import time

from src.data_preprocessing.META.METADataBuilder import MetaDataBuilder

import matplotlib.pyplot as plt
import src.utils.utils_generic as util
import src.constants as cst

import numpy as np
import seaborn as sb
import pandas as pd
import matplotlib.gridspec as gridspec
import os
import json
from decimal import Decimal


def setup_plotting_env():
    plt.rcParams["figure.figsize"] = [16, 9]
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["font.size"] = 20
    plt.rcParams["axes.labelsize"] = 20
    plt.rcParams["axes.titlesize"] = 20
    plt.rcParams["xtick.labelsize"] = 16
    plt.rcParams["ytick.labelsize"] = 16
    plt.rcParams["font.family"] = "serif"


def reproduced_metrics(path, metrics, list_models, list_horizons, list_seeds, dataset_type, train_src="ALL", test_src="ALL", time_period=cst.Periods.JULY2021.name, jolly_seed=None):
    METRICS = np.zeros(shape=(len(list_seeds), len(list_models), len(list_horizons), len(metrics)))

    for imod, mod in enumerate(list_models):
        for ik, k in enumerate(list_horizons):
            for iss, s in enumerate(list_seeds):
                # print(mod, k, s)

                bw, fw, fik = None, None, k
                if dataset_type == cst.DatasetFamily.LOB:
                    bw, fw = k
                    bw, fw = bw.value, fw.value
                    fik = cst.FI_Horizons.K10

                fname = "model={}-seed={}-trst={}-test={}-data={}-peri={}-bw={}-fw={}-fiw={}.json".format(
                    mod.name, s, train_src, test_src, cst.model_dataset(mod, bias=dataset_type), time_period, bw, fw, fik.value
                )

                if mod == cst.Models.MAJORITY and jolly_seed is not None:  # only 1 seed for the baseline
                    if s == jolly_seed:
                        jsn = util.read_json(path + fname)
                        vec = np.zeros(shape=(len(metrics)))
                        for imet, met in enumerate(metrics):
                            vec[imet] = jsn[met]
                        repeat = np.repeat(np.expand_dims(vec, axis=0), len(list_seeds), axis=0)
                        METRICS[:, imod, ik, :] = repeat
                else:
                    jsn = util.read_json(path + fname)
                    for imet, met in enumerate(metrics):
                        METRICS[iss, imod, ik, imet] = jsn[met]
    return METRICS


def reproduced_metrics_stocks(path, list_models, list_stocks, list_seeds, dataset_type, train_src="ALL", time_period=cst.Periods.JULY2021.name, jolly_seed=None, target_horizon=cst.WinSize.EVENTS5):
    n_metrics = 8
    METRICS = np.zeros(shape=(len(list_seeds), len(list_models), len(list_stocks), n_metrics))

    for imod, mod in enumerate(list_models):
        for ik, sto in enumerate(list_stocks):
            metrics = metrics_to_plot(sto)
            for iss, s in enumerate(list_seeds):

                bw, fw = cst.WinSize.EVENTS1, target_horizon
                bw, fw = bw.value, fw.value
                fik = cst.FI_Horizons.K10

                fname = "model={}-seed={}-trst={}-test={}-data={}-peri={}-bw={}-fw={}-fiw={}.json".format(
                    mod.name, s, train_src, sto, cst.model_dataset(mod, bias=dataset_type), time_period, bw, fw, fik.value
                )

                if mod == cst.Models.MAJORITY and jolly_seed is not None:  # only 1 seed for the baseline
                    if s == jolly_seed:
                        jsn = util.read_json(path + fname)
                        vec = np.zeros(shape=(len(metrics)))
                        for imet, met in enumerate(metrics):
                            vec[imet] = jsn[met]
                        repeat = np.repeat(np.expand_dims(vec, axis=0), len(list_seeds), axis=0)
                        METRICS[:, imod, ik, :] = repeat
                else:
                    jsn = util.read_json(path + fname)
                    for imet, met in enumerate(metrics):
                        METRICS[iss, imod, ik, imet] = jsn[met]
    return METRICS


def inference_data(path, list_models, dataset_type, seed=500, horizon=10, bw=1, fw=5, train_src="ALL", test_src="ALL", time_period=cst.Periods.JULY2021.name):
    INFERENCE = np.zeros(shape=(2, len(list_models)))
    for imod, mod in enumerate(list_models):
        if mod == cst.Models.MAJORITY:
            continue

        if dataset_type == cst.DatasetFamily.LOB:
            horizon = cst.FI_Horizons.K10.value
        elif dataset_type == cst.DatasetFamily.FI:
            bw, fw = None, None

        fname = "model={}-seed={}-trst={}-test={}-data={}-peri={}-bw={}-fw={}-fiw={}.json".format(
            mod.name, seed, train_src, test_src, cst.model_dataset(mod, bias=dataset_type), time_period, bw, fw, horizon
        )

        jsn = util.read_json(path + fname)
        INFERENCE[0, imod] = jsn['inference_mean']
        INFERENCE[1, imod] = jsn['inference_std']

    return INFERENCE


def original_metrics(metrics, list_models, list_horizons):
    METRICS_ORI = np.zeros(shape=(len(list_models), len(list_horizons), len(metrics)))

    for imod, mod in enumerate(list_models):
        for ik, k in enumerate(list_horizons):
            for imet, met in enumerate(metrics):
                mid = map_id_metric_declared(metrics, met)
                if mod in cst.DECLARED_PERF and mid is not None:
                    wid = cst.map_id_win(k)
                    METRICS_ORI[imod, ik, imet] = cst.DECLARED_PERF[cst.Models[mod.name]][wid][mid]
                else:
                    METRICS_ORI[imod, ik, imet] = None
    return METRICS_ORI


def confusion_metrix(path, list_models, list_horizons, list_seeds, dataset_type, train_src="ALL", test_src="ALL", time_period=cst.Periods.JULY2021.name, jolly_seed=None):
    CMS = np.zeros(shape=(len(list_seeds), len(list_models), len(list_horizons), 3, 3))  # 5 x 15 x 5 x 3 x 3

    for imod, mod in enumerate(list_models):
        for ik, k in enumerate(list_horizons):
            for iss, s in enumerate(list_seeds):

                bw, fw, fik = None, None, k
                if dataset_type == cst.DatasetFamily.LOB:
                    bw, fw = k
                    bw, fw = bw.value, fw.value
                    fik = cst.FI_Horizons.K10

                fname = "model={}-seed={}-trst={}-test={}-data={}-peri={}-bw={}-fw={}-fiw={}.json".format(
                    mod.name, s, train_src, test_src, cst.model_dataset(mod, bias=dataset_type), time_period, bw, fw,
                    fik.value
                )

                if mod == cst.Models.MAJORITY and jolly_seed is not None:  # only seed for the baseline
                    if s == jolly_seed:
                        jsn = util.read_json(path + fname)
                        repeat = np.repeat(np.expand_dims(np.array(jsn["cm"]), axis=0), len(list_seeds), axis=0)
                        CMS[:, imod, ik, :, :] = repeat
                else:
                    jsn = util.read_json(path + fname)
                    CMS[iss, imod, ik] = np.array(jsn["cm"])
    return CMS


# def relative_improvement_table(metrics_repr, metrics_original, list_models):
#     # relative improvement
#     me = np.nanmean(metrics_repr, axis=1)
#     ome = np.nanmean(metrics_original, axis=1) / 100
#
#     tab = (me - ome) / ome
#     tab = tab.T
#     tab = np.round(tab * 100, 2)
#
#     improvement = pd.DataFrame(tab, index=cst.metrics_name, columns=[k.name for k in list_models])
#     # print(improvement.to_latex(index=True))
#     return improvement


def metrics_vs_models_bars(met_name, horizons, met_vec, out_dir, list_models, dataset_type, met_vec_original=None, is_stocks=False):

    horizons = np.array(horizons)
    if dataset_type == cst.DatasetFamily.FI:
        labels = ["K={}".format(k.value) for k in horizons]
    elif dataset_type == cst.DatasetFamily.LOB and not is_stocks:
        labels = ["K={}".format(fw.value) for bw, fw in horizons]
    elif is_stocks:
        labels = ["S={}".format(bw) for bw in horizons]

    fmt = "%.2f"
    miny, maxy = -1, 1.1

    if "MCC" not in met_name:
        met_vec = met_vec * 100
        fmt = "%d%%"
        miny, maxy = 0, 110

    avg_met = np.average(met_vec, axis=0)  # avg seeds
    std_met = np.std(met_vec, axis=0)

    x = np.arange(len(labels)) * 9  # the label locations
    width = 0.48  # the width of the bars

    fig, ax = plt.subplots(figsize=(21, 10))

    indsxs = np.array(range(1, int(len(list_models) / 2) + 1))
    zero = [0] if len(list_models) % 2 == 1 else []
    ranges = list(reversed(indsxs * -1)) + zero + list(indsxs)

    LABEL_FONT_SIZE = 12

    R = []  # the bars
    for iri, ri in enumerate(list_models):
        if dataset_type == cst.DatasetFamily.FI:
            r_bar_i = ax.bar(x + width * ranges[iri], avg_met[iri, :], width, yerr=std_met[iri, :], label=ri.name,
                             color=util.sample_color(iri, "tab20"), align='center')  # hatch=util.sample_pattern(iri))

            R += [r_bar_i]
            if met_vec_original is None:
                bar_value = ["{}%".format(Decimal(str(round(it, 1))).normalize()) for it in avg_met[iri, :]]
                ax.bar_label(r_bar_i, labels=bar_value, padding=3, fmt=fmt, rotation=90, fontsize=LABEL_FONT_SIZE)

        elif dataset_type == cst.DatasetFamily.LOB:
            r_bar_i = ax.bar(x + width * ranges[iri], avg_met[iri, :], width, yerr=std_met[iri, :],
                             label=ri.name, color=util.sample_color(iri, "tab20"), align='center', edgecolor='black')  # hatch=util.sample_pattern(iri))

            bar_value = ["{}%".format(Decimal(str(round(it, 1))).normalize()) for it in avg_met[iri, :]]
            ax.bar_label(r_bar_i, labels=bar_value, padding=3, fmt=fmt, rotation=90, fontsize=LABEL_FONT_SIZE)

    if met_vec_original is not None:
        diffp = avg_met - met_vec_original
        for iri, ri in enumerate(R):  # models

            diffp_show = []
            for i, val in enumerate(diffp[iri, :]):  # horizons
                our = round(avg_met[iri, i], 1)
                our = str(Decimal(str(our)).normalize())

                # our = round(avg_met) if "MCC" not in met_name else round(avg_met, 2)
                # our_2 = int(our) if int(our) / our == 1.0 else our

                if not np.isnan(val):
                    val = str(Decimal(str(round(val, 1))).normalize())
                    val = val if float(val) < 0 else "+{}".format(val)
                    diffp_show += ["{}% (".format(our) + "{}%)".format(val)]
                else:
                    diffp_show += ["{}% ($\cdot$)".format(our)]

                if "MCC" in met_name:
                    diffp_show = [s.replace('%', '') for s in diffp_show]

            ax.bar_label(ri, labels=diffp_show, padding=3, fmt=fmt, rotation=90, fontsize=LABEL_FONT_SIZE)

        for iri, ri in enumerate(list_models):
            label = 'original' if iri == 0 else ''  # black bars in FI
            ax.bar(x + width * ranges[iri], met_vec_original[iri, :], width, alpha=1, bottom=0, fill=False,
                   edgecolor='black', label=label, align='center')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(met_name)

    dataset_name = dataset_type.name
    dataset_name = "FI-2010" if cst.DatasetFamily.FI == dataset_name else dataset_name

    ax.set_title("{}".format(dataset_name))
    ax.set_xticks(x, labels)  # , rotation=0, ha="right", rotation_mode="anchor")

    if dataset_type == cst.DatasetFamily.FI:
        ax.set_ylim((20, 100))

    elif dataset_type == cst.DatasetFamily.LOB:
        ax.set_ylim((25, 70))
        if is_stocks:
            ax.set_ylim((0, 75))

    ax.legend(fontsize=15, ncol=6, handleheight=2, labelspacing=0.05, loc="lower right", framealpha=1)

    # plt.ylim(miny, maxy)
    fig.tight_layout()
    met_name_new = met_name.replace("(%)", "perc")
    met_name_new = met_name_new.replace(" ", "_")
    pdf_path = out_dir + f"ultimate_bar-{'stocks' if is_stocks else 'horizons'}-nbars{len(horizons)}-" + met_name_new + ".pdf"
    print(pdf_path)
    plt.savefig(pdf_path)
    # plt.show()
    plt.close(fig)


def metrics_vs_models_k_line(met_name, horizons, met_vec, out_dir, list_models, dataset_type, type=None):

    horizons = np.array(horizons)
    if dataset_type == cst.DatasetFamily.FI:
        labels = ["K={}".format(k.value) for k in horizons]
    else:
        labels = ["K={}".format(fw.value) for bw, fw in horizons]

    if "MCC" not in met_name:
        met_vec = met_vec * 100

    avg_met = np.average(met_vec, axis=0)  # avg seeds
    std_met = np.std(met_vec, axis=0)

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9, 7))

    for iri, ri in enumerate(list_models):
        ax.fill_between(x, std_met[iri, :], -std_met[iri, :], alpha=.3, linewidth=0, color=util.sample_color(iri, "tab20"))
        ax.plot(x, avg_met[iri, :], label=ri.name, color=util.sample_color(iri, "tab20"), marker=util.sample_marker(iri))

    ax.relim()
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(met_name)
    ax.set_title("{}".format(dataset_type.name, met_name))
    ax.set_xticks(x, labels)

    ax.legend(fontsize=12, ncol=4, handleheight=2, labelspacing=0.05)

    fig.tight_layout()
    met_name_new = met_name.replace("(%)", "perc")
    plt.savefig(out_dir + type + "-line" + met_name + ".pdf")
    # plt.show()
    plt.close(fig)


def plot_inference_time(met_vec, met_vec_err, list_models, out_dir):

    labels = ["K=5"]

    x = np.arange(len(labels)) * 9  # the label locations
    print(x)
    exit()
    width = 0.5  # the width of the bars

    fig, ax = plt.subplots(figsize=(9, 7))

    indsxs = np.array(range(1, int(len(list_models) / 2) + 1))
    zero = [0] if len(list_models) % 2 == 1 else []
    ranges = list(reversed(indsxs * -1)) + zero + list(indsxs)

    for iri, ri in enumerate(list_models):
        ax.bar(x + width * ranges[iri], met_vec[iri], width, yerr=met_vec_err[iri], label=ri.name,
                         color=util.sample_color(iri, "tab20"), align='center')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("Time (s)")
    ax.set_title("Models Inference Time")
    ax.set_xticks(x, labels)  # , rotation=0, ha="right", rotation_mode="anchor")

    ax.legend(fontsize=12, ncol=3, handleheight=2, labelspacing=0.05)

    # plt.ylim(miny, maxy)
    fig.tight_layout()

    plt.savefig(out_dir + "inference_time.pdf")
    # plt.show()
    plt.close(fig)


def confusion_matrix_grid(cms, list_models, out_dir, chosen_horizons, dataset):
    fig = plt.figure(figsize=(30, 30 * 2.5))
    gs = gridspec.GridSpec(len(list_models), len(chosen_horizons) + 1,
                           width_ratios=[6 for _ in range(len(chosen_horizons))] + [1], figure=fig)

    annot_kws = {
        'fontsize': 16,
        'fontweight': 'bold',
        'fontfamily': 'serif'
    }

    for imod, mod in enumerate(list_models):
        for ik, k in enumerate(chosen_horizons):
            cbar = False  # ik == 0 and imod == 0
            csm_norm = cms[imod, ik] / np.sum(cms[imod, ik], axis=1)[:, None]

            axi = fig.add_subplot(gs[imod, ik])
            sb.heatmap(csm_norm, annot=True, ax=axi, cbar=cbar, fmt=".2%", cmap="Blues", annot_kws=annot_kws)

            axi.set_xticklabels([p.name for p in cst.Predictions], rotation=0, fontsize=12)
            axi.set_yticklabels([p.name for p in cst.Predictions], rotation=90, fontsize=12)

            if imod == 0:
                k = k.value if dataset == cst.DatasetFamily.FI else (k[0].value, k[1].value)
                axi.set_title("K={}".format(k), fontsize=30, fontweight="bold", pad=25)

            if ik == 0:
                axi.set_ylabel(mod.name, fontsize=30, fontweight="bold", labelpad=25)

    legend_ax = fig.add_subplot(gs[:, -1])

    sm = plt.cm.ScalarMappable(cmap="Blues", norm=plt.Normalize(0, 100))
    sm.set_array([])

    fig.colorbar(sm, cax=legend_ax)

    fig.supylabel('Real', x=0.02, fontsize=25)
    fig.supxlabel('Predicted', y=.965, fontsize=25)
    fig.suptitle('FI-2010 Confusion Matrix', y=.98, fontsize=30, fontweight="bold")

    fig.tight_layout()
    fig.subplots_adjust(top=.95, left=.087)

    fig.savefig(out_dir + "cm-fi.pdf")
    # plt.show()
    plt.close(fig)


def confusion_matrix_single(cms, list_models, out_dir, chosen_horizons, dataset):

    for imod, mod in enumerate(list_models):
        for ik, k in enumerate(chosen_horizons):

            fig, _ = plt.subplots(figsize=(10, 9))

            annot_kws = {
                'fontsize': 30,
                'fontweight': 'bold',
                'fontfamily': 'serif'
            }

            csm_norm = cms[imod, ik] / np.sum(cms[imod, ik], axis=1)[:, None]*100
            # print(mod, k)
            # print(csm_norm)
            ax = sb.heatmap(csm_norm, annot=True, cbar=True, fmt=".2f", cmap="Blues", annot_kws=annot_kws,
                       vmin=0, vmax=100,
                       # xticklabels=["D", "S", "U"],
                       # yticklabels=["D", "S", "U"],
                       )

            ax.set_yticklabels(["D", "S", "U"], size=30)
            ax.set_xticklabels(["D", "S", "U"], size=30)

            for t in ax.texts: t.set_text(t.get_text() + "%")

            dataset_name = "FI-2010" if cst.DatasetFamily.FI == dataset else dataset.name
            wind_value = k[1].value if cst.DatasetFamily.LOB == dataset else k.value
            fig.suptitle('{} K={} ({})'.format(mod, wind_value, dataset_name), fontsize=30, fontweight="bold")

            fig.supylabel('Real', fontsize=30)
            fig.supxlabel('Predicted', fontsize=30)

            fig.tight_layout()

            wind_id = k[1].name if cst.DatasetFamily.LOB == dataset else k.name
            print("OUT", out_dir + "cm-{}-{}.pdf")
            fig.savefig(out_dir + "cm-{}-{}.pdf".format(mod, wind_id))
            # plt.show()
            plt.clf()
            plt.close(fig)


def scatter_plot_year(met_name, met_data, list_models, list_models_years, out_dir, dataset_type):
    X = list_models_years
    met_data = np.average(met_data, axis=0)  # seeds

    fig, ax = plt.subplots(figsize=(9, 7))
    df = pd.DataFrame(dict(id=list_models_years, data=met_data))
    maxes = df.groupby('id')['data'].max()

    plt.plot(maxes.index, maxes, color="red", label="max")
    plt.scatter(list_models_years, met_data, color="red")

    for label, x, y in zip(list_models, X, met_data):
        xytext = (+10, -10)
        print(label)
        if label.name == "TLONBoF":
            xytext = (-30, -20)
        if label.name == "DLA":
            xytext = (+15, +20)
        elif label.name == "ATNBoF":
            xytext = (+15, +15)
        elif label.name == "AXIALLOB":
            xytext = (-30, -20)
        elif label.name == "CNN":
            xytext = (+15, +20)
        elif label.name == "MAJORITY":
            xytext = (-50, -20)
        elif label.name == "METALOB":
            xytext = (-60, +20)

        plt.annotate(
            label.name,
            fontsize=15,
            xy = (x, y), xytext = xytext,
            textcoords = 'offset points', ha = 'left', va = 'top',
            bbox = dict(boxstyle = 'round, pad=.2', fc = 'red', alpha = 0.3),
            arrowprops = dict(arrowstyle = 'wedge', connectionstyle = 'arc3, rad=0'))

    coef = np.polyfit(df["id"], df["data"], 1)
    poly1d_fn = np.poly1d(coef)
    plt.plot(list_models_years, poly1d_fn(list_models_years), '--', color="blue", label="linear fit")
    SLOPE = '%.2E' % Decimal(coef[0])

    plt.xlabel('Year')
    plt.ylabel(met_name)
    plt.xticks([int(i) for i in maxes.index])
    plt.legend(fontsize=20, loc="upper left")
    plt.title("{} {} (slope={})".format(dataset_type, met_name, SLOPE))
    plt.tight_layout()
    plt.savefig(out_dir + "year-" + met_name + ".pdf")
    # plt.show()
    plt.close()

# ADJUST


def plot_agreement_matrix(list_models, fw_win, preds, out_dir):
    fig, ax = plt.subplots(figsize=(30, 30))

    data = {}
    for imod, mod in enumerate(list_models):
        data[mod] = preds[:, imod]

    agreement_matrix = np.zeros((len(list_models), len(list_models)))
    list_names = [m.name for m in list_models]
    for i in range(len(list_models)):
        for j in range(len(list_models)):
            agr = 0
            for pred in range(preds.shape[0]):
                if preds[pred, i] == preds[pred, j]:
                    agr += 1
            agreement_matrix[i, j] = agr / preds.shape[0]

    heatmap = sb.heatmap(agreement_matrix * 100, annot=True, fmt=".1f", yticklabels=list_names, xticklabels=list_names, vmin=0, vmax=100,)
    for t in heatmap.texts: t.set_text(t.get_text() + "%")

    heatmap.set(title=f'Agreement matrix for K={fw_win}')
    # heatmap.figure.set_size_inches(20, 20)
    heatmap.figure.savefig(out_dir + f"agreement_matrix_K={fw_win}.pdf", bbox_inches='tight')
    # plt.show()
    plt.close()


def FI_plots():
    """ Make FI-2010 plots. """

    PATH = "final_data/FI-2010-TESTS/jsons/"
    OUT = "final_data/FI-2010-TESTS/all-pdfs/"

    DATASET = cst.DatasetFamily.FI

    train_src = "FI"
    test_src = "FI"
    time_period = "FI"

    LIST_SEEDS = [500, 501, 502, 503, 504]

    LIST_MODELS = cst.MODELS_17
    LIST_YEARS = [cst.MODELS_YEAR_DICT[m] for m in cst.MODELS_YEAR_DICT if m in LIST_MODELS]

    os.makedirs(OUT, exist_ok=True)
    setup_plotting_env()

    metrics = metrics_to_plot(test_src)

    LIST_HORIZONS = cst.FI_Horizons
    MAT_REP = reproduced_metrics(PATH, metrics, LIST_MODELS, LIST_HORIZONS, LIST_SEEDS,
                                 dataset_type=cst.DatasetFamily.FI,
                                 train_src=train_src, test_src=test_src, time_period=time_period, jolly_seed=None)

    MAT_ORI = original_metrics(metrics, LIST_MODELS, LIST_HORIZONS)

    # n: PLOT 1
    for imet, met in enumerate(metrics):
        print("plot done perf", met)
        met_name = metrics[met]
        mid = map_id_metric_declared(metrics, met)
        ori = MAT_ORI[:, :, mid] if mid is not None else None

        metrics_vs_models_bars(met_name, LIST_HORIZONS, MAT_REP[:, :, :, imet], OUT, LIST_MODELS, dataset_type=DATASET,
                               met_vec_original=ori)  # each mat has shape MODELS x K x METRICA

    LIST_HORIZONS = cst.FI_Horizons
    MAT_REP = reproduced_metrics(PATH, metrics, LIST_MODELS, LIST_HORIZONS, LIST_SEEDS,
                                 dataset_type=cst.DatasetFamily.FI,
                                 train_src=train_src, test_src=test_src, time_period=time_period, jolly_seed=None)
    print("QUIIII")
    print("Models performance:")
    MOD = LIST_MODELS
    AVG = np.average(MAT_REP[:, :, :, 0], axis=0) * 100
    STD = np.max(np.std(MAT_REP[:, :, :, 0], axis=0) * 100, axis=1)
    CLAIM = np.nanmean(MAT_ORI[:, :, 0], axis=1)
    # print(AVG)
    # print(STD)
    # print(LIST_MODELS)
    print("DOVREI 2")
    INFER = inference_data(PATH, LIST_MODELS, dataset_type=cst.DatasetFamily.FI, train_src=train_src,
                           test_src=test_src, time_period=time_period)
    print(INFER[0])
    for im, m in enumerate(LIST_MODELS):
        print("MODEL {} VALUE {:e}".format(m, INFER[0][im]))
    print("QUIIII")

    # 1: PLOT 2
    # 1: PLOT 2
    # passing CM 15 x 5 x 3 x 3

    # ved = list(reversed(sorted(AVG)))
    #
    # for im, m in enumerate(MOD):
    #     formato = "{}\t&${} \pm {}$\t&{}\t&{}\t&{}\t&${}\pm {}$\t&{}\t&{} \\\\ \n \hline".format(m.name,
    #                                                                              round(AVG[im], 1),
    #                                                                              round(STD[im], 1),
    #                                                                              ved.index(AVG[im])+1,
    #                                                                              round(CLAIM[im], 1),
    #                                                                              Decimal(str(round(AVG[im]-CLAIM[im],1))).normalize(),
    #                                                                              None,
    #                                                                              None,
    #                                                                              1,
    #                                                                              None
    #                                                                              )
    #
    #     print(formato)
    #
    #
    # print("QUI1")
    # for imet, met in enumerate(metrics):
    #     print("plot done perf", met)
    #     met_name = metrics[met]
    #     metrics_vs_models_k_line(met_name, LIST_HORIZONS, MAT_REP[:, :, :, imet], OUT, LIST_MODELS, DATASET, type="var-for")

    CMS = confusion_metrix(PATH, LIST_MODELS, LIST_HORIZONS, LIST_SEEDS, jolly_seed=None, dataset_type=cst.DatasetFamily.FI,
                           train_src=train_src, test_src=test_src, time_period=time_period)

    INFER = inference_data(PATH, LIST_MODELS, dataset_type=cst.DatasetFamily.FI, train_src=train_src,
                           test_src=test_src, time_period=time_period)

    # 1: PLOT 2
    # 1: PLOT 2
    # passing CM 15 x 5 x 3 x 3
    print("QUI2")
    confusion_matrix_single(CMS[0, :], LIST_MODELS, OUT, LIST_HORIZONS, DATASET)

    # plot_inference_time(INFER[0], INFER[1], cst.MODELS_15, OUT)

    # n: PLOT 3
    # for imet, met in enumerate(metrics):
    #     met_name = metrics[met]
    #     met_data = np.mean(MAT_REP[:, :, :, imet], axis=2)  # MODELS x K x METRICA
    #     scatter_plot_year(met_name, met_data, LIST_MODELS, LIST_YEARS, OUT, DATASET)
    #     print("plot done year", met)
    #
    # logits, pred = MetaDataBuilder.load_predictions_from_jsons(PATH,
    #                                                            cst.DatasetFamily.FI,
    #                                                            cst.TRAINABLE_16,
    #                                                            500,
    #                                                            cst.FI_Horizons.K5.value,
    #                                                            trst=train_src,
    #                                                            test=test_src,
    #                                                            peri=time_period,
    #                                                            bw=None,
    #                                                            fw=None,
    #                                                            is_raw=True,
    #                                                            is_ignore_deeplobatt=False)
    #
    # plot_agreement_matrix(cst.TRAINABLE_16, 5, pred, OUT)

def lobster_stocks_plots():
    """ Make FI-2010 plots. """

    time_period = cst.Periods.JULY2021.name
    month = 'JUL' if time_period == cst.Periods.JULY2021.name else 'FEB'

    PATH = f"final_data/LOBSTER-{month}-TESTS/jsons/"

    # HERE FOR PER STOCK
    LIST_SEEDS = [500]
    LIST_MODELS = cst.MODELS_15
    LIST_STOCKS = ["ALL", "SOFI", "NFLX", "CSCO", "WING", "SHLS", "LSTR"]  # "ALL","SOFI","NFLX","CSCO", "WING", "SHLS"
    # LOB
    MAT_REP = reproduced_metrics_stocks(PATH, LIST_MODELS, LIST_STOCKS, LIST_SEEDS,
                                        dataset_type=cst.DatasetFamily.LOB,
                                        train_src="ALL", time_period=time_period, jolly_seed=None, target_horizon=cst.WinSize.EVENTS5)

    print(MAT_REP.shape)
    src_stock = "ALL"
    metrics = metrics_to_plot(src_stock)

    OUT = f"final_data/LOBSTER-{month}-TESTS/pdfs/all-pdfs-{src_stock}/"
    DATASET = cst.DatasetFamily.LOB

    for imet, met in enumerate(metrics):
        met_name = metrics[met]
        metrics_vs_models_bars(met_name, LIST_STOCKS, MAT_REP[:, :, :, imet], OUT, LIST_MODELS, dataset_type=DATASET, is_stocks=True)  # each mat has shape MODELS x K x METRICA
        print("plot done perf", met)


def lobster_plots():
    """ Make FI-2010 plots. """

    time_period = cst.Periods.JULY2021.name
    month = 'JUL' if time_period == cst.Periods.JULY2021.name else 'FEB'

    PATH = f"final_data/LOBSTER-{month}-TESTS/jsons/"
    ALL_STOCK_NAMES = ["ALL"] #, "SOFI", "NFLX", "CSCO", "WING", "SHLS"]
    CMS = []

    for sto in ALL_STOCK_NAMES:
        plt.close('all')
        del CMS

        OUT = f"final_data/LOBSTER-{month}-TESTS/pdfs/all-pdfs-{sto}/"
        DATASET = cst.DatasetFamily.LOB

        train_src = "ALL"
        test_src = sto

        LIST_SEEDS = [500]

        # LIST_MODELS = [m for m in cst.MODELS_17 if (sto == 'ALL' or m not in [cst.Models.MAJORITY, cst.Models.METALOB])]
        # LIST_MODELS = [m for m in cst.MODELS_15 if m not in [cst.Models.AXIALLOB, cst.Models.ATNBoF]]
        LIST_MODELS = cst.MODELS_17

        LIST_YEARS = [cst.MODELS_YEAR_DICT[m] for m in cst.MODELS_YEAR_DICT if m in LIST_MODELS]

        os.makedirs(OUT, exist_ok=True)

        metrics = metrics_to_plot(test_src)

        setup_plotting_env()

        backwards = [cst.WinSize.EVENTS1, cst.WinSize.EVENTS1, cst.WinSize.EVENTS1]
        forwards = [cst.WinSize.EVENTS1, cst.WinSize.EVENTS5, cst.WinSize.EVENTS10]
        LIST_HORIZONS = list(zip(backwards, forwards))  # cst.FI_Horizons

        # LOB
        MAT_REP = reproduced_metrics(PATH, metrics, LIST_MODELS, LIST_HORIZONS, LIST_SEEDS, dataset_type=cst.DatasetFamily.LOB,
                                     train_src=train_src, test_src=test_src, time_period=time_period, jolly_seed=None)

        # n1: PLOT with 3 bars
        for imet, met in enumerate(metrics):
            met_name = metrics[met]
            metrics_vs_models_bars(met_name, LIST_HORIZONS, MAT_REP[:, :, :, imet], OUT, LIST_MODELS, dataset_type=DATASET)  # each mat has shape MODELS x K x METRICA
            print("plot done perf", met)

        backwards = [cst.WinSize.EVENTS1, cst.WinSize.EVENTS1, cst.WinSize.EVENTS1, cst.WinSize.EVENTS1, cst.WinSize.EVENTS1]
        forwards  = [cst.WinSize.EVENTS1, cst.WinSize.EVENTS2, cst.WinSize.EVENTS3, cst.WinSize.EVENTS5, cst.WinSize.EVENTS10]
        LIST_HORIZONS = list(zip(backwards, forwards))  # cst.FI_Horizons

        # LOB
        MAT_REP = reproduced_metrics(PATH, metrics, LIST_MODELS, LIST_HORIZONS, LIST_SEEDS,
                                     dataset_type=cst.DatasetFamily.LOB,
                                     train_src=train_src, test_src=test_src, time_period=time_period, jolly_seed=None)

        # n2: PLOT with 5 lines
        for imet, met in enumerate(metrics):
            met_name = metrics[met]
            metrics_vs_models_k_line(met_name, LIST_HORIZONS, MAT_REP[:, :, :, imet], OUT, LIST_MODELS, DATASET, type="var-for")
            metrics_vs_models_bars(met_name, LIST_HORIZONS, MAT_REP[:, :, :, imet], OUT, LIST_MODELS, dataset_type=DATASET)  # each mat has shape MODELS x K x METRICA
            print("plot done perf", met)

        print("Models performance:")
        print(np.average(MAT_REP[:, :, :, 0], axis=0) * 100)

        # 5 x 15 x 5 x 3 x 3
        CMS = confusion_metrix(PATH, LIST_MODELS, LIST_HORIZONS, LIST_SEEDS, jolly_seed=None,
                               dataset_type=cst.DatasetFamily.LOB, train_src=train_src, test_src=test_src,
                               time_period=time_period)

        #
        # # 1: PLOT 2
        # # passing CM 15 x 5 x 3 x 3
        confusion_matrix_single(CMS[0, :], LIST_MODELS, OUT, LIST_HORIZONS, DATASET)
        #
        # if sto == 'ALL':
        #     INFER = inference_data(PATH, LIST_MODELS, dataset_type=cst.DatasetFamily.LOB, train_src=train_src,
        #                            test_src=test_src, time_period=time_period)
        #     plot_inference_time(INFER[0], INFER[1], cst.MODELS_15, OUT)
        #
        # # n: PLOT 3
        # for imet, met in enumerate(metrics):
        #     met_name = metrics[met]
        #     met_data = np.mean(MAT_REP[:, :, :, imet], axis=2)  # MODELS x K x METRICA
        #     scatter_plot_year(met_name, met_data, LIST_MODELS, LIST_YEARS, OUT, DATASET)
        #     print("plot done year", met)

        # agreement_stocks = cst.TRAINABLE_16 if sto == 'ALL' else cst.MODELS_15

        # logits, pred = MetaDataBuilder.load_predictions_from_jsons(
        #     PATH,
        #     cst.DatasetFamily.LOB,
        #     agreement_stocks,
        #     500,
        #     cst.FI_Horizons.K10.value,
        #     trst=train_src,
        #     test=test_src,
        #     peri=time_period,
        #     bw=cst.WinSize.EVENTS1.value,
        #     fw=cst.WinSize.EVENTS5.value,
        #     is_raw=True,
        #     is_ignore_deeplobatt=False
        # )
        # plot_agreement_matrix(agreement_stocks, 5, pred, OUT)



def metrics_to_plot(test_src):
    metrics = {'testing_{}_f1'.format(test_src)         :'F1 Score (%)',
               'testing_{}_f1_w'.format(test_src)       :'Weighted F1 Score (%)',
               'testing_{}_precision'.format(test_src)  :'Precision (%)',
               'testing_{}_precision_w'.format(test_src):'Weighted Precision (%)',
               'testing_{}_recall'.format(test_src)     :'Recall (%)',
               'testing_{}_recall_w'.format(test_src)   :'Weighted Recall (%)',
               'testing_{}_accuracy'.format(test_src)   :'Accuracy (%)',
               'testing_{}_mcc'.format(test_src)        :'MCC',
               }
    return metrics


def map_id_metric_declared(metrics_dict, metric):
    # in cst.DECLARED_PERF
    list_values = list(metrics_dict.values())
    metrics_name = ['F1 Score (%)', 'Precision (%)', 'Recall (%)', 'Accuracy (%)', 'MCC']
    assert set(metrics_name).intersection(set(list_values)) == set(metrics_name)

    if metrics_dict[metric] == 'F1 Score (%)':
        return 0
    elif metrics_dict[metric] == 'Precision (%)':
        return 1
    elif metrics_dict[metric] == 'Recall (%)':
        return 2
    elif metrics_dict[metric] == 'Accuracy (%)':
        return 3
    elif metrics_dict[metric] == 'MCC':
        return 4

    return None


def perf_table():
    import matplotlib.cm as cm

    PATH = "final_data/FI-2010-TESTS/jsons/"
    OUT = "final_data/FI-2010-TESTS/all-pdfs/"

    DATASET = cst.DatasetFamily.FI

    train_src = "FI"
    test_src = "FI"
    time_period = "FI"

    LIST_SEEDS = [500, 501, 502, 503, 504]

    LIST_MODELS = cst.MODELS_17

    os.makedirs(OUT, exist_ok=True)
    setup_plotting_env()

    metrics = metrics_to_plot(test_src)

    LIST_HORIZONS = cst.FI_Horizons
    MAT_ORI = original_metrics(metrics, LIST_MODELS, LIST_HORIZONS)

    LIST_HORIZONS = cst.FI_Horizons
    MAT_REP = reproduced_metrics(PATH, metrics, LIST_MODELS, LIST_HORIZONS, LIST_SEEDS,
                                 dataset_type=cst.DatasetFamily.FI,
                                 train_src=train_src, test_src=test_src, time_period=time_period, jolly_seed=None)

    print("Models performance:")
    BOOL = np.where(np.isnan(MAT_ORI[:, :, 0]) == 1, np.nan, 1)  # np.nan USELESS FOR RUN_NAME_PREFIX

    MOD = LIST_MODELS
    AVG = np.nanmean(MAT_REP[:, :, :, 0], axis=(0, 2)) * 100
    STD = np.nanstd(MAT_REP[:, :, :, 0], axis=(0, 2)) * 100

    CLAIM_AVG = np.nanmean(MAT_ORI[:, :, 0], axis=1)
    CLAIM_STD = np.nanstd(MAT_ORI[:, :, 0], axis=1)
    print(CLAIM_AVG)
    ved = list(reversed(sorted(AVG)))

    DISTS = MAT_REP[:, :, :, 0] * 100 - MAT_ORI[:, :, 0]
    DISTS_AVG = np.nanmean(DISTS, axis=(0, 2))
    DISTS_STD = np.nanstd(DISTS, axis=(0, 2))
    print(DISTS_AVG)
    print(DISTS_STD)
    ROBUSTNESS_SCORE = -np.abs(DISTS_AVG) - DISTS_STD
    ROBUSTNESS_SCORE = 100 + ROBUSTNESS_SCORE
    print(ROBUSTNESS_SCORE)

    # RUN_NAME_PREFIX GEN FEB
    train_src = "ALL"
    test_src = "ALL"
    time_period = cst.Periods.FEBRUARY2022.name
    LIST_SEEDS = [500]
    metrics = metrics_to_plot(test_src)
    month = 'JUL' if time_period == cst.Periods.JULY2021.name else 'FEB'

    PATH = f"final_data/LOBSTER-{month}-TESTS/jsons/"

    backwards = [cst.WinSize.EVENTS1, cst.WinSize.EVENTS1, cst.WinSize.EVENTS1, cst.WinSize.EVENTS1, cst.WinSize.EVENTS1]
    forwards = [cst.WinSize.EVENTS1, cst.WinSize.EVENTS2, cst.WinSize.EVENTS3, cst.WinSize.EVENTS5, cst.WinSize.EVENTS10]
    LIST_HORIZONS = list(zip(backwards, forwards))  # cst.FI_Horizons

    # LOB
    MAT_REP_FEB = reproduced_metrics(PATH, metrics, LIST_MODELS, LIST_HORIZONS, LIST_SEEDS,
                                     dataset_type=cst.DatasetFamily.LOB,
                                     train_src=train_src, test_src=test_src, time_period=time_period, jolly_seed=None)

    AVG_FEB = np.nanmean(MAT_REP_FEB[:, :, :, 0], axis=(0, 2)) * 100
    STD_FEB = np.nanstd(MAT_REP_FEB[:, :, :, 0], axis=(0, 2)) * 100
    ved_FEB = list(reversed(sorted(AVG_FEB)))

    DISTS_3 = MAT_REP_FEB[:, :, :, 0] * 100 - MAT_ORI[:, :, 0]
    DISTS_AVG_3 = np.nanmean(DISTS_3, axis=(0, 2))
    DISTS_STD_3 = np.nanstd(DISTS_3, axis=(0, 2))
    GENERALIZATION_SCORE_2 = -np.abs(DISTS_AVG_3) - DISTS_STD_3
    GENERALIZATION_SCORE_2 = 100+GENERALIZATION_SCORE_2

    # RUN_NAME_PREFIX GEN FEB
    train_src = "ALL"
    test_src = "ALL"
    time_period = cst.Periods.JULY2021.name
    LIST_SEEDS = [500, 501, 502, 503, 504]
    metrics = metrics_to_plot(test_src)
    month = 'JUL' if time_period == cst.Periods.JULY2021.name else 'FEB'

    PATH = f"final_data/LOBSTER-{month}-TESTS/jsons/"

    backwards = [cst.WinSize.EVENTS1, cst.WinSize.EVENTS1, cst.WinSize.EVENTS1, cst.WinSize.EVENTS1, cst.WinSize.EVENTS1]
    forwards = [cst.WinSize.EVENTS1, cst.WinSize.EVENTS2, cst.WinSize.EVENTS3, cst.WinSize.EVENTS5, cst.WinSize.EVENTS10]
    LIST_HORIZONS = list(zip(backwards, forwards))  # cst.FI_Horizons

    # LOB
    MAT_REP_JUL = reproduced_metrics(PATH, metrics, LIST_MODELS, LIST_HORIZONS, LIST_SEEDS,
                                     dataset_type=cst.DatasetFamily.LOB,
                                     train_src=train_src, test_src=test_src, time_period=time_period, jolly_seed=None)

    AVG_JUL = np.nanmean(MAT_REP_JUL[:, :, :, 0], axis=(0, 2)) * 100
    print(np.min(AVG_JUL), np.max(AVG_JUL))

    STD_JUL = np.nanstd(MAT_REP_JUL[:, :, :, 0], axis=(0, 2)) * 100
    ved_JUL = list(reversed(sorted(AVG_JUL)))

    DISTS_2 = MAT_REP_JUL[:, :, :, 0] * 100 - MAT_ORI[:, :, 0]
    DISTS_AVG_2 = np.nanmean(DISTS_2, axis=(0, 2))
    DISTS_STD_2 = np.nanstd(DISTS_2, axis=(0, 2))
    GENERALIZATION_SCORE = -np.abs(DISTS_AVG_2) - DISTS_STD_2
    GENERALIZATION_SCORE = 100+GENERALIZATION_SCORE

    def get_hot_scale(value, max=0, min=100):
        normalized_value = (value - min) / (max-min)  # Normalize the value between 0 and 1
        r, g, b, _ = cm.RdYlGn(normalized_value)
        return r, g, b

    score_min, score_max = min(list(ROBUSTNESS_SCORE) + list(GENERALIZATION_SCORE) + list(GENERALIZATION_SCORE_2)), max(list(ROBUSTNESS_SCORE) + list(GENERALIZATION_SCORE) + list(GENERALIZATION_SCORE_2))
    for im, m in enumerate(MOD):
        signed_1 = round(ROBUSTNESS_SCORE[im], 1)  # round(AVG[im] - CLAIM_AVG[im], 1)
        signed_2 = round(GENERALIZATION_SCORE[im], 1)
        signed_3 = round(GENERALIZATION_SCORE_2[im], 1)

        signed_1 = "" + str(signed_1) if signed_1 > 0 else str(signed_1)
        signed_2 = "" + str(signed_2) if signed_2 > 0 else str(signed_2)
        signed_3 = "" + str(signed_3) if signed_3 > 0 else str(signed_3)

        if str(signed_1) != 'nan':
            rgb_1 = str(get_hot_scale(float(signed_1), score_max, score_min))[1:-1]
            rgb_1 = "\cellcolor[rgb]{{{}}}".format(rgb_1)
        else:
            signed_1 = "-"
            rgb_1 = ""

        if str(signed_2) != 'nan':
            rgb_2 = str(get_hot_scale(float(signed_2), score_max, score_min))[1:-1]
            rgb_2 = "\cellcolor[rgb]{{{}}}".format(rgb_2)
        else:
            signed_2 = "-"
            rgb_2 = ""

        if str(signed_3) != 'nan':
            rgb_3 = str(get_hot_scale(float(signed_3), score_max, score_min))[1:-1]
            rgb_3 = "\cellcolor[rgb]{{{}}}".format(rgb_3)
        else:
            signed_3 = "-"
            rgb_3 = ""

        claim = round(CLAIM_AVG[im], 1)
        claim = "-" if str(claim) == 'nan' else claim

        claim_std = round(CLAIM_STD[im], 1)

        formato = "{}\t&${} \pm {}$\t&${} \pm {}$\t&${}$\t&{}${}$\t&${}\pm {}$\t&${}$\t&{}${}$&${}\pm {}$\t&${}$\t&{}${}$ \\\\ \n \hline".format(m.name,
                                                                                                               claim,
                                                                                                               claim_std,
                                                                                                 round(AVG[im], 1),
                                                                                                 round(STD[im], 1),
                                                                                                 ved.index(AVG[im]) + 1,

                                                                                                 rgb_1,
                                                                                                 signed_1,
                                                                                                 round(AVG_JUL[im], 1),
                                                                                                 round(STD_JUL[im], 1),
                                                                                                 ved_JUL.index(AVG_JUL[im]) + 1,
                                                                                                 rgb_2,
                                                                                                 signed_2,
                                                                                                 round(AVG_FEB[im], 1),
                                                                                                 round(STD_FEB[im], 1),
                                                                                                 ved_FEB.index(AVG_FEB[im]) + 1,
                                                                                                 rgb_3,
                                                                                                 signed_3,
                                                                                                 )

        print(formato)


if __name__ == '__main__':

    lobster_plots()
    # FI_plots()
    # perf_table()
