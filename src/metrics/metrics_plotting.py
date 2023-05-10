
from src.data_preprocessing.META.METADataBuilder import MetaDataBuilder

import matplotlib.pyplot as plt
import src.utils.utilities as util
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
    plt.rcParams["axes.titlesize"] = 24
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
                if dataset_type == cst.DatasetFamily.LOBSTER:
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


def inference_data(path, list_models, dataset_type, seed=500, horizon=10, bw=1, fw=5, train_src="ALL", test_src="ALL", time_period=cst.Periods.JULY2021.name):
    INFERENCE = np.zeros(shape=(2, len(list_models)))
    for imod, mod in enumerate(list_models):
        if mod == cst.Models.MAJORITY:
            continue

        if dataset_type == cst.DatasetFamily.LOBSTER:
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
    METRICS_ORI = np.zeros(shape=(len(list_models), len(cst.FI_Horizons), len(metrics)))

    for imod, mod in enumerate(list_models):
        for ik, k in enumerate(list_horizons):
            for imet, met in enumerate(cst.metrics_name):
                if mod in cst.DECLARED_PERF:
                    wid = cst.map_id_win(k)
                    METRICS_ORI[imod, ik, imet] = cst.DECLARED_PERF[cst.Models[mod.name]][wid][imet]
                else:
                    METRICS_ORI[imod, ik, imet] = None
    return METRICS_ORI


def confusion_metrix(path, list_models, list_horizons, list_seeds, dataset_type, train_src="ALL", test_src="ALL", time_period=cst.Periods.JULY2021.name, jolly_seed=None):
    CMS = np.zeros(shape=(len(list_seeds), len(list_models), len(list_horizons), 3, 3))

    for imod, mod in enumerate(list_models):
        for ik, k in enumerate(list_horizons):
            for iss, s in enumerate(list_seeds):

                bw, fw, fik = None, None, k
                if dataset_type == cst.DatasetFamily.LOBSTER:
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


def metrics_vs_models_k(met_name, horizons, met_vec, out_dir, list_models, dataset_type, chosen_horizons_mask, met_vec_original=None):

    horizons = np.array(horizons)
    if dataset_type == cst.DatasetFamily.FI:
        labels = ["K={}".format(k.value) for k in horizons[chosen_horizons_mask]]
    else:
        labels = ["K={}".format(fw.value) for bw, fw in horizons[chosen_horizons_mask]]

    fmt = "%.2f"
    miny, maxy = -1, 1.1

    if "MCC" not in met_name:
        met_vec = met_vec * 100
        fmt = "%d%%"
        miny, maxy = 0, 110

    avg_met = np.average(met_vec, axis=0)  # avg seeds
    std_met = np.std(met_vec, axis=0)

    x = np.arange(len(labels)) * 9  # the label locations
    width = 0.44  # the width of the bars

    fig, ax = plt.subplots(figsize=(17, 15))

    indsxs = np.array(range(1, int(len(list_models) / 2) + 1))
    zero = [0] if len(list_models) % 2 == 1 else []
    ranges = list(reversed(indsxs * -1)) + zero + list(indsxs)

    R = []  # the bars
    for iri, ri in enumerate(list_models):
        if dataset_type == cst.DatasetFamily.FI:
            r_bar_i = ax.bar(x + width * ranges[iri], avg_met[iri, chosen_horizons_mask], width, yerr=std_met[iri, chosen_horizons_mask], label=ri.name,
                             color=util.sample_color(iri, "tab20"), align='center')  # hatch=util.sample_pattern(iri))

            R += [r_bar_i]

        elif dataset_type == cst.DatasetFamily.LOBSTER:
            r_bar_i = ax.bar(x + width * ranges[iri], avg_met[iri, chosen_horizons_mask], width, yerr=std_met[iri, chosen_horizons_mask],
                             label=ri.name, color=util.sample_color(iri, "tab20"), align='center', edgecolor='black')  # hatch=util.sample_pattern(iri))

            bar_value = ["{}%".format(round(it, 2)) for it in avg_met[iri, chosen_horizons_mask]]
            ax.bar_label(r_bar_i, labels=bar_value, padding=3, fmt=fmt, rotation=90, fontsize=10)

    if met_vec_original is not None:
        diffp = ((avg_met - met_vec_original) / met_vec_original * 100)  # text of the diff TODO check seems a wrong diff
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

            ax.bar_label(ri, labels=diffp_show, padding=3, fmt=fmt, rotation=90, fontsize=15)

        for iri, ri in enumerate(list_models):
            label = 'original' if iri == 0 else ''  # black bars in FI
            ax.bar(x + width * ranges[iri], met_vec_original[iri, chosen_horizons_mask], width, alpha=1, bottom=0, fill=False,
                   edgecolor='black', label=label, align='center')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(met_name)

    dataset_name = dataset_type.name
    dataset_name = "FI-2010" if cst.DatasetFamily.FI == dataset_name else dataset_name

    ax.set_title("{}".format(dataset_name))
    ax.set_xticks(x, labels)  # , rotation=0, ha="right", rotation_mode="anchor")

    if dataset_type == cst.DatasetFamily.FI:
        ax.set_ylim((0, 100))
    elif dataset_type == cst.DatasetFamily.LOBSTER:
        ax.set_ylim((0, 70))

    ax.legend(fontsize=15, ncol=6, handleheight=2, labelspacing=0.05, loc="lower left", framealpha=1)

    # plt.ylim(miny, maxy)
    fig.tight_layout()

    plt.savefig(out_dir + "ka" + met_name + ".pdf")
    # plt.show()
    plt.close(fig)


def metrics_vs_models_k_line(met_name, horizons, met_vec, out_dir, list_models, dataset_type, chosen_horizons_mask, type=None):

    horizons = np.array(horizons)
    if dataset_type == cst.DatasetFamily.FI:
        labels = ["K={}".format(k.value) for k in horizons[chosen_horizons_mask]]
    else:
        labels = ["K={}".format(fw.value) for bw, fw in horizons[chosen_horizons_mask]]

    if "MCC" not in met_name:
        met_vec = met_vec * 100

    avg_met = np.average(met_vec, axis=0)  # avg seeds
    std_met = np.std(met_vec, axis=0)

    x = np.arange(len(labels))

    # fig, ax = plt.subplots(figsize=(17, 15))
    fig, ax = plt.subplots(figsize=(9, 7))

    for iri, ri in enumerate(list_models):
        ax.fill_between(x, std_met[iri, chosen_horizons_mask], -std_met[iri, chosen_horizons_mask], alpha=.3, linewidth=0, color=util.sample_color(iri, "tab20"))
        ax.plot(x, avg_met[iri, chosen_horizons_mask], label=ri.name, color=util.sample_color(iri, "tab20"), marker=util.sample_marker(iri))

    ax.relim()
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(met_name)
    ax.set_title("{}".format(dataset_type.name, met_name))
    ax.set_xticks(x, labels)

    ax.legend(fontsize=12, ncol=4, handleheight=2, labelspacing=0.05)

    fig.tight_layout()
    plt.savefig(out_dir + type + "-line" + met_name + ".pdf")
    # plt.show()
    plt.close(fig)


def plot_inference_time(met_vec, met_vec_err, list_models, out_dir):

    labels = ["K=10"]

    x = np.arange(len(labels)) * 9  # the label locations
    width = 0.5  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 9))

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

    ax.legend(fontsize=10, ncol=2, handleheight=6, labelspacing=0.05)

    # plt.ylim(miny, maxy)
    fig.tight_layout()

    plt.savefig(out_dir + "inference_time.pdf")
    # plt.show()
    plt.close(fig)


def confusion_matrices(cms, list_models, out_dir, chosen_horizons, dataset):
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


def scatter_plot_year(met_name, met_data, list_models, list_models_years, out_dir, dataset_type):
    X = list_models_years
    met_data = np.average(met_data, axis=0)  # seeds

    fig = plt.figure(figsize=(16, 9))
    df = pd.DataFrame(dict(id=list_models_years, data=met_data))
    maxes = df.groupby('id')['data'].max()

    plt.plot(maxes.index, maxes, color="red", label="max")
    plt.scatter(list_models_years, met_data, color="red")

    for label, x, y in zip(list_models, X, met_data):
        plt.annotate(
            label.name,
            fontsize=20,
            xy = (x, y), xytext = (+80, -40),
            textcoords = 'offset points', ha = 'right', va = 'bottom',
            bbox = dict(boxstyle = 'round, pad=0.5', fc = 'red', alpha = 0.3),
            arrowprops = dict(arrowstyle = 'wedge', connectionstyle = 'arc3, rad=0'))

    plt.xlabel('Year')
    plt.ylabel(met_name)
    plt.xticks([int(i) for i in maxes.index])
    plt.legend(fontsize=20, loc="upper left")
    plt.title("{} {} in the Years".format(dataset_type, met_name))
    plt.tight_layout()
    plt.savefig(out_dir + "year-" + met_name + ".pdf")
    # plt.show()
    plt.close()

# ADJUST

def plot_corr_matrix(list_models, fw_win, preds, out_dir):

    # collect data
    # models = sorted([model.name for model in cst.Models if (model not in [cst.Models.METALOB, cst.Models.ATNBoF])])
    #
    # # we swap the order of DeepLOBATT and DeepLOB, because in the json there is DEEPLOBATT first
    # models[8], models[9] = models[9], models[8]

    data = {}
    for imod, mod in enumerate(list_models):
        data[mod] = preds[:, imod]

    # form dataframe
    dataframe = pd.DataFrame(data, columns=list(data.keys()))

    # form correlation matrix
    corr_matrix = dataframe.corr()

    ticks = [m.name for m in list_models]
    heatmap = sb.heatmap(corr_matrix, annot=True, fmt=".2f", yticklabels=ticks, xticklabels=ticks)
    heatmap.set(title=f"Correlation matrix for K={fw_win}")

    heatmap.figure.set_size_inches(20, 20)

    # save heatmap as PNG file
    heatmap.figure.savefig(out_dir + f"correlation_matrix_k={fw_win}.pdf", bbox_inches='tight')
    # plt.show()
    plt.close()


def plot_agreement_matrix(list_models, fw_win, preds, out_dir):
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

    heatmap = sb.heatmap(agreement_matrix, annot=True, fmt=".2f", yticklabels=list_names, xticklabels=list_names)

    heatmap.set(title=f'Agreement matrix for K={fw_win}')
    heatmap.figure.set_size_inches(20, 20)
    heatmap.figure.savefig(out_dir + f"agreement_matrix_K={fw_win}.pdf", bbox_inches='tight')
    # plt.show()
    plt.close()


def FI_plots():
    """ Make FI-2010 plots. """

    PATH = "final_data/FI-2010-TESTS/jsons/"
    OUT = "final_data/FI-2010-TESTS/pdfs/"

    DATASET = cst.DatasetFamily.FI

    train_src = "FI"
    test_src = "FI"
    time_period = "FI"

    LIST_SEEDS = [500, 501, 502, 503, 504]

    LIST_MODELS = cst.MODELS_17
    LIST_YEARS = [cst.MODELS_YEAR_DICT[m] for m in cst.MODELS_YEAR_DICT if m in LIST_MODELS]

    os.makedirs(OUT, exist_ok=True)

    metrics = ['testing_{}_f1_w'.format(test_src),
               'testing_{}_precision_w'.format(test_src),
               'testing_{}_recall_w'.format(test_src),
               'testing_{}_accuracy'.format(test_src),
               'testing_{}_mcc'.format(test_src),
               ]

    setup_plotting_env()

    # LIST_HORIZONS = cst.FI_Horizons
    LIST_HORIZONS = [cst.FI_Horizons.K1, cst.FI_Horizons.K5, cst.FI_Horizons.K10]  # cst.FI_Horizons

    MAT_REP = reproduced_metrics(PATH, metrics, LIST_MODELS, LIST_HORIZONS, LIST_SEEDS,
                                 dataset_type=cst.DatasetFamily.FI,
                                 train_src=train_src, test_src=test_src, time_period=time_period, jolly_seed=None)

    print("Models performance:")
    print(np.average(MAT_REP[:, :, :, 0], axis=0) * 100)

    MAT_ORI = original_metrics(metrics, LIST_MODELS, LIST_HORIZONS)

    CMS = confusion_metrix(PATH, LIST_MODELS, LIST_HORIZONS, LIST_SEEDS, jolly_seed=None, dataset_type=cst.DatasetFamily.FI,
                           train_src=train_src, test_src=test_src, time_period=time_period)

    INFER = inference_data(PATH, LIST_MODELS, dataset_type=cst.DatasetFamily.FI, train_src=train_src,
                           test_src=test_src, time_period=time_period)

    # r_imp = relative_improvement_table(MAT_REP, MAT_ORI, LIST_MODELS)

    # n: PLOT 1
    for imet, met in enumerate(cst.metrics_name):
        print("plot done perf", met)
        chosen_horizons_mask = np.arange(len(LIST_HORIZONS))
        metrics_vs_models_k(met, LIST_HORIZONS, MAT_REP[:, :, :, imet], OUT, LIST_MODELS, dataset_type=DATASET,
                            chosen_horizons_mask=chosen_horizons_mask, met_vec_original=MAT_ORI[:, chosen_horizons_mask, imet])  # each mat has shape MODELS x K x METRICA

        chosen_horizons_mask = np.arange(len(LIST_HORIZONS))
        metrics_vs_models_k_line(met, LIST_HORIZONS, MAT_REP[:, :, :, imet], OUT, LIST_MODELS, DATASET, chosen_horizons_mask, type="var-for")

    # # 1: PLOT 2
    # chosen_horizons_mask = [0, 1, 2, 3, 4]
    # chosen_horizons = np.array(LIST_HORIZONS)[chosen_horizons_mask]
    # confusion_matrices(CMS[0, :], LIST_MODELS, OUT, chosen_horizons, DATASET)
    # print("plot done cm")
    #
    # # n: PLOT 3
    # for imet, met in enumerate(cst.metrics_name):
    #     met_data = np.mean(MAT_REP[:, :, :, imet], axis=2)  # MODELS x K x METRICA
    #     scatter_plot_year(met, met_data, LIST_MODELS, LIST_YEARS, OUT, DATASET)
    #     print("plot done year", met)
    #
    plot_inference_time(INFER[0], INFER[1], cst.MODELS_15, OUT)

    # 20923 is the number of instances because test set is a portion of the original
    # logits, pred = MetaDataBuilder.load_predictions_from_jsons(cst.TRAINABLE_16, 502, cst.FI_Horizons.K10.value, n_instances=20923)

    # plot_corr_matrix(cst.TRAINABLE_16, 10, pred, OUT)
    # plot_agreement_matrix(cst.TRAINABLE_16, 10, pred, OUT)


def lobster_plots():
    """ Make FI-2010 plots. """

    PATH = "final_data/LOBSTER-TESTS/jsons/"
    OUT = "final_data/LOBSTER-TESTS/pdfs/"

    DATASET = cst.DatasetFamily.LOBSTER

    train_src = "ALL"
    test_src = "ALL" if DATASET == cst.DatasetFamily.LOBSTER else "FI"
    time_period = cst.Periods.JULY2021.name

    LIST_SEEDS = [500, 501, 502, 503, 504]

    # backwards = [cst.WinSize.SEC100, cst.WinSize.SEC100, cst.WinSize.SEC100, cst.WinSize.SEC50, cst.WinSize.SEC50, cst.WinSize.SEC10]
    # forwards  = [cst.WinSize.SEC100, cst.WinSize.SEC50, cst.WinSize.SEC10, cst.WinSize.SEC50, cst.WinSize.SEC10, cst.WinSize.SEC10]

    backwards = [cst.WinSize.EVENTS1, cst.WinSize.EVENTS1, cst.WinSize.EVENTS1, cst.WinSize.EVENTS1, cst.WinSize.EVENTS1]
    forwards = [cst.WinSize.EVENTS1, cst.WinSize.EVENTS2, cst.WinSize.EVENTS3, cst.WinSize.EVENTS5, cst.WinSize.EVENTS10]
    LIST_HORIZONS = list(zip(backwards, forwards))  # cst.FI_Horizons

    LIST_MODELS = cst.MODELS_17
    # LIST_MODELS = [m for m in cst.MODELS_15 if m not in [cst.Models.AXIALLOB, cst.Models.ATNBoF]]
    LIST_YEARS = [cst.MODELS_YEAR_DICT[m] for m in cst.MODELS_YEAR_DICT if m in LIST_MODELS]

    os.makedirs(OUT, exist_ok=True)

    metrics = ['testing_{}_f1_w'.format(test_src),
               'testing_{}_precision_w'.format(test_src),
               'testing_{}_recall_w'.format(test_src),
               'testing_{}_accuracy'.format(test_src),
               'testing_{}_mcc'.format(test_src),
               ]

    setup_plotting_env()

    # LOBSTER
    backwards = [cst.WinSize.EVENTS1, cst.WinSize.EVENTS1, cst.WinSize.EVENTS1]
    forwards  = [cst.WinSize.EVENTS1, cst.WinSize.EVENTS5, cst.WinSize.EVENTS10]
    LIST_HORIZONS = list(zip(backwards, forwards))  # cst.FI_Horizons

    MAT_REP = reproduced_metrics(PATH, metrics, LIST_MODELS, LIST_HORIZONS, LIST_SEEDS, dataset_type=cst.DatasetFamily.LOBSTER,
                                 train_src=train_src, test_src=test_src, time_period=time_period, jolly_seed=None)

    print("Models performance:")
    print(np.average(MAT_REP[:, :, :, 0], axis=0) * 100)

    CMS = confusion_metrix(PATH, LIST_MODELS, LIST_HORIZONS, LIST_SEEDS, jolly_seed=None, dataset_type=cst.DatasetFamily.LOBSTER, train_src=train_src, test_src=test_src, time_period=time_period)
    INFER = inference_data(PATH, LIST_MODELS, dataset_type=cst.DatasetFamily.LOBSTER, train_src=train_src, test_src=test_src, time_period=time_period)
    # r_imp = relative_improvement_table(MAT_REP, MAT_ORI, LIST_MODELS)

    # n: PLOT 1
    for imet, met in enumerate(cst.metrics_name):
        chosen_horizons_mask = np.arange(len(LIST_HORIZONS))
        metrics_vs_models_k(met, LIST_HORIZONS, MAT_REP[:, :, :, imet], OUT, LIST_MODELS, dataset_type=DATASET, chosen_horizons_mask=chosen_horizons_mask)  # each mat has shape MODELS x K x METRICA

        chosen_horizons_mask = np.arange(len(LIST_HORIZONS))
        metrics_vs_models_k_line(met, LIST_HORIZONS, MAT_REP[:, :, :, imet], OUT, LIST_MODELS, DATASET, chosen_horizons_mask, type="var-for")

        # chosen_horizons_mask = [0, 1, 2, 3, 4]
        # metrics_vs_models_k_line(met, LIST_HORIZONS, MAT_REP[:, :, :, imet], OUT, LIST_MODELS, DATASET, chosen_horizons_mask, type="var-ba")

        print("plot done perf", met)

    # 1: PLOT 2
    # chosen_horizons_mask = np.arange(len(LIST_HORIZONS))
    # chosen_horizons = np.array(LIST_HORIZONS)[chosen_horizons_mask]
    confusion_matrices(CMS[0, :], LIST_MODELS, OUT, LIST_HORIZONS, DATASET)
    print("plot done cm")

    # n: PLOT 3
    for imet, met in enumerate(cst.metrics_name):
        met_data = np.mean(MAT_REP[:, :, :, imet], axis=2)  # MODELS x K x METRICA
        scatter_plot_year(met, met_data, LIST_MODELS, LIST_YEARS, OUT, DATASET)
        print("plot done year", met)

    plot_inference_time(INFER[0], INFER[1], LIST_MODELS, OUT)

    # # 20923 is the number of instances because test set is a portion of the original
    # logits, pred = MetaDataBuilder.load_predictions_from_jsons(cst.TRAINABLE_16, 502, cst.FI_Horizons.K10.value, n_instances=20923)
    #
    # plot_corr_matrix(cst.TRAINABLE_16, 10, pred, OUT)
    # plot_agreement_matrix(cst.TRAINABLE_16, 10, pred, OUT)


if __name__ == '__main__':

    # lobster_plots()
    FI_plots()
