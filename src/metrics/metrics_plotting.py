
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


def setup_plotting_env():
    plt.rcParams["figure.figsize"] = [16, 9]
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["font.size"] = 20
    plt.rcParams["axes.labelsize"] = 20
    plt.rcParams["axes.titlesize"] = 24
    plt.rcParams["xtick.labelsize"] = 16
    plt.rcParams["ytick.labelsize"] = 16
    plt.rcParams["font.family"] = "serif"


def reproduced_metrics(path, metrics, list_models, list_horizons, list_seeds, jolly_seed=None):
    METRICS = np.zeros(shape=(len(list_seeds), len(list_models), len(list_horizons), len(metrics)))

    for imod, mod in enumerate(list_models):
        for ik, k in enumerate(list_horizons):
            for iss, s in enumerate(list_seeds):
                fname = "model={}-seed={}-trst=FI-test=FI-data={}-peri=FI-bw=None-fw=None-fiw={}.json".format(mod.name, s, cst.model_dataset(mod), k.value)

                if mod == cst.Models.MAJORITY:  # only seed for the baseline
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


def inference_data(path, list_models, seed=502, horizon=10):
    INFERENCE = np.zeros(shape=(2, len(list_models)))
    for imod, mod in enumerate(list_models):
        if mod == cst.Models.MAJORITY:
            continue

        fname = "model={}-seed={}-trst=FI-test=FI-data={}-peri=FI-bw=None-fw=None-fiw={}.json".format(mod.name, seed, cst.model_dataset(mod), horizon)
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
                    METRICS_ORI[imod, ik, imet] = cst.DECLARED_PERF[cst.Models[mod.name]][ik][imet]
                else:
                    METRICS_ORI[imod, ik, imet] = None
    return METRICS_ORI


def confusion_metrix(path, list_models, list_seeds, jolly_seed=None):
    CMS = np.zeros(shape=(len(list_seeds), len(list_models), len(cst.FI_Horizons), 3, 3))

    for imod, mod in enumerate(list_models):
        for ik, k in enumerate(cst.FI_Horizons):
            for iss, s in enumerate(list_seeds):
                fname = "model={}-seed={}-trst=FI-test=FI-data={}-peri=FI-bw=None-fw=None-fiw={}.json".format(mod.name, s, cst.model_dataset(mod), k.value)

                if mod == cst.Models.MAJORITY:  # only seed for the baseline
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


def metrics_vs_models_k(met_name, met_vec, out_dir, list_models, met_vec_original=None):

    labels = ["K=" + str(k.value) for k in cst.FI_Horizons]

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

    fig, ax = plt.subplots(figsize=(19, 9))

    indsxs = np.array(range(1, int(len(list_models) / 2) + 1))
    zero = [0] if len(list_models) % 2 == 1 else []
    ranges = list(reversed(indsxs * -1)) + zero + list(indsxs)

    R = []  # the bars
    for iri, ri in enumerate(list_models):
        r_bar_i = ax.bar(x + width * ranges[iri], avg_met[iri, :], width, yerr=std_met[iri, :], label=ri.name,
                         color=util.sample_color(iri, "tab20"), align='center')
        R += [r_bar_i]

    diffp = ((avg_met - met_vec_original) / met_vec_original * 100)  # text of the diff
    for iri, ri in enumerate(R):

        diffp_show = []
        for i, val in enumerate(diffp[iri, :]):
            our = round(avg_met[iri, i]) if "MCC" not in met_name else round(avg_met[iri, i], 2)
            if not np.isnan(val):
                diffp_show += ["{}% (".format(our) + "{0:+}%)".format(round(val))]
            else:
                diffp_show += ["{}% ($\cdot$)".format(our)]
            if "MCC" in met_name:
                diffp_show = [s.replace('%', '') for s in diffp_show]

        ax.bar_label(ri, labels=diffp_show, padding=3, fmt=fmt, rotation=90, fontsize=10)

    if met_vec_original is not None:
        for iri, ri in enumerate(list_models):
            label = 'original' if iri == 0 else ''
            ax.bar(x + width * ranges[iri], met_vec_original[iri, :], width, alpha=1, bottom=0, fill=False,
                   edgecolor='black', label=label, align='center')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(met_name)
    ax.set_title("FI-2010 " + met_name)
    ax.set_xticks(x, labels)  # , rotation=0, ha="right", rotation_mode="anchor")

    ax.legend(fontsize=12, ncol=6, handleheight=2, labelspacing=0.05)

    plt.ylim(miny, maxy)
    fig.tight_layout()

    plt.savefig(out_dir + "ka" + met_name + ".pdf")
    plt.show()
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
    plt.show()
    plt.close(fig)


def confusion_matrices(cms, list_models, out_dir):
    fig = plt.figure(figsize=(30, 30 * 2.5))
    gs = gridspec.GridSpec(len(list_models), len(cst.FI_Horizons) + 1,
                           width_ratios=[6 for _ in range(len(cst.FI_Horizons))] + [1], figure=fig)

    annot_kws = {
        'fontsize': 16,
        'fontweight': 'bold',
        'fontfamily': 'serif'
    }

    for imod, mod in enumerate(list_models):
        for ik, k in enumerate(cst.FI_Horizons):
            cbar = False  # ik == 0 and imod == 0
            csm_norm = cms[imod, ik] / np.sum(cms[imod, ik], axis=1)[:, None]

            axi = fig.add_subplot(gs[imod, ik])
            sb.heatmap(csm_norm, annot=True, ax=axi, cbar=cbar, fmt=".2%", cmap="Blues", annot_kws=annot_kws)

            axi.set_xticklabels([p.name for p in cst.Predictions], rotation=0, fontsize=12)
            axi.set_yticklabels([p.name for p in cst.Predictions], rotation=90, fontsize=12)

            if imod == 0:
                axi.set_title("K={}".format(k.value), fontsize=30, fontweight="bold", pad=25)

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
    plt.show()
    plt.close(fig)


def scatter_plot_year(met_name, met_data, list_models, list_models_years, out_dir):
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
    plt.title("FI-2010 {} in the Years".format(met_name))
    plt.tight_layout()
    plt.savefig(out_dir + "year-" + met_name + ".pdf")
    plt.show()
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
    plt.show()
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
    plt.show()
    plt.close()


if __name__ == '__main__':

    PATH = "data/experiments/all_models_28_03_23/"
    OUT = "data/experiments/pdfs/"

    LIST_SEEDS = [500, 501, 502, 503, 504]
    LIST_HORIZONS = cst.FI_Horizons

    LIST_MODELS = cst.MODELS_17  # cst.TRAINABLE_16
    LIST_YEARS =  [cst.MODELS_YEAR_DICT[m] for m in cst.MODELS_YEAR_DICT if m in LIST_MODELS]

    os.makedirs(OUT, exist_ok=True)

    metrics = ['testing_FI_f1',
               'testing_FI_precision',
               'testing_FI_recall',
               'testing_FI_accuracy',
               'testing_FI_mcc'
               ]

    setup_plotting_env()

    MAT_REP = reproduced_metrics(PATH, metrics, LIST_MODELS, LIST_HORIZONS, LIST_SEEDS, jolly_seed=502)
    print("Models performance:")
    print(np.average(MAT_REP[:, :, :, 0], axis=0)*100)

    MAT_ORI = original_metrics(metrics, LIST_MODELS, LIST_HORIZONS)
    CMS = confusion_metrix(PATH, LIST_MODELS, LIST_SEEDS, jolly_seed=502)
    INFER = inference_data(PATH, LIST_MODELS)
    # r_imp = relative_improvement_table(MAT_REP, MAT_ORI, LIST_MODELS)

    # n: PLOT 1
    for imet, met in enumerate(cst.metrics_name):
        metrics_vs_models_k(met, MAT_REP[:, :, :, imet], OUT, LIST_MODELS, MAT_ORI[:, :, imet])  # each mat has shape MODELS x K x METRICA
        print("plot done perf", met)

    # 1: PLOT 2
    confusion_matrices(CMS[1, :], LIST_MODELS, OUT)
    print("plot done cm")

    # n: PLOT 3
    for imet, met in enumerate(cst.metrics_name):
        met_data = np.mean(MAT_REP[:, :, :, imet], axis=2)   # MODELS x K x METRICA
        scatter_plot_year(met, met_data, LIST_MODELS, LIST_YEARS, OUT)
        print("plot done year", met)

    plot_inference_time(INFER[0], INFER[1], cst.TRAINABLE_16, OUT)

    # 20923 is the number of instances because test set is a portion of the original
    logits, pred = MetaDataBuilder.load_predictions_from_jsons(cst.TRAINABLE_16, 502, cst.FI_Horizons.K10.value, n_instances=20923)

    plot_corr_matrix(cst.TRAINABLE_16, 10, pred, OUT)
    plot_agreement_matrix(cst.TRAINABLE_16, 10, pred, OUT)
