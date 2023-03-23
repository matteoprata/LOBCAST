
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


def reproduced_metrics(path, metrics, list_models):
    METRICS = np.zeros(shape=(len(list_models), len(cst.FI_Horizons), len(metrics)))

    for imod, mod in enumerate(list_models):
        for ik, k in enumerate(cst.FI_Horizons):
            fname = "model={}-seed=0-trst=FI-test=FI-data=FI-peri=FI-bw=None-fw=None-fiw={}.json".format(mod.name, k.value)
            jsn = util.read_json(path + fname)
            for imet, met in enumerate(metrics):
                METRICS[imod, ik, imet] = jsn[met]
    return METRICS


def original_metrics(metrics, list_models):
    METRICS_ORI = np.zeros(shape=(len(list_models), len(cst.FI_Horizons), len(metrics)))

    for imod, mod in enumerate(list_models):
        for ik, k in enumerate(cst.FI_Horizons):
            for imet, met in enumerate(cst.metrics_name):
                if mod in cst.DECLARED_PERF:
                    METRICS_ORI[imod, ik, imet] = cst.DECLARED_PERF[cst.Models[mod.name]][ik][imet]
                else:
                    METRICS_ORI[imod, ik, imet] = None
    return METRICS_ORI


def confusion_metrix(path, list_models):
    CMS = np.zeros(shape=(len(list_models), len(cst.FI_Horizons), 3, 3))

    for imod, mod in enumerate(list_models):
        for ik, k in enumerate(cst.FI_Horizons):
            fname = "model={}-seed=0-trst=FI-test=FI-data=FI-peri=FI-bw=None-fw=None-fiw={}.json".format(mod.name, k.value)
            jsn = util.read_json(path + fname)
            CMS[imod, ik] = np.array(jsn["cm"])
    return CMS


def relative_improvement_table(metrics_repr, metrics_original, list_models):
    # relative improvement
    me = np.nanmean(metrics_repr, axis=1)
    ome = np.nanmean(metrics_original, axis=1) / 100

    tab = (me - ome) / ome
    tab = tab.T
    tab = np.round(tab * 100, 2)

    improvement = pd.DataFrame(tab, index=cst.metrics_name, columns=[k.name for k in list_models])
    # print(improvement.to_latex(index=True))
    return improvement


def metrics_vs_models_k(met_name, met_vec, out_dir, list_models, met_vec_original=None):

    labels = ["K=" + str(k.value) for k in cst.FI_Horizons]

    fmt = "%.2f"
    miny, maxy = -1, 1.1
    if "MCC" not in met_name:
        met_vec = met_vec * 100
        fmt = "%d%%"
        miny, maxy = 0, 110

    x = np.arange(len(labels)) * 9  # the label locations
    width = 0.5  # the width of the bars

    fig, ax = plt.subplots(figsize=(19, 9))

    indsxs = np.array(range(1, int(len(list_models) / 2) + 1))
    zero = [0] if len(list_models) % 2 == 1 else []
    ranges = list(reversed(indsxs * -1)) + zero + list(indsxs)

    R = []
    for iri, ri in enumerate(list_models):
        r_bar_i = ax.bar(x + width * ranges[iri], met_vec[iri, :], width, label=ri.name,
                         color=util.sample_color(iri, "tab20"), align='center')
        R += [r_bar_i]

    diffp = ((met_vec - met_vec_original) / met_vec_original * 100)
    for iri, ri in enumerate(R):

        diffp_show = []
        for i, val in enumerate(diffp[iri, :]):
            our = round(met_vec[iri, i]) if "MCC" not in met_name else round(met_vec[iri, i], 2)
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

def load_predictions_from_jsons(path, list_models, fw_win):
    logits = list()
    N = 139487

    for imod, mod in enumerate(list_models):
        fname = "model={}-seed=0-trst=FI-test=FI-data=FI-peri=FI-bw=None-fw=None-fiw={}.json".format(mod.name, fw_win)

        d = util.read_json(path + fname)

        logits_str = d['LOGITS']
        logits_ = np.array(json.loads(logits_str))

        print(mod, logits_.shape)
        # there are models for which the predictions are more because of the smaller len window, so we have to cut them
        if logits_.shape[0] != N:
            cut = logits_.shape[0] - N
            logits_ = logits_[cut:]

        if mod == cst.Models.DEEPLOBATT:
            horizons = [horizon.value for horizon in cst.FI_Horizons]
            h = horizons.index(fw_win)
            logits_ = logits_[:, :, h]

        logits_ = logits_[-20923:]   # TODO REMOVE
        logits.append(logits_)

    logits = np.dstack(logits)
    # logits.shape = [n_samples, n_classes, n_models]

    preds = np.argmax(logits, axis=1)
    # preds.shape = [n_samples, n_models]

    n_samples, n_classes, n_models = logits.shape
    logits = logits.reshape(n_samples, n_classes * n_models)
    # logits.shape = [n_samples, n_classes*n_models]

    return logits, preds


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


if __name__ == '__main__':

    PATH = "data/experiments/fi_final_jsons/"
    OUT = "data/experiments/pdfs/"

    LIST_MODELS = list(set(list(cst.Models))-{cst.Models.ATNBoF})
    LIST_MODELS = [m for m in cst.MODELS_YEAR_DICT if m in LIST_MODELS]
    LIST_YEARS = [cst.MODELS_YEAR_DICT[m] for m in cst.MODELS_YEAR_DICT if m in LIST_MODELS]

    os.makedirs(OUT, exist_ok=True)

    metrics = ['testing_FI_f1',
               'testing_FI_precision',
               'testing_FI_recall',
               'testing_FI_accuracy',
               'testing_FI_mcc']

    setup_plotting_env()

    MAT_REP = reproduced_metrics(PATH, metrics, LIST_MODELS)
    MAT_ORI = original_metrics(metrics, LIST_MODELS)
    CMS = confusion_metrix(PATH, LIST_MODELS)

    r_imp = relative_improvement_table(MAT_REP, MAT_ORI, LIST_MODELS)

    # n: PLOT 1
    for imet, met in enumerate(cst.metrics_name):
        metrics_vs_models_k(met, MAT_REP[:, :, imet], OUT, LIST_MODELS, MAT_ORI[:, :, imet])  # each mat has shape MODELS x K x METRICA
        print("plot done perf", met)

    # 1: PLOT 2
    confusion_matrices(CMS, LIST_MODELS, OUT)
    print("plot done cm")

    # n: PLOT 3
    for imet, met in enumerate(cst.metrics_name):
        met_data = np.mean(MAT_REP[:, :, imet], axis=1)   # MODELS x K x METRICA
        scatter_plot_year(met, met_data, LIST_MODELS, LIST_YEARS, OUT)
        print("plot done year", met)

    # for the meta learner
    lo, pred = load_predictions_from_jsons(PATH, LIST_MODELS, 10)

    plot_corr_matrix(LIST_MODELS, 10, pred, OUT)
    plot_agreement_matrix(LIST_MODELS, 10, pred, OUT)