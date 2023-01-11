import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt

def plot_data_vary(df_model, x, model_type, metrics_plot):
	df_model = df_model[df_model["model"]==model_type]  # LSTM
	df_model = df_model[df_model["cod"]==x]  # F

	model_mean = df_model.groupby(x).agg({"Loss": np.mean, "Precision": np.mean, "Recall": np.mean, "Fscore": np.mean})
	model_std = df_model.groupby(x).agg({"Loss": np.std, "Precision": np.std, "Recall": np.std, "Fscore": np.std})
	
	model_mean = model_mean.rename(columns={'Loss':'Loss-mean', 'Precision':'Precision-mean', 'Recall':'Recall-mean', 'Fscore':'Fscore-mean'})
	model_std = model_std.rename(columns={'Loss':'Loss-std', 'Precision':'Precision-std', 'Recall':'Recall-std', 'Fscore':'Fscore-std'})

	out = pd.concat([model_mean, model_std], axis=1)
	out[x] = out.index

	metrics = ["Loss", "Precision", "Recall", "Fscore"]
	for m in metrics:
		out[m + "_std_up_band"] = out[m+"-mean"] + out[m+"-std"] 
		out[m + "_std_lo_band"] = out[m+"-mean"] - out[m+"-std"] 
	
	for m in metrics_plot:
		plt_name = "metric-x{}-mod{}-met{}".format(x, model_type, m)

		plt.plot(out[x], out[m+"_std_up_band"], '.-')
		plt.plot(out[x], out[m+"_std_lo_band"], '.-')
		plt.fill_between(out[x], out[m+"_std_lo_band"], out[m+"_std_up_band"], alpha=.3)
		plt.plot(out[x], out[m+"-mean"], '.-')

		plt.ylabel(m)
		plt.xlabel(x)
		plt.title(plt_name)

		# plt.tight_layout()
		plt.savefig("adv_plots" + "/" + plt_name, dpi=400)
		plt.clf()

# ------------

metrics_to_plot = ["Loss", "Precision", "Recall", "Fscore"]
df_model = pd.read_csv("attack_metrics.csv")

for model_type in ["LSTM", "CNN2"]:
	for xval in ["F", "L"]:
		plot_data_vary(df_model, xval, model_type, metrics_to_plot)
