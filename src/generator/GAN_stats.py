import pandas as pd
from matplotlib import pyplot as plt
import json

fname = "exp|bas-32|hor-20|win-20|atk-CNN2|isfi-1"

# Opening JSON file
data = None
with open('data/archive/{}.json'.format(fname)) as jf:
    data = json.load(jf)

df1_columns = ['gen_total_loss', 'gen_pred_loss', 'gen_cost_loss', 'gen_dis_loss',
               'dis_total_loss', 'dis_real_loss', 'dis_fake_loss']

df2_columns = ['val_f_avgloss', 'val_f1', 'val_prec', 'val_recall']

df1 = pd.DataFrame({k: data[k] for k in df1_columns})
df2 = pd.DataFrame({k: data[k] for k in df2_columns})

df1[['dis_real_loss', 'dis_fake_loss']].plot()
plt.show()

for k in df1_columns:
    df1[k].plot(xlabel="Epochs", ylabel=k)
    plt.savefig('data/'+k)
    plt.show()
    plt.clf()

for k in df2_columns:
    df2[k].plot(xlabel="Epochs", ylabel=k)
    plt.savefig('data/'+k)
    plt.clf()
