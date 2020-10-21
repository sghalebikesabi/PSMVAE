import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import glob
import os

list_of_files = glob.glob('/home/ghalebik/Downloads/*') # * means all if need specific format then *.csv
latest_file = max(list_of_files, key=os.path.getctime)

file_in = latest_file
file_out = 'report/tables.tex'

df = pd.read_csv(file_in, index_col=False, delimiter = ',', engine='python')
# df = df.loc[df['Tags'].isin(['run_VAEs', 'run_benchmarks', 'run_gain', 'run_miwae', 'run_mice', 'run_PSMVAE'])]
df = df.loc[df['Tags'].isin(['run_pi'])]
df = df.loc[~df['pi'].isin([np.nan, 0.5])] ######
df = df.loc[df['cluster'].isin([np.nan, False])] ######
df = df.loc[df['model_class'].isin(['PSMVAE_a', 'PSMVAE_b'])] ######
df = df.loc[df['num_samples'].isin(['1.0'])] ######

df['Missingness'] = [i.split('/')[3].split('_')[0] for i in df['miss_data_file']]
df['miss ratio'] = [i.split('/')[3].split('_')[3] for i in df['miss_data_file']]
df['Data']  = [i.split('/')[1] for i in df['miss_data_file']]
df.loc[df['Tags']=='run_gain', 'model_class'] = 'corgain'
df.loc[df['Tags']=='run_miwae', 'model_class'] = 'cormiwae'
df.loc[df['Tags']=='run_mice', 'model_class'] = 'cormice'

# df = df.loc[(df['Data'].isin(['breast', 'credit', 'spam']))]
# df = df.loc[(df['Data'].isin(['adult', 'letter', 'wine']))]

idx_columns = ['Missingness', 'miss ratio', 'num_samples', 'model_class', 'pi', 'seed']

results = df.loc[~df['Train Imputation RMSE loss'].isin([0., np.nan]), df.columns.isin(['Train Imputation RMSE loss', 'Data'] + idx_columns)]
# results = df.loc[~df['Cluster accuracy'].isin([0., np.nan]), df.columns.isin(['Cluster accuracy', 'Data'] + idx_columns)]

print(len(results))
results.drop_duplicates(subset=['seed', 'Missingness', 'miss ratio', 'Data', 'model_class', 'num_samples', 'pi'], keep='first', inplace=True)
print(len(results))

idx_columns = ['Missingness', 'miss ratio', 'num_samples', 'model_class', 'pi']
averaged_df = results.groupby(['Data'] + idx_columns).mean()
std_df = results.groupby(['Data'] + idx_columns).std()

averaged_df = averaged_df.reset_index(['Data'] + idx_columns)
std_df = std_df.reset_index(['Data'] + idx_columns)

averaged_df = averaged_df.pivot(index=idx_columns, columns='Data', values='Train Imputation RMSE loss')
std_df = std_df.pivot(index=idx_columns, columns='Data', values='Train Imputation RMSE loss')

averaged_df = averaged_df.reset_index(idx_columns)
std_df = std_df.reset_index(idx_columns)

color=plt.cm.rainbow(np.linspace(0,1,9))
fig = plt.figure()
ax = plt.subplot(111)
i = 0
averaged_df = averaged_df[(averaged_df['model_class'].isin(['PSMVAE_a', 'PSMVAE_b'])) & (averaged_df['Missingness'].isin(['MCAR', 'MNAR1var'])) & (averaged_df['miss ratio']=='20')]
min_ = .055 #averaged_df.loc[:, averaged_df.columns.isin(['spam', 'wine'])].values.min()
max_ = averaged_df.loc[:, averaged_df.columns.isin(['spam', 'wine'])].values.max()
for model_class in ['PSMVAE_a', 'PSMVAE_b']:
    for missi in ['MCAR', 'MNAR1var']:
        for miss_ratio in ['20']: # '20', 
            for data in ['spam', 'wine']:
                i += 1
                datai = averaged_df[(averaged_df['model_class']==model_class) & (averaged_df['Missingness']==missi) & (averaged_df['miss ratio']==miss_ratio)]
                ax.plot(
                    datai['pi'].values, datai[data].values,
                    'o-', color = color[i],
                    label = model_class + " " + missi + " " + data 
                    )
                ax.vlines(datai['pi'].iloc[datai[data].argmin()], ymin=min_, ymax=datai[data].min(), color = color[i])
# ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
plt.xlabel(r'$\pi$')
plt.ylabel(r'RMSE')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xscale(r'log')
# plt.show()
plt.savefig('report/run_pi.eps', bbox_inches='tight')


averaged_df[(averaged_df['model_class'].isin(['PSMVAE_a'])) & (averaged_df['Missingness'].isin(['MCAR'])) & (averaged_df['miss ratio']=='20')]['spam']