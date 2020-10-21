import numpy as np
import pandas as pd

import glob
import os

miss_rate = '80'

list_of_files = glob.glob('~/Downloads/*') # * means all if need specific format then *.csv
latest_file = max(list_of_files, key=os.path.getctime)

file_in = latest_file
file_out = 'report/tables.tex'

df = pd.read_csv(file_in, index_col=False, delimiter = ',', engine='python')
# df = df.loc[df['Tags'].isin(['run_VAEs', 'run_benchmarks', 'run_gain', 'run_miwae', 'run_mice', 'run_PSMVAE'])]
# df = df.loc[df['Tags'].isin(['run_new'])]
# df = df.loc[df['no_post_sample'].isin([False])]
# df = df.loc[df['Tags'].isin(['run_cluster'])]
# df = df.loc[df['pi'].isin([np.nan, 0.5, 0.9])] ######
# df = df.loc[df['cluster'].isin([np.nan, False])] ######

df = df.loc[~(df['miss_data_file']!=df['miss_data_file'])]
df['Missingness'] = [i.split('/')[3].split('_')[0] for i in df['miss_data_file']]
df['miss ratio'] = [i.split('/')[3].split('_')[3] for i in df['miss_data_file']]
df['Data']  = [i.split('/')[1] for i in df['miss_data_file']]
df.loc[df['Tags']=='run_gain', 'model_class'] = 'corgain'
df.loc[df['Tags']=='run_miwae', 'model_class'] = 'cormiwae'
df.loc[df['Tags']=='run_mice', 'model_class'] = 'cormice'

df = df.loc[df['model_class'].isin(['MissPriorVAEwoPrior', 'PSMVAEr'])]

# df = df.loc[(df['Data'].isin(['breast']))]
if miss_rate == '20':
    df = df.loc[(df['Data'].isin(['adult', 'letter', 'wine']))]
else:
    df = df.loc[(df['Data'].isin(['breast', 'credit', 'spam']))]

idx_columns = ['Missingness', 'miss ratio', 'num_samples', 'model_class', 'seed']

results = df.loc[~df['Train Imputation RMSE loss'].isin([0., np.nan]), df.columns.isin(['Train Imputation RMSE loss', 'Data'] + idx_columns)]
# results = df.loc[~df['Cluster accuracy'].isin([0., np.nan]), df.columns.isin(['Cluster accuracy', 'Data'] + idx_columns)]

print(len(results))
results.drop_duplicates(subset=['seed', 'Missingness', 'miss ratio', 'Data', 'model_class', 'num_samples'], keep='last', inplace=True)
print(len(results))

idx_columns = ['Missingness', 'miss ratio', 'num_samples', 'model_class']

averaged_df = results.groupby(['Data'] + idx_columns).mean()
std_df = results.groupby(['Data'] + idx_columns).std()

averaged_df = averaged_df.reset_index(['Data'] + idx_columns)
std_df = std_df.reset_index(['Data'] + idx_columns)

averaged_df = averaged_df.pivot(index=idx_columns, columns='Data', values='Train Imputation RMSE loss')
std_df = std_df.pivot(index=idx_columns, columns='Data', values='Train Imputation RMSE loss')
# averaged_df = averaged_df.pivot(index=idx_columns, columns='Data', values='Cluster accuracy')
# std_df = std_df.pivot(index=idx_columns, columns='Data', values='Cluster accuracy')

averaged_df = averaged_df.reset_index(idx_columns)
std_df = std_df.reset_index(idx_columns)

# print(averaged_df.loc[:, averaged_df.columns.isin(['Missingness', 'model_class', 'breast'])].to_latex())
pres = ""
for ratio in [miss_rate]: #std_df['miss ratio'].unique():
    for missi in ['MNAR1var', 'MCAR']: #std_df['Missingness'].unique():
        for num_sam in std_df['num_samples'].unique():
            if len(std_df.loc[(std_df['Missingness']==missi) & (std_df['miss ratio']==ratio) & (std_df['num_samples']==num_sam)].values) > 0:
                table = "\\begin{table}\n\\centering\n"
                table += "\\begin{tabularx}{\\textwidth}{|" + "".join(["X"]*(len(idx_columns)-3)) + "||" + "|".join(["X"]*(len(std_df.columns)-len(idx_columns))) + "|}\hline\n"
                table += ' & '.join(averaged_df.columns[3:]).replace('_', ' ') + '\\\\ \\hline  \n'
                table += ' '.join([' & '.join([str(avr[idx]).replace('MissPriorVAE', '') for idx in range(3, len(idx_columns))]) + ' & ' + \
                    ' & '.join([str(i)[1:6] + "$\\pm$" + str(j)[1:6] for i, j in zip(avr[4:], mad[4:])]) + '\\\\\n' 
                    for (avr, mad) in zip(averaged_df.loc[(averaged_df['Missingness']==missi) & (averaged_df['miss ratio']==ratio) & (averaged_df['num_samples']==num_sam)].values, 
                    std_df.loc[(std_df['Missingness']==missi) & (std_df['miss ratio']==ratio) & (std_df['num_samples']==num_sam)].values)])
                table += '\\hline\n\\end{tabularx}\n'
                table += f"\\caption\u007b Train RMSE for {missi} with missingness ratio {ratio} and {num_sam} samples\u007d\n"
                table += '\\end{table}'
                pres += table + "\n\n"

with open(file_out, 'w') as file:
    file.write(pres)




# results_test = df.loc[~df['Test Imputation RMSE loss'].isin([0., np.nan]), df.columns.isin(['Test Imputation RMSE loss', 'Data'] + idx_columns)]

# averaged_df_test = results_test.groupby(['Data'] + idx_columns).mean()
# std_df_test = results_test.groupby(['Data'] + idx_columns).std()

# averaged_df_test = averaged_df_test.reset_index(['Data'] + idx_columns)
# std_df_test = std_df_test.reset_index(['Data'] + idx_columns)

# averaged_df_test = averaged_df_test.pivot(index=idx_columns, columns='Data', values='Test Imputation RMSE loss')
# std_df_test = std_df_test.pivot(index=idx_columns, columns='Data', values='Test Imputation RMSE loss')

# averaged_df_test = averaged_df_test.reset_index(idx_columns)
# std_df_test = std_df_test.reset_index(idx_columns)


# pres = ""

# for ratio in std_df_test['miss ratio'].unique():
#     for missi in std_df_test['Missingness'].unique():
#         for num_sam in std_df_test['num_samples'].unique():
#                 table = "\\begin{table}\n\\centering\n"
#                 # table += "\\label{" + idx0 + idx1 + str(idx2) + idx3 + "}\n"
#                 table += "\\begin{tabularx}{\\textwidth}{|" + "".join(["X"]*(len(idx_columns)-2)) + "||" + "|".join(["X"]*(len(std_df.columns)-len(idx_columns))) + "|}\hline\n"
#                 table += ' & '.join(averaged_df_test.columns[3:]).replace('_', ' ') + '\\\\ \\hline  \n'
#                 table += ' '.join([' & '.join([str(avr[idx]).replace('MissPriorVAE', '') for idx in range(3, len(idx_columns))]) + ' & ' + \
#                     ' & '.join([str(i)[:6] + "$\\pm$" + str(j)[:6] for i, j in zip(avr[4:], mad[4:])]) + '\\\\\n' 
#                     for (avr, mad) in zip(averaged_df_test.loc[(averaged_df_test['Missingness']==missi) & (averaged_df_test['miss ratio']==ratio) & (averaged_df_test['num_samples']==num_sam)].values, 
#                     std_df.loc[(std_df_test['Missingness']==missi) & (std_df_test['miss ratio']==ratio) & (std_df_test['num_samples']==num_sam)].values)])
#                 table += '\\hline\n\\end{tabularx}\n'
#                 table += f"\\caption\u007b Test RMSE for {missi} with missingness ratio {ratio} and {num_sam} samples\u007d\n"
#                 table += '\\end{table}'
#                 pres += table + "\n\n"

