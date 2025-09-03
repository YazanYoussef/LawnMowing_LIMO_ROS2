
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd


file_path = os.path.join('/home', 'yazanyoussef', 'Desktop', 'IROS_2024_Preparation', 'Final_Work', 'analysis', 'Full_Framework', 'tsp_11-20_run_Critic_Wed_2024_05_15_T_23_21_35_15OPT_GAP.csv')
df1 = pd.read_csv(file_path)

file_path = os.path.join('/home', 'yazanyoussef', 'Desktop', 'IROS_2024_Preparation', 'Final_Work', 'analysis', 'Full_Framework', 'tsp_11-20_run_Critic_Wed_2024_05_15_T_23_21_35 _20OPT_GAP.csv')
df2 = pd.read_csv(file_path)

file_path = os.path.join('/home', 'yazanyoussef', 'Desktop', 'IROS_2024_Preparation', 'Final_Work', 'analysis', 'Full_Framework', 'tsp_11-20_run_Mon_2024_03_18_T_16_49_46_OPT_GAP_15.csv')
df3 = pd.read_csv(file_path)

file_path = os.path.join('/home', 'yazanyoussef', 'Desktop', 'IROS_2024_Preparation', 'Final_Work', 'analysis', 'Full_Framework', 'tsp_11-20_run_Mon_2024_03_18_T_16_49_46_OPT_GAP_20.csv')
df4 = pd.read_csv(file_path)

fig, ax = plt.subplots()
ax.plot(df1.loc[:, 'Step'].to_numpy(), df1.loc[:, 'Value'], color='tab:green', linestyle='--', label='15-areas/map|Critic')
ax.plot(df1.loc[:, 'Step'].to_numpy(), df3.loc[:, 'Value'], color='tab:green', linestyle='-', label='15-areas/map|Rollout')
ax.plot(df2.loc[:, 'Step'].to_numpy(), df2.loc[:, 'Value'], color='tab:purple', linestyle='--', label='20-areas/map|Critic')
ax.plot(df2.loc[:, 'Step'].to_numpy(), df4.loc[:, 'Value'], color='tab:purple', linestyle='-', label='20-areas/map|Rollout')

ax.yaxis.set_major_locator(ticker.LinearLocator(numticks=23))
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,d}'.format(int(x/1000))))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: '{:,.2f}'.format(y)))


ax.set_ylim(-11, 0)
ax.set_xlim(0, df1.loc[:, 'Step'].iat[-1])
ax.set_xlabel('Epoch')
ax.set_ylabel('Optimality Gap (%)')
ax.grid(linestyle=':')
plt.legend()
plt.tight_layout()
plt.show()

