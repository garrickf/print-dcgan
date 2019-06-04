import matplotlib.pyplot as plt
import pandas as pd
import argparse

"""
plot-data
---
Utility for plotting data.
"""

DEFAULT_FONTSIZE = 6
DEFAULT_LINESIZE = 0.5

plt.rcParams["font.size"] = DEFAULT_FONTSIZE

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_name', type=str, default='experiment_default', help='name of experiment')
args = parser.parse_args()

def plot_loss_plots(df, experiment_name = '', linewidth=0.5):
	fig = plt.figure(figsize=(6, 3), dpi=300)
	# Three ints for add_subplot: nrows, ncols, index
	gplt = fig.add_subplot(211)
	dplt = fig.add_subplot(212)

	gplt.plot(df['Iter'].values, df['Dloss'].values, color='b', linewidth=linewidth, label='Discriminator loss')
	gplt.plot(df['Iter'].values, df['Gloss'].values, color='r', linewidth=linewidth, label='Generator loss')
	dplt.plot(df['Iter'].values, df['Gloss'].values, color='r', linewidth=linewidth, label='Generator loss')
	dplt.plot(df['Iter'].values, df['Dloss'].values, color='b', linewidth=linewidth, label='Discriminator loss')

	gplt.set_title(experiment_name + ': Generator loss (stacked on Discriminator loss)')
	dplt.set_title(experiment_name + ': Discriminator loss (stacked on Generator loss)')
	plt.legend()
	plt.xlabel('Iteration')
	plt.ylabel('Loss')

	plt.tight_layout() # Fix sizing
	plt.show()
	
"""
Main part of script
"""
linesize = DEFAULT_LINESIZE
ename = args.experiment_name

df = pd.read_csv('./' + args.experiment_name + '/data.csv', index_col=False) # No column is the index column
plot_loss_plots(df, experiment_name=ename, linewidth=linesize)
