import logging
import os

"""
logging.py
---
Utility functions for logging information during a run.
"""

output_logger = logging.getLogger(name='output');
data_logger = logging.getLogger(name='data');

# Output format is just the string; data format is CSV-like
OUTPUT_FILE = 'log.log'
DATA_FILE = 'data.csv'
OUTPUT_FORMAT = '%(message)s'
CSV_FORMAT = '%(iter)s,%(dloss)s,%(gloss)s,%(epoch)s,%(batch)s,"%(asctime)15s"'
def setup_loggers(experiment_name, output_filename=OUTPUT_FILE, data_filename=DATA_FILE):
	output_logger.setLevel(logging.DEBUG)
	output_handler = logging.FileHandler(experiment_name + '/' + output_filename)
	output_handler.setFormatter(logging.Formatter(OUTPUT_FORMAT))
	output_logger.addHandler(output_handler)

	data_logger.setLevel(logging.DEBUG)
	data_handler = logging.FileHandler(experiment_name + '/' + data_filename)
	data_handler.setFormatter(logging.Formatter(CSV_FORMAT))
	data_logger.addHandler(data_handler)

	# Write CSV header if file is empty
	if os.stat(experiment_name + '/' + data_filename).st_size == 0:
		f = open(experiment_name + '/' + DATA_FILE, 'a') # For appending
		f.write('Iter,Dloss,Gloss,Epoch,Batch,Timestamp\n')
		f.close()

"""
When called, print the string message to cout, and logs the message
to a file.
"""
def log_debug(msg):
	print(msg)
	output_logger.debug(msg)

"""
When called, logs the data in CSV format to the data file.
"""
def log_data(iteration=-1, dloss=-1, gloss=-1, epoch=-1, batch=-1):
	data_logger.debug('', extra={
		'iter': iteration,
		'dloss': dloss,
		'gloss': gloss,
		'epoch': epoch,
		'batch': batch,
	})