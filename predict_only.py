import argparse
import datetime
import sys
#################################################
# Automatically checking train and test set sizes!
#################################################
from subprocess import check_output

import numpy as np
from keras.callbacks import Callback, LearningRateScheduler, ModelCheckpoint
from keras.layers import (Conv1D, Dense, Dropout, GlobalAveragePooling1D,
                          GlobalMaxPooling1D, Input, concatenate)
from keras.models import Model  # , Sequential
from keras.models import load_model
from keras.optimizers import SGD, Adam, Nadam
from sklearn.metrics import confusion_matrix, roc_auc_score

from helper_with_N import *

################################
##Read in the parameter values##
################################
parser = argparse.ArgumentParser()
parser.add_argument("--input_file")  # data file name
parser.add_argument("--model_path")  # pretrained model
parser.add_argument("--output_path")  # Prefix to save the predictions.

args = parser.parse_args()

tick = datetime.datetime.now()


def wc(filename):
    return int(check_output(["wc", "-l", filename]).split()[0])


test_set_size = wc(args.input_file)
print("test_set_size: ", test_set_size)
te_steps_per_ep = int(test_set_size / 128)
print("##\n input data: \n ", args.input_file, "\n##")

#######################
# read in the test data, only true labels are needed for ROC
test_labels = []
counter = 0

f = open(args.input_file)
for line in f:
    line = line[:-1]  #remove \n
    seq, lab = process_line(line)
    test_labels.append(lab)
    counter += 1
f.close()

test_labels = np.array(test_labels)  # put to numpy format

###############################
####Defining the model#########
###############################

# load pretrained models
model = load_model(args.model_path)
model.summary()

model_name = (
    args.model_path).split(".hdf")[0]  # for saving predictions and labels
print(model_name)
###############################
####Testing the model##########
###############################

print("##########################")
pred_probas = model.predict_generator(generate_batches_from_file(
    args.input_file, 128),
                                      steps=te_steps_per_ep + 1,
                                      workers=1,
                                      use_multiprocessing=False)
print("original pred_probas size (divisible with batch size)",
      np.shape(pred_probas))
pred_probas = pred_probas[:test_set_size, :]
print("cropped the repetitions away, leaving", np.shape(pred_probas))
preds = pred_probas > 0.5
predict_labels = [1 if p else 0 for p in preds]

print("TEST ROC area under the curve \n",
      roc_auc_score(test_labels, pred_probas))
np.savetxt(args.output_path + "_TEST_predictions.txt", pred_probas, fmt="%.5f")
np.savetxt(args.output_path + "_TEST_pred_labels.txt",
           predict_labels,
           fmt="%d")
tock = datetime.datetime.now()
print("The process took time: {}".format(tock - tick), file=sys.stderr)
