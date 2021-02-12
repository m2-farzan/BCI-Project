"""By Erfan Mahmoodtabar & Mohammad Farzan."""

# Our modules:
from cnn import CNN1
from preprocessing import FBCSP_Select 
from preprocessing import Hilbert
from preprocessing import Resample

# Library modules:
import matplotlib.pyplot as plt
import moabb
from moabb.datasets import BNCI2014001
from moabb.paradigms import MotorImagery
from moabb.evaluations import CrossSessionEvaluation
from mne.decoding import CSP
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline


# Definitions
moabb.set_log_level('info')
N_SUBJECTS = 2

pipelines = {
    'CSP + LDA': make_pipeline(
        CSP(n_components=8),
        LDA()),
    'Sakhavi (2018)': make_pipeline(
        FBCSP_Select(),
        Hilbert(),
        Resample(),
        CNN1()),
}


# Load dataset
dataset = BNCI2014001()
dataset.subject_list = dataset.subject_list[:N_SUBJECTS]
datasets = [dataset]
overwrite = True  # set to True if we want to overwrite cached results


# Evaluate methods
results = {}
for name, pipeline in pipelines.items():
    print("Evaluating pipeline: " + name)
    paradigm = MotorImagery(fmin=8, fmax=35, n_classes=4)
    evaluation = CrossSessionEvaluation(paradigm=paradigm, datasets=datasets,
                                        suffix='examples', overwrite=overwrite)
    results[name] = evaluation.process({name: pipeline})


# Visualize
results = pd.concat(list(results.values()))

## Bar chart
fig = plt.figure(figsize=[8, 4])
sns.stripplot(data=results, y='score', x='pipeline', jitter=True,
              alpha=.5, zorder=1, palette="Set1")
sns.pointplot(data=results, y='score', x='pipeline',
              zorder=1, palette="Set1")
plt.ylabel('Score')
plt.ylim([0.5, 1])
plt.tight_layout()
fig.savefig("fig-bar.png", dpi=150)

## Paired plots
for i in range(len(pipelines)):
    for j in range(i+1, len(pipelines)):
        name_i = list(pipelines.keys())[i]
        name_j = list(pipelines.keys())[j]
    
        fig = plt.figure(figsize=[6, 4])
        paired = results.pivot_table(values='score', columns='pipeline',
                                     index=['subject', 'session'])
        paired = paired.reset_index()

        sns.regplot(data=paired, y=name_i, x=name_j,
                    fit_reg=False)
        plt.plot([0, 1], [0, 1], ls='--', c='k')
        plt.xlim([0.3, 1])
        plt.tight_layout()
        fig.savefig("fig-reg-%d-vs-%d.png" % (i, j), dpi=150)