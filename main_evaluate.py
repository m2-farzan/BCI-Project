# Our modules:
from cnn import CNN1
from preprocessing import FBCSP_Select 
from preprocessing import Hilbert
from preprocessing import Resample

# Library modules:
import moabb
from moabb.datasets import BNCI2014001
from moabb.paradigms import MotorImagery
from moabb.evaluations import CrossSessionEvaluation
from mne.decoding import CSP
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline
import tensorflow as tf

# Fix random states:
tf.random.set_seed(221)
np.random.seed(75)


def evaluate():
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


    # Save
    results = pd.concat(list(results.values()))
    results.to_pickle('results.pandas')


if __name__ == "__main__":
    evaluate()
