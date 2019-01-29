from MCD_DA.classification.model import svhn2mnist
from MCD_DA.classification.model import usps
from MCD_DA.classification.model import syn2gtrsb


# from MCD_DA.classification.model import syndig2svhn


def Generator(source, target, pixelda=False):
    if source == 'usps' or target == 'usps':
        return usps.Feature()
    elif source == 'svhn':
        return svhn2mnist.Feature()
        # return svhn2mnist_Feature()
    elif source == 'synth':
        return syn2gtrsb.Feature()


def Classifier(source, target):
    if source == 'usps' or target == 'usps':
        return usps.Predictor()
    if source == 'svhn':
        return svhn2mnist.Predictor()
    if source == 'synth':
        return syn2gtrsb.Predictor()
