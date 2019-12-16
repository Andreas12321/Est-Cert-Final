'''Import packages'''
import numpy as np
import sklearn.metrics as sk_met
from AdaptiveBinning import AdaptiveBinning

"""**Returns the negative log-likelihood as implemented by scikit.**"""
def nll_score(y_pred, y_test):
    return sk_met.log_loss(y_test, y_pred, normalize = True)


"""**Brier score från** https://stats.stackexchange.com/questions/403544/how-to-compute-the-brier-score-for-more-than-two-classes"""
def brier_score(y_pred, y_test):
    return np.mean(np.sum((y_pred - y_test)**2, axis=1)) #**2 utför operationen elementvis


"""**Derive correctness, vector of length(number of samples). Element == true if correct classification**"""
def correctness_calc(y_pred, y_test):
    prediction = np.argmax(y_pred, axis=1)
    label = np.argmax(y_test, axis=-1)
    correctness = (label == prediction)
    return correctness


"""**Packs results from inference for calculations of AECE, demo: https://github.com/yding5/AdaptiveBinning/blob/master/demo.py. Also returns accuracy.**"""
def collect_confidence(y_pred, y_test):
    probability = y_pred
    prediction = np.argmax(probability, axis=1)
    label = np.argmax(y_test, axis=-1)
    infer_results = []

    for i in range(len(y_test)):
        correctness = (label[i] == prediction[i])
        infer_results.append([probability[i][prediction[i]], correctness])  
    
    return infer_results

"""**Computes certainty measures**"""
def certainty_measures(y_pred, y_test):
    NLL = nll_score(y_pred, y_test)
    brier = brier_score(y_pred, y_test)
    print("--------------------------------")
    print("NLL: ", NLL )
    print("Brier: ", brier)
    ECE, ece_confidence, ece_accuracy = ece_score(y_pred, y_test)
    print("ECE: ", ECE)
    AECE, AMCE, cof_min, cof_max, adaptive_confidence, adaptive_accuracy = AdaptiveBinning(collect_confidence(y_pred, y_test), False)
    print("AECE: ", AECE)
    print("AMCE: ", AMCE)
    print("--------------------------------")

    return NLL, brier, ECE, AECE, AMCE, ece_confidence, ece_accuracy, adaptive_confidence, adaptive_accuracy

"""**Returns final ECE and acccuracies and confidence of each bin. Uses equidistant bins. https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py**"""
def ece_score(y_pred, y_test, n_bins = 20):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    accuracy_array = []
    confidence_array = []

    confidences = np.amax(y_pred, axis =-1) #Value
    predictions = np.argmax(y_pred, axis = -1) #Index
    correctness = correctness_calc(y_pred, y_test) #True or false
    num_samples = y_pred.shape[0]
    ece = np.zeros(num_samples)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.greater(confidences, bin_lower.item()) * np.less(confidences, bin_upper.item())
        sum_in_bin = np.sum(in_bin)
        if sum_in_bin > 0:
            prop_in_bin = np.mean(in_bin)
            accuracy_in_bin = np.mean(correctness[in_bin]) #Accuracy in bin
            avg_confidence_in_bin = np.mean(confidences[in_bin]) #Average confidence in bin

            #Append results
            accuracy_array.append(accuracy_in_bin) 
            confidence_array.append(avg_confidence_in_bin)
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    ece = np.sum(ece)/num_samples
    return ece, confidence_array, accuracy_array
