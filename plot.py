import networks
import utils

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
import os, os.path

def parse_arguments():
    """
    Parses input arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", dest="network", metavar="NETWORK", default=None,
                        help="Choose network.")        
    
    args = parser.parse_args()
    return args

"""**Plots reliability diagram. (Confidence on x-axis, accuracy on y-axis)**"""
def plot_reliability(network):
    result = utils.load_results(network) #Load results in dictionary

    if not (os.path.exists('Plots')): #Create directory if needed
        os.mkdir('Plots')

    factor = 1.5
    #Plot for ECE
    plt.figure(figsize = (8*factor,6*factor) )
    ax = plt.axes()
    line = np.linspace(0,1,50)
    ax.plot(line, line, linestyle = "--", color = "k", label = "calibrated".capitalize())
    plt.title("Reliability diagram, " + network.capitalize() + ', uniform bins', fontsize = 22)
    plt.xlabel("Confidence", fontsize = 20)
    plt.ylabel("Accuracy", fontsize = 20)
    plt.xlim(0,1)
    plt.ylim(0,1)
    ax.xaxis.set_minor_formatter(mpl.ticker.FormatStrFormatter(""))
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=14)

    
    for method in result.keys():
        conf = result[method]['ece_confidence']
        acc = result[method]['ece_accuracy']
        ax.plot(conf, acc, "-o", label = np.array2string(result[method]['method']).strip("\'").capitalize())
    plt.legend(fontsize = 18)
    plt.savefig('Plots/' + network + '_ECE.png') #Save figure

    #Plot for AECE
    plt.figure(figsize = (8*factor,6*factor))
    ax = plt.axes()
    ax.plot(line, line, linestyle = "--", color = "k", label = "calibrated".capitalize())
    plt.title("Reliability diagram, " + network.capitalize()+ ', adaptive bins', fontsize = 22)
    plt.xlabel("Confidence", fontsize = 20)
    plt.ylabel("Accuracy", fontsize = 20)
    plt.xlim(0,1)
    plt.ylim(0,1)
    ax.xaxis.set_minor_formatter(mpl.ticker.FormatStrFormatter(""))
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=14)

    for method in result.keys():
        conf = result[method]['adaptive_confidence']
        acc = result[method]['adaptive_accuracy']
        ax.plot(conf, acc, "-o", label = np.array2string(result[method]['method']).strip("\'").capitalize())
    plt.legend(fontsize = 18)
    plt.savefig('Plots/' + network + '_AECE.png') #Save figure


args = parse_arguments()
plot_reliability(args.network)