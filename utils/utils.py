"""
Various subroutines used to train/evaluate deep-active subspace surrogates
"""

import numpy as np
import easysurrogate as es
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict

def plot_sensitivity(results, param_names, n_include = 25, height=0.4, **kwargs):
    """
    Plot the global derivative-based sensitivity indices with confidenceintervals

    Parameters
    ----------
    results : dict
        dict containing the results for each of the n_rep replica networks
    n_include : int, optional
        How many indices to plot. The default is 25.

    Returns
    -------
    None.

    """     
    # the sensitivity indices
    V_i = results['V_i']
    # compute mean over replicas
    V_i_mean = np.mean(V_i, axis = 0)
    # rank indices according to mean
    idx = np.fliplr(np.argsort(np.abs(V_i_mean.reshape([-1, 1])).T))[0]
    
    # compute confidence interval
    analysis = es.analysis.BaseAnalysis()
    lower, upper = analysis.get_confidence_intervals(V_i)
    x_err = np.array([V_i_mean[idx] - lower[idx], upper[idx] - V_i_mean[idx]])
  
    # plot results
    fig = plt.figure(figsize=[5, 10])
    ax = fig.add_subplot(111, title=kwargs.get('title', ''))

    offset = 0
    if 'bar2' in kwargs:
        offset = height
        bar2 = kwargs['bar2'][param_names[idx]].values.flatten()
        bar2 /= bar2[0]
        ax.barh(np.array(range(n_include))-offset/2, width = bar2[0:n_include],
                color = 'salmon', height=height, label='KAS-GP', hatch='//')

    ax.set_xlabel(r'$\nu_i\;/\;\nu_1$', fontsize=18)
    nu_i = V_i_mean[idx].flatten()
    nu_i /= nu_i[0]
    ax.barh(np.array(range(n_include))+offset/2, width = nu_i[0:n_include],
            color = 'dodgerblue', height=height, label='DAS')#, xerr = x_err[:, 0:n_include])

    ax.set_xticks(np.linspace(0, 1, 5))
    ax.set_yticks(range(n_include))
    ax.set_yticklabels(param_names[idx][0:n_include], fontsize=18)
    ax.invert_yaxis()
    sns.despine(top=True)
    plt.legend(loc=0, fontsize=18)
    plt.tight_layout()
    plt.savefig('./images/SA.png')
    

def plot_dist(output, lbl):
    analysis = es.analysis.BaseAnalysis()

    fig = plt.figure(figsize=[10,4.5])
    ax1 = fig.add_subplot(1, 2, 1, title='all samples')
    count, _, _ = ax1.hist(output.flatten(), 30, color='lightgray')
    dom, kde = analysis.get_pdf(output)
    # scale kde to match histogram
    ax1.plot(dom, kde * np.max(count) / np.max(kde))
    ax1.set_yticks([])
    ax1.set_xlabel(lbl)

    n_bootstrap = 1000
    n_replicas = output.shape[1]
    n_samples = output.shape[0]

    output_replicas = np.zeros([n_bootstrap, n_replicas])
    output_params = np.zeros([n_bootstrap, n_samples])

    for i in range(n_bootstrap):

        idx = np.random.randint(0, n_samples - 1, n_samples)
        # average out parameters
        output_replicas[i] = np.mean(output[idx], axis=0)
        idx = np.random.randint(0, n_replicas - 1, n_replicas)
        # average out replicas
        output_params[i] = np.mean(output[:, idx], axis=1)

    ax2 = fig.add_subplot(2, 2, 2, title='samples averaged over replicas',
                         xlim=[np.min(output), np.max(output)])

    count, _, _ = ax2.hist(output_params.flatten(), 30, color='lightgray')
    dom, kde = analysis.get_pdf(output_params.flatten())
    # scale kde to match histogram
    ax2.plot(dom, kde * np.max(count) / np.max(kde))
    ax2.set_yticks([])
    ax2.set_xlabel(lbl)

    ax3 = fig.add_subplot(2, 2, 4, title='samples averaged over parameters',
                         xlim=[np.min(output), np.max(output)])
    count, _, _ = ax3.hist(output_replicas.flatten(), 30, color='lightgray')
    dom, kde = analysis.get_pdf(output_replicas.flatten())
    # scale kde to match histogram
    ax3.plot(dom, kde * np.max(count) / np.max(kde))
    ax3.set_yticks([])
    ax3.set_xlabel(lbl)

    plt.tight_layout()
    plt.savefig("./images/dist.png")
    
    
def get_errors(surrogate, feats_train, target_train, feats_test, target_test):
    """
    Compute the training and test errors.

    Parameters
    ----------
    feats_train : array, shape (n_train, n_inputs)
        The training features.
    target_train : array, shape (n_train, n_outputs)
        The training target data points.
    feats_test : array (n_test, n_inputs)
        The test features.
    target_test : array, shape (n_test, n_outputs)
        The test target data points.

    Returns
    -------
    err_train : float
        The relative training error.
    err_test : float
        The relatiove test error.

    """

    train_pred = np.zeros([target_train.shape[0], target_train.shape[1]])
    for i in range(target_train.shape[0]):
        train_pred[i, :] = surrogate.predict(feats_train[i])

    err_train = np.mean(np.linalg.norm(target_train - train_pred, axis=0) /
                        np.linalg.norm(target_train, axis=0), axis=0)
    print("Relative training error = %.4f %%" % (err_train * 100))

    # run the trained model forward at test locations
    test_pred = np.zeros([target_test.shape[0], target_test.shape[1]])
    for i in range(target_test.shape[0]):
        test_pred[i] = surrogate.predict(feats_test[i])

    err_test = np.mean(np.linalg.norm(target_test - test_pred, axis=0) /
                       np.linalg.norm(target_test, axis=0), axis=0)
    print("Relative test error = %.4f %%" % (err_test * 100))

    return err_train, err_test
    
def plot_errors(results):
    """
    Plots the training and test error vs the number of epochs
    """
    epoch_stop = results['epoch_stop']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('epoch', fontsize=14)
    ax.set_ylabel('relative error', fontsize=14)
    for idx, err in enumerate(results['errors']):
        ax.plot(err[0:epoch_stop[idx], 0], '-o', label='training error', color = 'dodgerblue')
        ax.plot(err[0:epoch_stop[idx], 1], '-s', label='test error', color = 'salmon')
#     ax.set_xticks(np.arange(n_epochs_max))
    handles, labels = ax.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.tight_layout()

def early_stopping(surrogate, n_iter, params_train, samples_train, params_test, samples_test,
                   patience=3, min_delta=0.001, n_epochs_max=100):
    """
    Stop training when the test error has stopped improving.


    Parameters
    ----------
    surrogate : easysurrogate.methods
        An easySurrogate neural network.
    n_epochs_max : int
        The maximum number of epochs.
    n_iter : int
        The number mini-batch iterations per epoch.
    patience : int
        Number of epochs with no improvement after which training will be stopped.
    min_delta : int
        Minimum change in the test error to qualify as an improvement.

    Returns
    -------
    errors : array, shape (n_epochs_max, 2)
        Array containing the training and test errors.
    epoch : int
        The epoch at which training was stopped.

    """

    # store training and test errors
    errors = np.zeros([n_epochs_max, 2])

    # improvement is defined as a redution in testing error > min_delta
    improvement = np.zeros(n_epochs_max)

    # compute training & testing error 1st epoch
    errors[0] = get_errors(surrogate, params_train, samples_train, params_test, samples_test)
    improvement[0] = 1.

    # set the training flag to True in any layer that uses batch normalization
    surrogate.neural_net.set_batch_norm_training_flag(True)
    
    # retrain 
    for epoch in range(1, n_epochs_max):
        surrogate.neural_net.train(n_iter, store_loss = True)

        # compute training & testing error
        errors[epoch] = get_errors(surrogate, params_train, samples_train, params_test, samples_test)
        
        # check for improvement in test error
        improvement[epoch] = errors[epoch - 1, 1] - errors[epoch, 1] >= min_delta

        # stop if patience is exceeded
        if epoch >= patience:
            
            # exceeded patience: no improvent over the last patience epochs
            stop = np.array_equal(improvement[epoch - patience: epoch], np.zeros(patience))

            if stop:
                return errors, epoch

    # set the training flag to False in any layer that uses batch normalization
    surrogate.neural_net.set_batch_norm_training_flag(False)
            
    return errors, epoch
    
plt.rcParams['savefig.dpi'] = 300