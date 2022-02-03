import time
from time import time

import numpy as np
import scipy.stats as stats
from scipy.stats import randint as sp_randint
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.neural_network import MLPClassifier
from sklearn.utils.fixes import loguniform
from skorch import NeuralNetClassifier

from signor.format.format import banner
from signor.ioio.dir import signor_dir
from signor.monitor.probe import summary
from signor.utils.dict import update_dict
from signor.utils.np import sampler
from signor.utils.pt.mlp import MLP_simple
# Utility function to report best scores
from signor.utils.random_ import fix_seed


def report(results, n_top=3):
    """ print top n results
        return best result
    """
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

    best = np.flatnonzero(results['rank_test_score'] == 1)
    if len(best)!=1: print(f'Best candidates are not single. Both {best} are best')
    best = best[0]
    return results['params'][best]


def hyperparameter_seach(clf, x, y, param_dist=None, model=None, **kwargs):
    """
    :param clf: classifier
    :param x: train+val
    :param y: train+val
    :param param_dist:
    :param model:
    :return:
    """
    fix_seed()
    print(f'Use {y.shape[0]} datapoints for hyperparameter search for model {model}.')

    if param_dist == None:
        # specify parameters and distributions to sample from
        param_dist = {"max_depth": [3, None],
                      "max_features": sp_randint(1, 11),
                      "min_samples_split": sp_randint(2, 11),
                      "bootstrap": [True, False],
                      "criterion": ["gini", "entropy"]}

    # run randomized search

    default_kwargs = {'param_distributions':param_dist,'n_iter':20, 'cv':5, 'iid':False,
                      'n_jobs':2, 'scoring':'neg_mean_squared_error', 'random_state':42}
    if y.shape[0] > 5000: default_kwargs['cv'] = KFold(5)


    kwargs = update_dict(kwargs, default_kwargs)
    random_search = RandomizedSearchCV(clf, **kwargs)

    start = time()
    banner('random_search.fit')
    random_search.fit(x, y) # todo: handle warning./home/cai.507/.local/lib/python3.6/site-packages/sklearn/utils/deprecation.py:144: FutureWarning: The sklearn.metrics.scorer module is  deprecated in version 0.22 and will be removed in version 0.24. The corresponding classes / functions should instead be imported from sklearn.metrics. Anything that cannot be imported from sklearn.metrics is now part of the private API.
    banner('finish random_search.fit')

    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), kwargs['n_iter']))
    banner('Report Top models')
    return report(random_search.cv_results_)

from signor.datasets.image_classification.mnist import load_mnist

import argparse
parser = argparse.ArgumentParser(description='Experiment')
parser.add_argument('--skorch', action='store_true', help='use skorch')

parser.add_argument('--device', default=0, type=int, help='device num')
parser.add_argument('--sample', default=1000, type=int, help='')

if __name__ == '__main__':
    # X, y = load_digits(return_X_y=True)
    args = parser.parse_args()
    train_data, _, _ = load_mnist(data_dir=f'{signor_dir()}../data/')
    X, y = train_data

    X, y = sampler((X, y), s=args.sample)
    X = X.astype(np.float32) # need this
    y = y.astype(np.int64)
    summary(X, 'X')
    summary(y, 'y')

    if args.skorch:
        clf = NeuralNetClassifier(
            MLP_simple,
            max_epochs=20,
            lr=0.1,
            device='cuda:1',
            batch_size=256,
        )
        param_dist = {
            'lr': [0.01, ],
            'batch_size': [32], # [32, 64, 128, 16],
            'max_epochs': [30],
            'module__h_sizes': [[784, 100, 10]],
            'verbose': [0]
        }

    else:
        clf = MLPClassifier()
        param_dist = {
            'learning_rate_init': [0.01, ],
            'batch_size': [16, 32, 64],
            'max_iter': [30],
            'hidden_layer_sizes': [[100]],
        }


    kf = KFold(5, shuffle=True)
    for train_index, test_index in kf.split(X):
        pass

    kwargs = {'scoring': 'accuracy', 'cv': [(np.array(train_index), np.array(test_index))], 'n_jobs': -1}
    best = hyperparameter_seach(clf, X, y, param_dist=param_dist, model='Skorch_net', **kwargs)
    print(best)

    clf = NeuralNetClassifier(MLP_simple(best['module__h_sizes'], **best), device='cuda:1', **best)
    clf.fit(X, y)
    y_proba = clf.predict_proba(X)
    exit()

    # build a classifier
    clf = SGDClassifier(loss='hinge', penalty='elasticnet', fit_intercept=True)


    param_dist = {'average': [True, False],
                  'l1_ratio': stats.uniform(0, 1),
                  'alpha': loguniform(1e-4, 1e0)}

    kwargs = {'scoring':'accuracy'}
    best = hyperparameter_seach(clf, X, y, param_dist=param_dist, model='SGDClassifier', **kwargs)
    print(best)

    exit()
    cv_results = hyperparameter_seach(clf, X, y, param_dist=param_dist, model='SGDClassifier')


    for k, v in cv_results.items():
        try:
            summary(v, name=k)
        except:
            print(f'Cannot summarize {k}')
            print(v)
