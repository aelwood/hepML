from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc
from sklearn.model_selection import learning_curve
from sklearn.pipeline import Pipeline

from pandasPlotting.histFunctions import hist1dErrorInputs

import matplotlib.pyplot as plt
import os
import numpy as np

def classificationReport(y_predicted,clfResult,y_test,outputFile=None):
    report = classification_report(y_test, y_predicted,
                                            target_names=["background", "signal"])
    auc= "Area under ROC curve: %.4f"%(roc_auc_score(y_test,clfResult))

    if outputFile:
        outputFile.write(report)
        outputFile.write('\n')
        outputFile.write(auc)
    else:
        print report
        print auc
        

def rocCurve(y_preds,y_test=None,output=None,append=''):
    '''Compute the ROC curves, can either pass the predictions and the truth set or 
    pass a dictionary that contains one value 'truth' of the truth set and the other 
    predictions labeled as you want'''

    # Compute ROC curve and area under the curve
    if not isinstance(y_preds,dict):
        assert not y_test is None,'Need to include testing set if not passing dict'
        y_preds={'ROC':y_preds}
    else:
        y_test=y_preds['truth']

    for name,y_pred in y_preds.iteritems():
        if name=='truth': continue
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, lw=1, label=name+' (area = %0.2f)'%(roc_auc))

    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.grid()
    if not os.path.exists(output): os.makedirs(output)
    plt.savefig(os.path.join(output,'rocCurve'+append+'.pdf'))
    plt.clf()

def compareTrainTest(clf, X_train, y_train, X_test, y_test, output, bins=30,append=''):
    '''Compares the decision function for the train and test BDT'''
    decisions = []
    for X,y in ((X_train, y_train), (X_test, y_test)):
        d1 = clf(X[y>0.5]).ravel()
        d2 = clf(X[y<0.5]).ravel()
        decisions += [d1, d2]
        
    low = min(np.min(d) for d in decisions)
    high = max(np.max(d) for d in decisions)
    low_high = (low,high)
    
    plt.hist(decisions[0],
             color='r', alpha=0.5, range=low_high, bins=bins,
             histtype='stepfilled', normed=True,
             label='S (train)')
    plt.hist(decisions[1],
             color='b', alpha=0.5, range=low_high, bins=bins,
             histtype='stepfilled', normed=True,
             label='B (train)')

    hist, bins = np.histogram(decisions[2],
                              bins=bins, range=low_high, normed=True)
    scale = len(decisions[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale
    
    width = (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.errorbar(center, hist, yerr=err, fmt='o', c='r', label='S (test)')
    
    hist, bins = np.histogram(decisions[3],
                              bins=bins, range=low_high, normed=True)
    scale = len(decisions[3]) / sum(hist)
    err = np.sqrt(hist * scale) / scale

    plt.errorbar(center, hist, yerr=err, fmt='o', c='b', label='B (test)')

    plt.xlabel("Classifier output")
    plt.ylabel("Arbitrary units")
    plt.yscale('log')
    plt.legend(loc='best')
    if not os.path.exists(output): os.makedirs(output)
    plt.savefig(os.path.join(output,'compareTrainTest'+append+'.pdf'))
    plt.clf()

def plotDiscriminator(clf,X_test,y_test,output,bins=30):
    plt.hist(clf.decision_function(X_test[y_test==0]).ravel(),color='r', alpha=0.5, bins=bins)
    plt.hist(clf.decision_function(X_test[y_test==1]).ravel(),color='b', alpha=0.5, bins=bins)
    plt.xlabel("scikit-learn classifier output")
    if not os.path.exists(output): os.makedirs(output)
    plt.savefig(os.path.join(output,'discriminator.pdf'))
    plt.clf()

def plotPredVsTruth(y_pred,y_test,output,bins=30,append=''):

    y_pred = np.array(y_pred.ravel())
    #sort the binning
    maxValue=max(max(y_pred),max(y_test))
    minValue=min(0,min(min(y_pred),min(y_test)))

    print maxValue,minValue
     
    #plt.hist(y_pred,color='r', alpha=0.5, bins=bins, range=(minValue,maxValue), label='Predicted')
    plt.hist(y_test,color='r', alpha=0.5, bins=bins, range=(minValue,maxValue), label='Truth')

    #Make a hist with errors for the predicted
    binCentres,hist,err = hist1dErrorInputs(y_pred,weights=None,bins=bins, range=(minValue,maxValue))
    plt.errorbar(binCentres,hist,yerr=err,drawstyle='steps-mid',fmt='o',label='Pred')

    # plt.hist(y_pred,color='r', alpha=0.5, bins=bins, label='Predicted')
    # plt.hist(y_test,color='b', alpha=0.5, bins=bins, label='Truth')
    plt.xlabel("variable value")
    plt.legend(loc='best')
    if not os.path.exists(output): os.makedirs(output)
    plt.savefig(os.path.join(output,'predVsTruth'+append+'.pdf'))
    plt.clf()

def learningCurve(model, X_train, y_train, output,
                       ylim=None, cv=None, n_jobs=1,scoring=None,
                       train_sizes=np.linspace(0.1, 1.0, 5, endpoint=True)):
    # taken from https://gitlab.com/Contreras/hepML/blob/master/visualization/plotter.py

    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    model : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    X_train : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y_train : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
    Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
    Determines the cross-validation splitting strategy.
    Possible inputs for cv are:
      - None, to use the default 3-fold cross-validation,
      - integer, to specify the number of folds.
      - An object to be used as a cross-validation generator.
      - An iterable yielding train/test splits.

      For integer/None inputs, if ``y`` is binary or multiclass,
      :class:`StratifiedKFold` used. If the estimator is not a classifier
      or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

      Refer :ref:`User Guide <cross_validation>` for the various
      cross-validators that can be used here.

      n_jobs : integer, optional
         Number of jobs to run in parallel (default 1).

      train_sizes = np.linspace(0.1, 1.0, 10, endpoint=True) produces
         8 evenly spaced points in the range 0 to 10
      """

    # check to see if model is a pipeline object or not
    if isinstance(model, Pipeline):
        data_type = type(model._final_estimator)
    else:
        data_type = type(model)

    # plot title
    name = filter(str.isalnum, str(data_type).split(".")[-1])
    title = "Learning Curves (%s)" % name

    # create blank canvas
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.grid(which='major', linestyle='-', linewidth='0.2', color='gray')
    ax.set_facecolor('white')

    train_sizes_abs, train_scores, test_scores = learning_curve(model,
                                                                X_train, y_train,
                                                                train_sizes=np.linspace(0.1, 1.0, 10),
                                                                cv=cv,
                                                                scoring=scoring,
                                                                exploit_incremental_learning=False,
                                                                n_jobs=n_jobs,
                                                                pre_dispatch="all",
                                                                verbose=0)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std  = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std  = np.std(test_scores, axis=1)

    # plot the std deviation as a transparent range at each training set size
    plt.fill_between(train_sizes_abs, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")

    plt.fill_between(train_sizes_abs, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    # plot the average training and test score lines at each training set size
    plt.plot(train_sizes_abs, train_scores_mean, 'o-', color="r",
             label="Training score")

    plt.plot(train_sizes_abs, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.title(title, fontsize=14)

    # sizes the window for readability and displays the plot
    # shows error from 0 to 1.1
    if ylim is not None:
        plt.ylim(*ylim)
        #plt.ylim(-.1, 1.1)

    plt.xlabel("Training set size")
    plt.ylabel("Score")

    leg = plt.legend(loc="best", frameon=True, fancybox=False, fontsize=12)
    leg.get_frame().set_edgecolor('w')

    frame = leg.get_frame()
    frame.set_facecolor('White')

    # box-like grid
    #plt.grid(figsize=(8, 6))

    #plt.gca().invert_yaxis()
    if not os.path.exists(output): os.makedirs(output)
    plt.savefig(os.path.join(output,'learningCurve.pdf'))
    plt.clf()

    
