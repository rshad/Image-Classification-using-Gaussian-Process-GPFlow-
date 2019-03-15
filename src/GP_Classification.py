from scipy.io import loadmat
import pandas as pd
import numpy as np
import gpflow
from sklearn.metrics import confusion_matrix

data = loadmat('../data/data.mat')


def create_folds(data):
    """
    create_folds, is a function used to create the training and test folds from data.

    :param data: the dataset to decompose in folds
    :return: lists, lists of folds, test_folds and train_folds
    """
    test_folds = []
    train_folds = []

    healthy = 'Healthy_folds'
    malign = 'Malign_folds'

    if data[healthy][0].shape[0] == data[malign][0].shape[0]:
        folds_num = data[healthy][0].shape[0]
    else:
        print(" Healthy and Malign Folds Numbers are Different ")
        return None

    for i in range(folds_num):  # iterating over the folds in parallel for healthy and malign.

        # Convert the numpy arrays into a dataframe and we add a new column, the class column in order to differentiate
        # the healthy observ. from the malign one.
        fold_h = pd.DataFrame(data[healthy][0][i][0])  # fold i of data[healthy][0]
        fold_h['class'] = -1

        fold_m = pd.DataFrame(data[malign][0][i][0])  # fold i of data[malign][0]
        fold_m['class'] = 1

        # concatenating healthy fold i with malign fold i
        test_folds.append(pd.concat((fold_h, fold_m), axis=0))  # appending the fold i to folds

        print("Shape of test fold %i, after adding the class column : %s" % (i, str(test_folds[i].shape)))

    print("\n")

    """
     Once we have the the folds prepared where each fold contains both healthy and malign observations, we consider 
     then, each one of these folds is a test fold and the correspondent train  one would be the rest of test folds.

     So an example would be as follows:
        test_fold_1 = test_folds[1]
        train_fold_1 = test_folds[i != 1]
    """
    for i in range(folds_num):  # looping in range of the number of folds

        # getting the list of all folds except the i one
        train_folds_aux = [x for j, x in enumerate(test_folds) if j != i]

        # concatenate all the dataframes corresponding to the result list of folds
        #train_folds_aux_pd = pd.concat(train_folds_aux)

        # adding the created fold to train_folds
        train_folds.append(train_folds_aux)

    return test_folds, train_folds


def separate_data_labels(data, num_features):
    """
    separate_data_labels, is a function used to separate a dataset into data and labels.
    :param data: the global dataset with labels included
    :param num_features: the number of features in data, discounted the class column.
    :return: X and Y, where X is the data without labels and Y represents the labels
    """
    X = data.iloc[:, 0:num_features]
    Y = data.iloc[:, num_features]

    X = np.array(X)
    Y = np.array(Y).reshape(-1, 1)

    return X, Y


def model_VGP(Xtrain, Ytrain, kernel):
    """
    model_VGP, is a function used to get VGP model with " kernel " using Xtrain and Ytrain for training.

    :param Xtrain: the train data without labels
    :param Ytrain: the labels Xtrain
    :param kernel: the kernel to use, in this case we only works with RBF or Linear

    :return: model, the final defined model.
    """
    likelihood = gpflow.likelihoods.Bernoulli()

    # build a model with this likelihood
    if kernel == "RBF":
        model = gpflow.models.VGP(Xtrain, Ytrain,
                                  kern=gpflow.kernels.RBF(Xtrain.shape[1]),  # Specify the input dimension for the
                                                                             # classifier. " Multidimensional "
                                  likelihood=likelihood)
    elif kernel == "Linear":
        model = gpflow.models.VGP(Xtrain, Ytrain,
                                  kern=gpflow.kernels.Linear(Xtrain.shape[1]),  # Specify the input dimension for the
                                                                                # classifier. " Multidimensional "
                                  likelihood=likelihood)
    else:
        " modelling failed: The provided kernel is wrong or not available "
        return None

    # fit the model
    gpflow.train.ScipyOptimizer().minimize(model)

    return model


def EvaluationMetrics(Ytest, pred_probabilities):
    """
    EvaluationMetrics, is a function used o calculate the different evaluation metrics.

    :param Ytest: the real labels.
    :param pred_probabilities: the predicted labels.
    :return: metrics, a dictionary with an entry for each calculated metric.
    """
    TN, FP, FN, TP = confusion_matrix(Ytest, pred_probabilities).ravel()
    print((TN, FP, FN, TP))

    accuracy = (TP + TN) / (TP + FN + FP + TN)
    specificity = TN / (TN + FP)
    sensitivity = TP / (TP + FP)
    precision = TP / (TP + FP)
    F_score = (2 * precision * sensitivity) / (precision + sensitivity);

    metrics = {'accuracy': accuracy, 'specificity': specificity,
               'sensitivity': sensitivity,'precision': precision,
               'F_score': F_score}

    return metrics


def cross_validation(train_folds, test_folds, kernel):
    """
    cross_validation
    :param train_folds: a group of train data
    :param test_folds: a group of test data
    :param kernel: the kernel to use when modelling, RBF or Lineal
    :return: models, a list of models, with model for each fold
    """

    final_predictions = []  # The list of models to return

    if len(train_folds) == len(test_folds):
        folds_num = len(train_folds)
    else:
        print(" Size of test and train folds not equal ")
        return None

    num_features = train_folds[0][0].shape[1] - 1

    theta = 0.5

    for i in range(folds_num):  # Iterating over the folds.

        i_predictions = []  # auxiliary list will be used to store the 4 predictions per each fold.

        # Getting the train and test data and labels
        Xtest, Ytest = separate_data_labels(test_folds[i], num_features)  # separated train data in data and labels

        for j in range(4):
            Xtrain, Ytrain = separate_data_labels(train_folds[i][j],  # working with the fold j of the list of train
                                                  # folds of the fold i.
                                                  num_features)  # separated test data in data and labels

            # Modelling
            model = model_VGP(Xtrain, Ytrain, kernel)

            # Getting the predictions labels of "Xtest"
            prediction = model.predict_y(Xtest)

            # prediction probabilities for the labels
            pred_probabilities = np.array(prediction[0])

            # adding pred_probabilities to the list of 4 predictions for the fold i.
            i_predictions.append(pred_probabilities)

        # The final prediction for the fold i is calculated as the mean of all the calculated predictions in
        # i_predictions
        prediction_i = sum(i_predictions) / len(i_predictions)

        # Replacing the probabilities with -1 if it's lower or equal to 0.5 and with 1 otherwise.
        np.place(prediction_i, prediction_i <= theta, -1)  # Healthy => -1
        np.place(prediction_i, prediction_i > theta, 1)  # Malign => 1


        final_predictions.append(prediction_i)

    return final_predictions


if __name__ == "__main__":
    test_folds, train_folds = create_folds(data)

    predictions = cross_validation(train_folds, test_folds, "Linear")

    for i in predictions:
        print(i)
