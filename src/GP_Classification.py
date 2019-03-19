from matplotlib import pyplot
from scipy.io import loadmat
import pandas as pd
import numpy as np
import gpflow
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, auc, \
    average_precision_score, matthews_corrcoef

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
        # train_folds_aux_pd = pd.concat(train_folds_aux)

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
    elif kernel == "Combined":
        # Defining the kernel's combination
        kernel_combined = gpflow.kernels.RBF(Xtrain.shape[1]) * gpflow.kernels.Linear(Xtrain.shape[1])

        model = gpflow.models.VGP(Xtrain, Ytrain,
                                  kern=kernel_combined,
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

    accuracy = (TP + TN) / (TP + FN + FP + TN)
    specificity = TN / (TN + FP)
    sensitivity = TP / (TP + FN)
    precision = TP / (TP + FP)
    F_score = (2 * precision * sensitivity) / (precision + sensitivity);
    auc_roc = roc_auc_score(Ytest, pred_probabilities)
    precision_, recall, thresholds = precision_recall_curve(Ytest, pred_probabilities)
    auc_prc_recall = auc(recall, precision_)
    ap = average_precision_score(Ytest, pred_probabilities)
    mmc = matthews_corrcoef(Ytest, pred_probabilities)

    metrics = {'accuracy': accuracy, 'specificity': specificity,
               'sensitivity': sensitivity, 'precision': precision,
               'F_score': F_score, 'AUC_Recall': auc_roc,
               'AUC_PRECISION_RECALL': auc_prc_recall, 'Average Precision Score': ap,
               'mmc': mmc}

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


def generate_ROC_curve(path, fig_name, kernel, Ytest, pred):
    """
    generate_ROC_curve, is a function used to generate the ROC curve.

    :param path: the path to save the plot figure
    :param fig_name: the figure name
    :param kernel: the used kernel, used in the plot figure's name
    :param Ytest: the real labels
    :param pred: the predicted labels.

    """
    # calculate roc curve
    fpr, tpr, thresholds = roc_curve(Ytest, pred)

    pyplot.plot([0, 1], [0, 1], linestyle='--')
    pyplot.plot(fpr, tpr, marker='.')
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.savefig(path + "/" + kernel + "/" + "ROC" + "/" + fig_name)


def generate_precision_recall_curve(path, fig_name, kernel, Ytest, pred):
    """
    generate_precision_recall_curve, is a function used to generate the precision-recall curve.

    :param path: the path to save the plot figure
    :param fig_name: the figure name
    :param kernel: the used kernel, used in the plot figure's name
    :param Ytest: the real labels
    :param pred: the predicted labels.

    """
    precision, recall, thresholds = precision_recall_curve(Ytest, pred)

    pyplot.plot([0, 1], [0.2, 0.2], linestyle='--')
    pyplot.plot(recall, precision, marker='.')
    pyplot.xlabel('Recall')
    pyplot.ylabel('Presicion')
    pyplot.savefig(path + "/" + kernel + "/" + "Pres-Recall" + "/" + fig_name)


if __name__ == "__main__":

    # Reading Datasets, Dividing them in test and train, creating the folds,
    test_folds, train_folds = create_folds(data)
    num_features = train_folds[0][0].shape[1] - 1

    plot_path = '../plots'

    # Apply the different models.
    run_flag = False
    if run_flag:
        predictions_RBF = cross_validation(train_folds, test_folds, "RBF")
        rbf_predictions = np.array(predictions_RBF)
        np.save("../data/saved_data_code/rbf_predictions", rbf_predictions)

        predictions_Linear = cross_validation(train_folds, test_folds, "Linear")
        linear_predictions = np.array(predictions_Linear)
        np.save("../data/saved_data_code/linear_predictions", linear_predictions)

        predictions_Combined = cross_validation(train_folds, test_folds, "Combined")
        combined_predictions = np.array(predictions_Combined)
        np.save("../data/saved_data_code/predictions_Combined", predictions_Combined)

    # Loading the saved predictions and Evaluation
    run_flag = True
    if run_flag:
        linear_predictions = np.load("../data/saved_data_code/linear_predictions.npy")
        rbf_predictions = np.load("../data/saved_data_code/rbf_predictions.npy")
        Combined_predictions = np.load("../data/saved_data_code/predictions_Combined.npy")

        kernels = ['LINEAR', 'RBF', 'COMBINED']
        global_prediction_list = [linear_predictions, rbf_predictions, Combined_predictions]

        for i in range(len(kernels)):
            for j in range(global_prediction_list[0].shape[0]):

                Xtest, Ytest = separate_data_labels(test_folds[j], num_features)

                print("Results for %s" % (kernels[i]))
                print(" ****************************** ")
                print("Fold %i: %s" % (j, EvaluationMetrics(Ytest, global_prediction_list[i][j])))

                run_flag = False
                if run_flag:  # ROC plots
                    fig_name = 'roc' + kernels[i] + str(j)
                    generate_ROC_curve(plot_path, fig_name, kernels[i], Ytest, global_prediction_list[i][j])

                run_flag = False
                if run_flag:  # Precision-Recall Plots
                    fig_name = 'Precision-Recall' + kernels[i] + str(j)
                    generate_precision_recall_curve(plot_path, fig_name, kernels[i], Ytest,
                                                    global_prediction_list[i][j])

            pyplot.clf()
            print(" >>>>>>>>>>>> ")
