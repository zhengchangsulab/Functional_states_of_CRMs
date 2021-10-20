#!/usr/bin/env python
# coding: utf-8

# In[61]:


#!/usr/bin/python
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, OrdinalEncoder
from sklearn.feature_selection import SelectKBest, chi2,f_classif,mutual_info_classif
from sklearn.linear_model import LogisticRegression, Lasso, Ridge
#from sklearn.metrics import accuracy_score, precision_recall_curve,plot_precision_recall_curve, plot_roc_curve, roc_auc_score, auc
from sklearn.metrics import accuracy_score, plot_roc_curve, roc_auc_score, auc, plot_precision_recall_curve
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report,roc_curve
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import matplotlib.pyplot as plt
import numpy as np
import json
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import os
from joblib import load, dump



plt.rcParams['svg.fonttype'] = 'none'

def create_df(cell_name):
    #df_name = "{}.data.tf.info.signal.clean.csv".format(cell_name)
    df_name = "{}.data.tf.info.signal.0.5.clean.csv".format(cell_name)


    df = pd.read_csv(df_name, index_col=0)

    df_features = df[['CA', 'H3K27ac', 'H3K4me1', 'H3K4me3']].astype("float64")
    df_label = df['TFs'].clip(0,1).astype("int64")

    norm_flag = True
    if norm_flag == True:
        scaler = MinMaxScaler()
        features_data = scaler.fit_transform(df_features)
        df_features = pd.DataFrame(data=features_data, columns=['CA', 'H3K27ac', 'H3K4me1', 'H3K4me3'])
    else:
        pass
    

    return df_features, df_label


def select_feauters(X, feature_list):
    X_select = X[feature_list]
    return X_select


def build_model(X_train, y_train):
    model = LogisticRegression(fit_intercept=False)
    model.fit(X_train, y_train)
    return model


def predict_test(model, X_test):
    yhat = model.predict(X_test)
    #y_prob = model.predict_proba(X_test)
    return yhat#, y_prob

def report(y_test, yhat):
    report = classification_report(y_test, yhat, target_names=['1', '0'], output_dict=True)
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1_score = report['weighted avg']['f1-score']
    accuracy = report['accuracy']
    return precision, recall, f1_score, accuracy



# In[95]:


def train_and_evaluate(kf, X, y, feature_list, classifer, name):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)


    #df_report = pd.DataFrame(columns=['Classifer', 'Features', 'Fold', 'Precision', 'Recall', 'F1_score', 'Accuracy', 'AUC'])
    df_report = pd.DataFrame(columns=['Classifer', 'Features', 'Precision', 'Recall', 'F1_score', 'Accuracy', 'AUROC', 'AUPR'])



    df_coef = pd.DataFrame(columns=feature_list)


    fig, ax = plt.subplots(1,1, figsize=(8,8), dpi=300)
    fig0, ax0 = plt.subplots(1,1, figsize=(8,8), dpi=300)

    feature_name = "-".join(feature_list)


    i = 0
    for train_index, test_index in kf.split(X):
        i += 1
        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model = classifer.fit(X_train, y_train)
        model.fit(X_train, y_train)



        model_name = "{}-{}.island.primary.model-{}.tf.importance.signal.0.5.jobib".format(name.replace(" ", "_"), feature_name, i)
        test_index_name = "{}-{}.island.primary.test_index-{}.tf.importance.signal.0.5.csv".format(name.replace(" ", "_"), feature_name, i)
        dump(model, model_name)
        df_test_index = pd.Series(test_index, name="test_index")
        df_test_index.to_csv(test_index_name)


        if name in ["LogesticRegression","Linear SVM", "Ridge Regression", "Lasso"]:
            coef = model.coef_[0]
            ds_coef = pd.Series(coef, index=df_coef.columns)
            df_coef = df_coef.append(ds_coef, ignore_index=True)

        elif name in ["Decision Tree", "Random Forest", "AdaBoost"]:           
            coef = model.feature_importances_
            ds_coef = pd.Series(coef, index=df_coef.columns)
            df_coef = df_coef.append(ds_coef, ignore_index=True)


        viz = plot_roc_curve(model, X_test, y_test,
                             name='ROC fold {}'.format(i),
                             
                             alpha=0.5, lw=1, ax=ax)


        viz2 = plot_precision_recall_curve(model, X_test, y_test,
                                           name='AUPR fold {}'.format(i),
                                           lw=1,
                                           alpha=1,
                                           ax=ax0)

        aupr = auc(viz2.recall, viz2.precision)


        yhat = predict_test(model, X_test)
        precision, recall, f1_score, accuracy = report(y_test, yhat)
        df_report = df_report.append({"Classifer":name, 'Features':"-".join(feature_list), "Fold":i, 'Precision':precision, 'Recall':recall,
                                      'F1_score':f1_score, 'Accuracy':accuracy, 'AUROC':viz.roc_auc, 'AUPR':aupr}, ignore_index=True)

        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)


    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Random', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title="{}-{}".format(name, "-".join(feature_list)))
    ax.legend(loc="lower right")


    #figure_name = "{}-{}.island.primary.ROC.tf.importance.signal.svg".format(name.replace(" ", "_"),feature_name)
    #report_name = "{}-{}.island.primary.report.tf.importance.signal.csv".format(name.replace(" ", "_"), feature_name)
    #coef_name = "{}-{}.island.primary.coef.tf.importance.signal.csv".format(name.replace(" ", "_"), feature_name)





    figure_auroc_name = "{}-{}.island.primary.AUROC.tf.importance.signal.0.5.svg".format(name.replace(" ", "_"),feature_name)
    figure_aupr_name = "{}-{}.island.primary.AUPR.tf.importance.signal.0.5.svg".format(name.replace(" ", "_"),feature_name)
    report_name = "{}-{}.island.primary.report.tf.importance.signal.0.5.csv".format(name.replace(" ", "_"), feature_name)
    coef_name = "{}-{}.island.primary.coef.tf.importance.signal.0.5.csv".format(name.replace(" ", "_"), feature_name)


    #figure_name = "{}-{}.island.primary.ROC.tf.importance.signal.0.8.svg".format(name.replace(" ", "_"),feature_name)
    #report_name = "{}-{}.island.primary.report.tf.importance.signal.0.8.csv".format(name.replace(" ", "_"), feature_name)
    #coef_name = "{}-{}.island.primary.coef.tf.importance.signal.0.8.csv".format(name.replace(" ", "_"), feature_name)


    #figure_name = "{}-{}.island.primary.ROC.tf.importance.signal.0.8.unnorm.svg".format(name.replace(" ", "_"),feature_name)
    #report_name = "{}-{}.island.primary.report.tf.importance.signal.0.8.unnorm.csv".format(name.replace(" ", "_"), feature_name)
    #coef_name = "{}-{}.island.primary.coef.tf.importance.signal.0.8.unnorm.csv".format(name.replace(" ", "_"), feature_name)

    fig.savefig(figure_auroc_name)
    fig0.savefig(figure_aupr_name)
    df_report.to_csv(report_name)

    if name in ["LogesticRegression","Linear SVM", "Decision Tree", "Random Forest", "AdaBoost", "Ridge Regression", "Lasso"]:
        df_coef.to_csv(coef_name)
    else:
        pass

# In[96]:



feature_list_list = [['CA'], ['H3K27ac'], ['H3K4me1'], ['H3K4me3'],
                         ['CA', 'H3K27ac'],['CA','H3K4me1'], ['CA','H3K4me3'],['H3K27ac', 'H3K4me1'],['H3K27ac', 'H3K4me3'],['H3K4me1', 'H3K4me3'],
                         ['CA','H3K27ac','H3K4me1'],
                         ['CA','H3K27ac','H3K4me3'],
                         ['CA', 'H3K4me1','H3K4me3'],
                         ['H3K27ac', 'H3K4me1', 'H3K4me3'],
                         ['CA','H3K27ac', 'H3K4me1', 'H3K4me3']
                    ]
names = ["LogesticRegression","Linear SVM",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "Ridge Regression", "Lasso"]

classifiers = [
    LogisticRegression(fit_intercept=True),
    LinearSVC(fit_intercept=True, random_state=np.random.RandomState(0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    Ridge(alpha=1.0, fit_intercept=False),
    Lasso(alpha=1.0, fit_intercept=False)
    ]


# In[99]:

def main():
    cell_name = sys.argv[1]
    classifer_index = int(sys.argv[2])

    X, y = create_df(cell_name)
    #X, y = create_df_v2(cell_name)

    for feature_list in feature_list_list:
        X_s = select_feauters(X, feature_list)

        kf = KFold(n_splits=10, shuffle=True, random_state=np.random.RandomState(123456))

        classifer = classifiers[classifer_index]
        name = names[classifer_index]

        feature_name = "-".join(feature_list)
        train_and_evaluate(kf, X_s, y, feature_list, classifer, name)

if __name__=="__main__":
    main()
