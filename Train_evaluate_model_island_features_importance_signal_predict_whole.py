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

def create_df_train(cell_name):

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


def create_df_test(cell_name):

    df_name = "{}.data.tf.info.signal.0.5.features.csv".format(cell_name)


    df = pd.read_csv(df_name, index_col=0)

    df_features = df[['CA', 'H3K27ac', 'H3K4me1', 'H3K4me3']].astype("float64")

    norm_flag = True
    if norm_flag == True:
        scaler = MinMaxScaler()
        features_data = scaler.fit_transform(df_features)
        df_features = pd.DataFrame(data=features_data, columns=['CA', 'H3K27ac', 'H3K4me1', 'H3K4me3'], index=df_features.index)
    else:
        pass
    

    return df_features


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
def train_and_save_model_predict_whole(X, y, df_features, feature_list, classifer, classifer_name, cell_name):
    X_train = X
    y_train = y
    model = classifer.fit(X_train, y_train)
    model.fit(X_train, y_train)
    model_name = "{}+{}.0.5.signal.whole_model.jobib".format(classifer_name, "-".join(feature_list))
    dump(model, model_name)
    y_hat = predict_test(model, df_features)    


    df_features = df_features.assign(Predict_Label=y_hat)
    df_features.insert(0, "Cell", cell_name)
    df_features.insert(1, "Features", "-".join(feature_list))
    df_features.insert(2, "Classifer", classifer_name)

    output_name = "{}+{}+{}.signal.0.5.predict.whole_genome.csv".format(cell_name, classifer_name, "-".join(feature_list))
    df_features.to_csv(output_name)
    


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


    X, y = create_df_train(cell_name)
    X_whole = create_df_test(cell_name)

    for feature_list in feature_list_list:
        X_s = select_feauters(X, feature_list)
        X_whole_s = select_feauters(X_whole, feature_list)

        classifer = classifiers[classifer_index]
        name = names[classifer_index]
        feature_name = "-".join(feature_list)
        train_and_save_model_predict_whole(X_s, y, X_whole_s, feature_list, classifer, name, cell_name)

if __name__=="__main__":
    main()
