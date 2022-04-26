import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.metrics import mean_squared_error
from math import sqrt
from models import *
from UI import *
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns



def create_train_test_sets(X, Y, y_name):
    df = pd.concat([X, Y], axis=1)
    train, test = train_test_split(df, test_size = 0.3)

    x_train = train.drop(y_name, axis=1)
    y_train = train[y_name]

    x_test = test.drop(y_name, axis = 1)
    y_test = test[y_name]
    return x_train, y_train, x_test, y_test

def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

def create_dummies(X):
    """
    creates dummies variables from Beat, Location Description, and Ward.
    :param X: data set.
    :return: updated data set.
    """
    dummies_age_list = ['20','20-30','30-50','50-65','65']
    for var in dummies_age_list:
        row = []
        for x in X['age']:
            if str(x) == var:
                row.append(1)
            else:
                row.append(0)
        dummy = pd.DataFrame({"age: " + str(var): row})
        X = pd.concat([X, dummy], axis=1)
    del X['age']

    dummies_gender_list = ['M','F']
    for var in dummies_gender_list:
        row = []
        for x in X['gender']:
            if x == var:
                row.append(1)
            else:
                row.append(0)
        dummy = pd.DataFrame({"gender: " + str(var): row})
        X = pd.concat([X, dummy], axis=1)
    del X['gender']

    dummies_employment_list = ['nature', 'social', 'humanities', 'education', 'services', 'law', 'medical',
                               'security', 'artist']
    for var in dummies_employment_list:
        row = []
        for x in X['employment']:
            if x == var:
                row.append(1)
            else:
                row.append(0)
        dummy = pd.DataFrame({"employment: " + str(var): row})
        X = pd.concat([X, dummy], axis=1)
    del X['employment']

    dummies_labor_list = ['Yemina', 'Avoda', 'Kachol-lavan', 'Other', 'Tzionut-datit', 'Yesh-atid', 'Tikva-hadasha',
                               'Likud', 'Meretz', 'Israel-beytenu']
    for var in dummies_labor_list:
        row = []
        for x in X['labor']:
            if x == var:
                row.append(1)
            else:
                row.append(0)
        dummy = pd.DataFrame({"labor: " + str(var): row})
        X = pd.concat([X, dummy], axis=1)
    del X['labor']

    dummies_location_list = ['Jerusalem', 'Dan', 'North', 'South', 'Other', 'Judea and samaria']
    for var in dummies_location_list:
        row = []
        for x in X['location']:
            if x == var:
                row.append(1)
            else:
                row.append(0)
        dummy = pd.DataFrame({"location: " + str(var): row})
        X = pd.concat([X, dummy], axis=1)
    del X['location']

    dummies_status_list = ['Married', 'Single']
    for var in dummies_status_list:
        row = []
        for x in X['status']:
            if x == var:
                row.append(1)
            else:
                row.append(0)
        dummy = pd.DataFrame({"status: " + str(var): row})
        X = pd.concat([X, dummy], axis=1)
    del X['status']
    return X

def create_pairs(Y):
    result = pd.DataFrame()
    for i in range(len(Y.columns)-1):
        for j in range(i+1,len(Y.columns)):
            col = pd.DataFrame([1 if row[Y.columns[i]] >= row[Y.columns[j]] else 0 for index,row in Y.iterrows()],
                               columns=[f'{Y.columns[i]},{Y.columns[j]}'])
            result = pd.concat([result, col], axis=1)
    return result

def create_map():
    map = {
        "or yarok": 0,
        "tza'ar baaley chaim": 0,
        "enosh": 0,
        "kav lachaim": 0,
        "gesher":0,
        "hasomer hachdash": 0,
        "Leket Israel": 0,
        "haaguda lemilchama basartan": 0,
        "keren ramon": 0,
        "yad sara": 0,
        "latet": 0,
        "retorno": 0
    }
    return map

# gets predicted row of all pairs, and emits the maximum
def find_max(Y):
    map = create_map()
    for col in Y.columns:
        names = col.split(',')
        if(Y.iloc[0][col] == 1):
            map[names[0]] = map[names[0]] + 1
        else:
            map[names[1]] = map[names[1]] + 1
    max_key = max(map, key=map.get)
    return max_key

def plot_correlation_of_features(X,Y):
    df = pd.concat([X,Y], axis=1)
    cormat = df.corr()
    round(cormat,2)
    sns.heatmap(cormat, vmin= -0.5, vmax=0.5)
    plt.show()
    # cor_dict = dict()
    # for f in X.columns:
    #     cor = 0
    #     for label in Y.columns:
    #         cor = X[f].corr(Y[label])
    #         print(f, label, cor)
        # cor_dict[f] = cor / 12
    # print(cor_dict)


def pre_process(path):
    """
    preproccessing the data.
    :return: data set, response vector, list of all the features.
    """
    data = pd.read_excel(path, sep='\s+')
    data_with_dummies = create_dummies(data)
    X = normalize(data_with_dummies)
    Y = X["or yarok"]
    del X["or yarok"]
    Y = pd.concat([X["tza'ar baaley chaim"], Y], axis=1)
    del X["tza'ar baaley chaim"]
    Y = pd.concat([X["enosh"], Y], axis=1)
    del X["enosh"]
    Y = pd.concat([X["kav lachaim"], Y], axis=1)
    del X["kav lachaim"]
    Y = pd.concat([X["gesher"], Y], axis=1)
    del X["gesher"]
    Y = pd.concat([X["hasomer hachdash"], Y], axis=1)
    del X["hasomer hachdash"]
    Y = pd.concat([X["Leket Israel"], Y], axis=1)
    del X["Leket Israel"]
    Y = pd.concat([X["haaguda lemilchama basartan"], Y], axis=1)
    del X["haaguda lemilchama basartan"]
    Y = pd.concat([X["keren ramon"], Y], axis=1)
    del X["keren ramon"]
    Y = pd.concat([X["yad sara"], Y], axis=1)
    del X["yad sara"]
    Y = pd.concat([X["latet"], Y], axis=1)
    del X["latet"]
    Y = pd.concat([X["retorno"], Y], axis=1)
    del X["retorno"]

    # plot_correlation_of_features(X,Y)
    pairs_Y = create_pairs(Y)
    return X, pairs_Y, Y

def runKnnRegression(x_train, y_train, x_test, y_test, name_y, results):
    for K in range(3):
        K = K+1
        model = neighbors.KNeighborsRegressor(n_neighbors = K)

        model.fit(x_train, y_train)  #fit the model
        pred=model.predict(x_test) #make prediction on test set
        error = sqrt(mean_squared_error(y_test,pred)) #calculate rmse
        results[K-1] += error #store rmse values
        print('RMSE value for k = ', K, 'for ', name_y,' is: ', error)
        # print("predicted: ", pred)
        # print("y test: ", y_test)
        # print("The mean of predicted array of rank to: ", name_y, "is: ", np.mean(pred))

def runClassification(x_train, y_train, x_test, y_test, name_y):
    accuracy_list = []
    recall_list = []
    precision_list = []
    print("run col: ", name_y)
    lst = runKnn(x_train.to_numpy(), y_train.to_numpy(), x_test.to_numpy(), y_test.to_numpy())
    accuracy_list += lst[0]
    recall_list += lst[1]
    precision_list += lst[2]
    lst = runTrees(x_train.to_numpy(), y_train.to_numpy(), x_test.to_numpy(), y_test.to_numpy())
    accuracy_list += lst[0]
    recall_list += lst[1]
    precision_list += lst[2]
    lst = runSVM(x_train.to_numpy(), y_train.to_numpy(), x_test.to_numpy(), y_test.to_numpy())
    accuracy_list += lst[0]
    recall_list += lst[1]
    precision_list += lst[2]
    lst = runBag(x_train.to_numpy(), y_train.to_numpy(), x_test.to_numpy(), y_test.to_numpy())
    accuracy_list += lst[0]
    recall_list += lst[1]
    precision_list += lst[2]


    return accuracy_list, recall_list, precision_list

def accuracy_recall_precision_of_learner(X,Y):
    accuracy = [0,0,0,0,0,0,0,0]
    recall = [0,0,0,0,0,0,0,0]
    precision = [0,0,0,0,0,0,0,0]
    for col in Y:
        x_train, y_train, x_test, y_test = create_train_test_sets(X, Y[col], col)
        # runKnnRegression(x_train, y_train, x_test, y_test, col)
        accuracy_list, recall_list, precision_list = runClassification(x_train, y_train, x_test, y_test, col)
        for i in range(len(accuracy_list)):
            accuracy[i] += accuracy_list[i]
            recall[i] += recall_list[i]
            precision[i] += precision_list[i]
    for i in range(len(accuracy_list)):
        accuracy[i] /= 66
        recall[i] /= 66
        precision[i] /= 66
    learners = ['KNN - 1','KNN - 2', 'KNN - 3',
                'Decision Tree - 1', 'Decision Tree - 2', 'Decision Tree - 3',
                'SVM - 1', 'Bag']
    df1 = pd.DataFrame({'Learners': learners, 'Accuracy': accuracy})
    df1.plot(x="Learners", y="Accuracy", kind="bar")
    df2 = pd.DataFrame({'Learners': learners, 'Precision': precision})
    df2.plot(x="Learners", y="Precision", kind="bar")
    df2 = pd.DataFrame({'Learners': learners, 'Recall': recall})
    df2.plot(x="Learners", y="Recall", kind="bar")
    plt.show()
    # for i in range(len(accuracy)):
    #     print(learners[i] ," - accuracy: ", accuracy[i] / 66)
    #     print(learners[i] ," - recall: ", recall[i] / 66)
    #     print(learners[i] ," - precision: ", precision[i] / 66)

def create_dummies_for_real_index(new_df):
    dummies_age_list = ['20','20-30','30-50','50-65','65']
    dummies_gender_list = ['M','F']
    dummies_employment_list = ['nature', 'social', 'humanities', 'education', 'services', 'law', 'medical',
                               'security', 'artist']
    dummies_labor_list = ['Yemina', 'Avoda', 'Kachol-lavan', 'Other', 'Tzionut-datit', 'Yesh-atid', 'Tikva-hadasha',
                          'Likud', 'Meretz', 'Israel-beytenu']
    dummies_location_list = ['Jerusalem', 'Dan', 'North', 'South', 'Other', 'Judea and samaria']
    dummies_status_list = ['Single', 'Married']

    X = pd.DataFrame()
    row = [[1,] if new_df['age'][0] == i else [0,] for i in dummies_age_list]
    dummies = pd.DataFrame({"age: " + val: row[index] for index,val in enumerate(dummies_age_list)})
    X = pd.concat([X, dummies], axis=1)

    row = [[1,] if new_df['gender'][0] == i else [0,] for i in dummies_gender_list]
    dummies = pd.DataFrame({"gender: " + val: row[index] for index,val in enumerate(dummies_gender_list)})
    X = pd.concat([X, dummies], axis=1)

    row = [[1,] if new_df['employment'][0] == i else [0,] for i in dummies_employment_list]
    dummies = pd.DataFrame({"employment: " + val: row[index] for index,val in enumerate(dummies_employment_list)})
    X = pd.concat([X, dummies], axis=1)

    row = [[1,] if new_df['labor'][0] == i else [0,] for i in dummies_labor_list]
    dummies = pd.DataFrame({"labor: " + val: row[index] for index,val in enumerate(dummies_labor_list)})
    X = pd.concat([X, dummies], axis=1)

    row = [[1,] if new_df['location'][0] == i else [0,] for i in dummies_location_list]
    dummies = pd.DataFrame({"location: " + val: row[index] for index,val in enumerate(dummies_location_list)})
    X = pd.concat([X, dummies], axis=1)

    row = [[1,] if new_df['status'][0] == i else [0,] for i in dummies_status_list]
    dummies = pd.DataFrame({"status: " + val: row[index] for index,val in enumerate(dummies_status_list)})
    X = pd.concat([X, dummies], axis=1)

    new_df['shuk hofshi'] = (new_df['shuk hofshi'].astype(int) - 1) / 4
    X = pd.concat([X, new_df['shuk hofshi']], axis=1)
    return X

def pre_process_real_index(new_df):
    data_with_dummies = create_dummies_for_real_index(new_df)
    return data_with_dummies

def learn(model, learned_df, Y):
    y_predicted = model.predict(learned_df)
    df = pd.DataFrame({value: [y_predicted[0][index]] for index,value in enumerate(Y.columns)})
    name = find_max(df)
    return name

def predict_organization(answers_list):
    X, Y, Y_orig = pre_process('data.xlsx')
    cols = ['age', 'gender', 'employment', 'labor', 'location', 'status', 'shuk hofshi']
    df = pd.DataFrame({cols[i] : [answers_list[i]] for i in range(len(cols))})
    preprocessed_df = pre_process_real_index(df)
    dt = DecisionTree(1, )
    dt.fit(X,Y)
    name = learn(dt, preprocessed_df, Y)
    return name


if __name__ == '__main__':
    # X, Y, Y_orig = pre_process('data.xlsx')
    # results = [0,0,0]
    # for col in Y_orig:
    #     x_train, y_train, x_test, y_test = create_train_test_sets(X, Y[col], col)
    #     runKnnRegression(x_train.to_numpy(), y_train.to_numpy(), x_test.to_numpy(), y_test.to_numpy(), col, results)
    # for i in range(1,4):
    #     print("k=", i+1, results[i]/12)
    # accuracy_recall_precision_of_learner(X,Y)

    app = Survey()
    app.mainloop()
