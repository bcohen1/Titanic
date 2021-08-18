import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection
from sklearn import (
    linear_model,
    svm,
    neighbors,
    gaussian_process,
    naive_bayes,
    tree,
    ensemble,
)


def standardize_data(df):
    """Fill missing values with median or mode"""
    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Fare"].fillna(df["Fare"].median(), inplace=True)
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
    return df


def create_columns(df):
    """Create new columns for total family size, whether someone was alone, and title"""
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = np.where(df["FamilySize"] == 1, 1, 0)

    df["Title"] = df["Name"].str.extract(r"([A-Za-z]+)\.", expand=False)
    df["Title"].replace(["Sir"], "Mr", inplace=True)
    df["Title"].replace(["Mlle", "Ms"], "Miss", inplace=True)
    df["Title"].replace(["Mme"], "Mrs", inplace=True)
    df["Title"].replace(["Don", "Jonkheer", "Countess", "Lady"], "Master", inplace=True)
    df["Title"].replace(["Sir"], "Mr", inplace=True)
    df["Title"].replace(["Dr", "Rev", "Col", "Major", "Capt"], "Misc", inplace=True)
    return df


def create_bins(df):
    """Create bins for age and fare"""
    df["AgeBin"] = pd.cut(df["Age"], 8)
    df["FareBin"] = pd.cut(df["Fare"], 10)
    return df


def encode_data(df):
    """Encode categorical fields"""
    le = LabelEncoder()
    df["Sex_Code"] = le.fit_transform(df["Sex"])
    df["AgeBin_Code"] = le.fit_transform(df["AgeBin"])
    df["FareBin_Code"] = le.fit_transform(df["FareBin"])
    df["Embarked_Code"] = le.fit_transform(df["Embarked"])
    df["Title_Code"] = le.fit_transform(df["Title"])
    return df


def test_models(mla, train_df, train_df_x_bin, target):
    """Index through MLA list and saves cross-validation performance to table"""
    mla_columns = [
        "mla_name",
        "parameters",
        "train_accuracy_mean",
        "test_accuracy_mean",
        "test_accuracy_3*std",
        "fit_time",
    ]
    mla_compare = pd.DataFrame(columns=mla_columns)

    row_index = 0
    for alg in mla:

        cv_results = model_selection.cross_validate(
            alg,
            train_df[train_df_x_bin],
            train_df[target].values.ravel(),
            cv=model_selection.KFold(n_splits=5, shuffle=True),
            return_train_score=True,
        )

        mla_name = alg.__class__.__name__
        mla_compare.loc[row_index, "mla_name"] = mla_name
        mla_compare.loc[row_index, "parameters"] = str(alg.get_params())
        mla_compare.loc[row_index, "train_accuracy_mean"] = cv_results[
            "train_score"
        ].mean()
        mla_compare.loc[row_index, "test_accuracy_mean"] = cv_results[
            "test_score"
        ].mean()
        mla_compare.loc[row_index, "test_accuracy_3*std"] = (
            cv_results["test_score"].std() * 3
        )
        mla_compare.loc[row_index, "fit_time"] = cv_results["fit_time"].mean()

        row_index += 1
    mla_compare.sort_values(by=["test_accuracy_mean"], ascending=False, inplace=True)
    return mla_compare


def optimize_params(mla, mla_compare, train_df, train_df_x_bin, target):
    """Optimizes best performing model with Grid Search"""
    best_alg = mla[mla_compare["test_accuracy_mean"].astype(float).idxmax()]
    try:
        param_grid = {"C": [0.1, 1, 10, 100, 1000], "gamma": [1, 0.1, 0.01, 0.001, 0.0001]}
        grid = model_selection.GridSearchCV(best_alg, param_grid, verbose=3)
    except ValueError:
        param_grid = {"nu": [1, 0.1, 0.01, 0.001, 0.0001], "gamma": [1, 0.1, 0.01, 0.001, 0.0001]}
        grid = model_selection.GridSearchCV(best_alg, param_grid, verbose=3)
    grid.fit(train_df[train_df_x_bin], train_df[target].values.ravel())
    return grid.best_estimator_


def generate_submission_csv(test_df, train_df_x_bin, best_estimator):
    """Uses optimized model to make predictions and generate submission.csv"""
    submission_df = pd.DataFrame(test_df["PassengerId"])
    submission_df["Survived"] = best_estimator.predict(test_df[train_df_x_bin])
    submission_df.to_csv("submission.csv", index=False)


def main():
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
    combine = [train_df, test_df]

    for df in combine:
        df.info()
        standardize_data(df)
        create_columns(df)
        create_bins(df)
        encode_data(df)
    # Define target (Y variable)
    target = ["Survived"]

    # Define features (X variables)
    train_df_x = [
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked",
        "FamilySize",
        "IsAlone",
        "Title",
    ]

    # Define numerical features (binned and encoded)
    train_df_x_bin = [
        "Pclass",
        "Sex_Code",
        "AgeBin_Code",
        "FareBin_Code",
        "Embarked_Code",
        "FamilySize",
        "IsAlone",
        "Title_Code",
    ]

    # Analyze feature correlation with target
    for x in train_df_x:
        if train_df[x].dtype != "float64":
            print(train_df[[x, target[0]]].groupby(x).mean())

    # Graph individual features by survival
    fig, axis = plt.subplots(1, 3, figsize=(9, 6))
    sns.histplot(x="Fare", data=train_df, hue="Survived", multiple="stack", ax=axis[0])
    sns.histplot(x="Age", data=train_df, hue="Survived", multiple="stack", ax=axis[1])
    sns.histplot(
        x="FamilySize", data=train_df, hue="Survived", multiple="stack", ax=axis[2]
    )

    fig, axis = plt.subplots(2, 3, figsize=(16, 12))
    sns.barplot(x="Pclass", y="Survived", data=train_df, ax=axis[0, 0])
    sns.barplot(x="Sex", y="Survived", data=train_df, ax=axis[0, 1])
    sns.barplot(x="Embarked", y="Survived", data=train_df, ax=axis[0, 2])
    sns.barplot(x="IsAlone", y="Survived", data=train_df, ax=axis[1, 0])
    sns.barplot(x="Title", y="Survived", data=train_df, ax=axis[1, 1])

    # Compare class with a 2nd feature
    fig, axis = plt.subplots(1, 3, figsize=(9, 6))
    sns.barplot(x="Pclass", y="Survived", data=train_df, hue="Sex", ax=axis[0])
    sns.barplot(x="Pclass", y="Survived", data=train_df, hue="IsAlone", ax=axis[1])
    sns.barplot(x="Pclass", y="Survived", data=train_df, hue="Embarked", ax=axis[2])

    # Compare Sex with a 2nd feature
    fig, axis = plt.subplots(1, 3, figsize=(9, 6))
    sns.barplot(x="Sex", y="Survived", data=train_df, hue="Pclass", ax=axis[0])
    sns.barplot(x="Sex", y="Survived", data=train_df, hue="IsAlone", ax=axis[1])
    sns.barplot(x="Sex", y="Survived", data=train_df, hue="Embarked", ax=axis[2])

    # Correlation heatmap of dataset
    fig, ax = plt.subplots(figsize=(14, 12))
    fig = sns.heatmap(
        train_df.corr(),
        cmap=sns.diverging_palette(240, 10, as_cmap=True),
        annot=True,
        ax=ax,
    )

    # Machine Learning Algorithm (MLA) selection and initialization
    mla = [
        linear_model.LogisticRegressionCV(),
        linear_model.SGDClassifier(),
        linear_model.Perceptron(),
        linear_model.PassiveAggressiveClassifier(),
        linear_model.RidgeClassifierCV(),
        svm.SVC(probability=True),
        svm.NuSVC(probability=True),
        svm.LinearSVC(dual=False),
        neighbors.KNeighborsClassifier(),
        gaussian_process.GaussianProcessClassifier(),
        naive_bayes.GaussianNB(),
        naive_bayes.BernoulliNB(),
        tree.DecisionTreeClassifier(),
        tree.ExtraTreeClassifier(),
        ensemble.BaggingClassifier(),
        ensemble.RandomForestClassifier(),
        ensemble.ExtraTreesClassifier(),
        ensemble.AdaBoostClassifier(),
        ensemble.GradientBoostingClassifier(),
    ]

    mla_compare = test_models(mla, train_df, train_df_x_bin, target)

    best_estimator = optimize_params(mla, mla_compare, train_df, train_df_x_bin, target)

    generate_submission_csv(test_df, train_df_x_bin, best_estimator)


if __name__ == "__main__":
    main()
