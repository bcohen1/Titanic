import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection

from sklearn import linear_model, svm, neighbors, gaussian_process, naive_bayes, tree, ensemble

dir = os.getcwd()
os.chdir(dir)

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

combine = [train_df, test_df]

train_df_raw = train_df.copy()
test_df_raw = test_df.copy()

'''Analyze'''
train_df.info()
train_df.isnull().sum()
test_df.isnull().sum()
train_df.describe()

'''Clean'''
for df in combine:
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)

    df['FamilySize'] = df['SibSp'] + train_df['Parch'] + 1
    df['IsAlone'] = np.where(df['FamilySize']==1, 1, 0)

    df['Title'] = df['Name'].str.extract('([A-Za-z]+)\.', expand=False)
    df['Title'].replace(['Sir'], 'Mr', inplace=True)
    df['Title'].replace(['Mlle', 'Ms'], 'Miss', inplace=True)
    df['Title'].replace(['Mme'], 'Mrs', inplace=True)
    df['Title'].replace(['Don', 'Jonkheer', 'Countess', 'Lady'], 'Master', inplace=True)
    df['Title'].replace(['Sir'], 'Mr', inplace=True)
    df['Title'].replace(['Dr', 'Rev', 'Col', 'Major', 'Capt'], 'Misc', inplace=True)

train_df['Title'].value_counts()
train_df.drop(['PassengerId', 'Ticket', 'Cabin'], axis=1, inplace=True)

# plt.hist(train_df['Fare'], 10)
# plt.hist(train_df['Age'], 8)

'''consider qcut instead for fare'''
for df in combine:
    df['FareBin'] = pd.cut(df['Fare'], 10)
    df['AgeBin'] = pd.cut(df['Age'], 8)

'''Encode categorical fields'''
label = LabelEncoder()

for df in combine:
    df['Sex_Code'] = label.fit_transform(df['Sex'])
    df['Embarked_Code'] = label.fit_transform(df['Embarked'])
    df['Title_Code'] = label.fit_transform(df['Title'])
    df['AgeBin_Code'] = label.fit_transform(df['AgeBin'])
    df['FareBin_Code'] = label.fit_transform(df['FareBin'])

'''define y variable'''    
Target = ['Survived']

'''define x variables for feature selection'''
train_df_x = ['Sex', 'Pclass', 'Embarked', 'Title', 'SibSp', 'Parch',
              'Age', 'Fare', 'FamilySize', 'IsAlone']
# train_df_x_calc = ['Sex_Code', 'Pclass', 'Embarked_Code', 'Title_Code',
#                  'SibSp', 'Parch', 'Age', 'Fare']
train_df_xy =  Target + train_df_x

'''define x variables for original w/bin features to remove continuous variables'''
train_df_x_bin = ['Sex_Code', 'Pclass', 'Embarked_Code', 'Title_Code',
                  'FamilySize', 'IsAlone', 'AgeBin_Code', 'FareBin_Code']
train_df_xy_bin = Target + train_df_x_bin

'''Analyze feature x correlation with output y'''
for x in train_df_x:
    if train_df[x].dtype != 'float64':
        print(train_df[[x, Target[0]]].groupby(x).mean())

plt.figure(figsize=[16,12])

plt.subplot(231)        
plt.boxplot(x=train_df['Fare'], showmeans = True, meanline = True)
plt.title('Fare Boxplot')

plt.subplot(232)
plt.boxplot(train_df['Age'], showmeans = True, meanline = True)
plt.title('Age Boxplot')

plt.subplot(233)
plt.boxplot(train_df['FamilySize'], showmeans = True, meanline = True)
plt.title('Family Size Boxplot')

plt.subplot(234)
plt.hist(x = [train_df[train_df['Survived']==1]['Fare'], train_df[train_df['Survived']==0]['Fare']], 
         stacked=True, color = ['g','r'], label = ['Survived','Died'])
plt.title('Fare Histogram by Survival')
plt.legend()

plt.subplot(235)
plt.hist(x = [train_df[train_df['Survived']==1]['Age'], train_df[train_df['Survived']==0]['Age']], 
         stacked=True, color = ['g','r'],label = ['Survived','Died'])
plt.title('Age Histogram by Survival')
plt.legend()

plt.subplot(236)
plt.hist(x = [train_df[train_df['Survived']==1]['FamilySize'], train_df[train_df['Survived']==0]['FamilySize']], 
         stacked=True, color = ['g','r'],label = ['Survived','Died'])
plt.title('Family Size Histogram by Survival')
plt.legend()

'''graph individual features by survival'''
fig, saxis = plt.subplots(2, 3, figsize=(16,12))
sns.barplot(x = 'Embarked', y = 'Survived', data=train_df[['Embarked','Survived']], ax = saxis[0,0])
sns.barplot(x = 'Pclass', y = 'Survived', order=[1,2,3], data=train_df, ax = saxis[0,1])
sns.barplot(x = 'IsAlone', y = 'Survived', order=[1,0], data=train_df, ax = saxis[0,2])
sns.pointplot(x = 'FareBin', y = 'Survived',  data=train_df, ax = saxis[1,0])
sns.pointplot(x = 'AgeBin', y = 'Survived',  data=train_df, ax = saxis[1,1])
sns.pointplot(x = 'FamilySize', y = 'Survived', data=train_df, ax = saxis[1,2])

'''Compare class and a 2nd feature'''
fig, (axis1,axis2,axis3) = plt.subplots(1, 3, figsize=(14,12))
sns.boxplot(x = 'Pclass', y = 'Fare', hue = 'Survived', data = train_df, ax = axis1)
axis1.set_title('Pclass vs Fare Survival Comparison')
sns.violinplot(x = 'Pclass', y = 'Age', hue = 'Survived', data = train_df, split = True, ax = axis2)
axis2.set_title('Pclass vs Age Survival Comparison')
sns.boxplot(x = 'Pclass', y ='FamilySize', hue = 'Survived', data = train_df, ax = axis3)
axis3.set_title('Pclass vs Family Size Survival Comparison')

'''Compare Sex and a 2nd feature'''
fig, qaxis = plt.subplots(1,3,figsize=(14,12))
sns.barplot(x = 'Sex', y = 'Survived', hue = 'Embarked', data=train_df, ax = qaxis[0])
axis1.set_title('Sex vs Embarked Survival Comparison')
sns.barplot(x = 'Sex', y = 'Survived', hue = 'Pclass', data=train_df, ax  = qaxis[1])
axis1.set_title('Sex vs Pclass Survival Comparison')
sns.barplot(x = 'Sex', y = 'Survived', hue = 'IsAlone', data=train_df, ax  = qaxis[2])
axis1.set_title('Sex vs IsAlone Survival Comparison')

'''Compare Family Size and Class with Sex'''
fig, (maxis1, maxis2) = plt.subplots(1, 2,figsize=(14,12))
#how does family size factor with sex & survival compare
sns.pointplot(x="FamilySize", y="Survived", hue="Sex", data=train_df,
              palette={"male": "blue", "female": "pink"},
              markers=["*", "o"], linestyles=["-", "--"], ax = maxis1)
#how does class factor with sex & survival compare
sns.pointplot(x="Pclass", y="Survived", hue="Sex", data=train_df,
              palette={"male": "blue", "female": "pink"},
              markers=["*", "o"], linestyles=["-", "--"], ax = maxis2)

'''how does embark port factor with class, sex, and survival compare'''
e = sns.FacetGrid(train_df, col = 'Embarked')
e.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', ci=95.0, palette = 'deep')
e.add_legend()

'''histogram comparison of sex, class, and age by survival'''
h = sns.FacetGrid(train_df, row = 'Sex', col = 'Pclass', hue = 'Survived')
h.map(plt.hist, 'Age', alpha = .75)
h.add_legend()

'''correlation heatmap of dataset'''
fig, ax = plt.subplots(figsize =(14, 12))
colormap = sns.diverging_palette(220, 10, as_cmap = True)
fig = sns.heatmap(
    train_df.corr(), 
    cmap = colormap,
    square=True, 
    cbar_kws={'shrink':.9 }, 
    ax=ax,
    annot=True, 
    linewidths=0.1,vmax=1.0, linecolor='white',
    annot_kws={'fontsize':12 }
)
plt.title('Pearson Correlation of Features', y=1.05, size=15)

'''Machine Learning Algorithm (MLA) Selection and Initialization'''

MLA = [
    #GLM
    linear_model.LogisticRegressionCV(),
    # linear_model.SGDClassifier(),
    # linear_model.Perceptron(),    
    # linear_model.PassiveAggressiveClassifier(),
    # linear_model.RidgeClassifierCV(),
  
    # #SVM
    # svm.SVC(probability=True),
    # svm.NuSVC(probability=True),
    # svm.LinearSVC(),   
    
    # #Nearest Neighbor
    # neighbors.KNeighborsClassifier(),
    
    # #Gaussian Processes
    # gaussian_process.GaussianProcessClassifier(),    

    # #Navies Bayes
    # naive_bayes.GaussianNB(),    
    # naive_bayes.BernoulliNB(),
       
    # #Trees    
    # tree.DecisionTreeClassifier(),
    # tree.ExtraTreeClassifier(),    
    
    # #Ensemble Methods
    # ensemble.BaggingClassifier(),
    # ensemble.RandomForestClassifier(),   
    # ensemble.ExtraTreesClassifier(),    
    # ensemble.AdaBoostClassifier(),
    # ensemble.GradientBoostingClassifier(),
    ]

'''split dataset in cross-validation, note: this is an alternative to train_test_split'''
cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0 )
'''run model 10x with 60/30 split intentionally leaving out 10%'''

'''create table to compare MLA metrics'''
MLA_columns = ['MLA Name', 'MLA Parameters', 'MLA Train Accuracy Mean',
               'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD' , 'MLA Time']
MLA_compare = pd.DataFrame(columns = MLA_columns)

'''create table to compare MLA predictions'''
MLA_predict = train_df[Target]

'''index through MLA and save performance to table'''
row_index = 0
for alg in MLA:

    #set name and parameters
    MLA_name = alg.__class__.__name__
    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
    MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())
    
    '''score model with cross validation'''
    cv_results = model_selection.cross_validate(alg, train_df[train_df_x_bin], train_df[Target], cv = cv_split, return_train_score=True)
    
    MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
    MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()
    MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()   
    MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results['test_score'].std()*3

    '''save MLA predictions'''
    alg.fit(train_df[train_df_x_bin], train_df[Target].values.ravel())
    MLA_predict[MLA_name] = alg.predict(train_df[train_df_x_bin])
    
    row_index+=1

'''print and sort table'''
MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'], ascending = False, inplace = True)
MLA_compare

# '''define x and y variables for dummy features original'''
# #Move down further
# train_df_dummy = pd.get_dummies(train_df[train_df_x])
# train_df_x_dummy = train_df_dummy.columns.tolist()
# train_df_xy_dummy = Target + train_df_x_dummy

