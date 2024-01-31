import pandas as pd
import matplotlib.pyplot as plt
import scipy
import numpy as np
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
df = pd.read_excel(io="Dryad_CCMrisk_upload.xlsx")

df = df.dropna(subset = ['AgeatDiagnosis', 'Surgery', 'Sex', 'Lokalization_CCM',
                          'Loc_Brainstem', 'AgeCoxRe', 'Art_Hypertension', 'Diabetes',
        'Hyperlipidemia', 'Smoking', 'Obesity_BMI',
         'MOP_ICH',])


f, ax = plt.subplots(figsize=(10, 8))
corr = df.corr()
sns.heatmap(corr,
    cmap=sns.diverging_palette(220, 10, as_cmap=True),
    vmin=-1.0, vmax=1.0,
    square=True, ax=ax)
plt.show()

from dtreeviz.trees import *
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics


df['Event'] = np.where(df['Time_Event'].isna(), 0, 1)
bad_events = []
bad_noes = []
f1s , rec, acc, prec = 0,0,0,0




################## DECISION TREE #################



for j in range(1):

    train, test = train_test_split( df , test_size=0.3 , stratify= df["Event"])

    x_train = train.loc[:, ~df.columns.isin(['No','Event', "Time_Event", "Color","Crcen>5a","ICH_Event_CR", "AgeCoxRe"])]
    x_test  =  test.loc[:, ~df.columns.isin(['No','Event', "Time_Event", "Color","Crcen>5a","ICH_Event_CR",  "AgeCoxRe"])]

    clf = sklearn.tree.DecisionTreeClassifier(class_weight = "balanced", min_samples_split = 15, max_depth= 10)
    model = clf.fit(x_train, train["Event"])
    importances = clf.feature_importances_

    y_pred = model.predict(x_test)

    good_event,bad_event, good_no, bad_no = 0,0,0,0
    event,no  = 0,0
    for  i in range(len(y_pred)):
        if  list(test["Event"])[i] == 0 :
            no+=1
            if y_pred[i]  == 0 :
                good_no +=1
            else:
                bad_no +=1
        elif list(test["Event"])[i] == 1:
                    event+=1
                    if y_pred[i]  == 1 :
                        good_event += 1
                    else:
                        bad_event += 1

    bad_events.append((bad_event/event)*100)
    bad_noes.append((bad_no/no)*100)

    f1s += metrics.f1_score(test["Event"], model.predict(x_test))
    prec += metrics.precision_score(test["Event"], model.predict(x_test))
    acc += metrics.accuracy_score(test["Event"], model.predict(x_test))
    rec += metrics.recall_score(test["Event"], model.predict(x_test))

print(f1s / 1000, prec / 1000, acc / 1000, rec / 1000)
plt.hist(bad_events)
plt.show()
plt.hist(bad_noes)
plt.show()

for  i in range(len(x_test.columns)):
    print(x_test.columns[i] ,importances[i] )

#################################################################################################

df['Color'] = np.where(df['Event'] ==1 , "firebrick", "lightgrey")
plt.hist(df["AgeatDiagnosis"])
plt.show()

plt.plot(df["AgeatDiagnosis"], df["Time_Event"], "o")  # really no relationship between time of event and age
plt.show()
without_na = df.dropna()
print(scipy.stats.pearsonr(without_na["AgeatDiagnosis"], without_na["Time_Event"])) # statistic=-0.0072557579401945245, pvalue=0.9636257541853248

sns.scatterplot( df["AgeatDiagnosis"], color =  df["Color"], marker = "o")
plt.show()
print(sum(df["Event"]))


train, test = train_test_split( df , test_size=0.4 , stratify= df["Event"])
print(sum(train["Event"]), sum(test["Event"])) # 50  21
print(len(train["Event"]), len(test["Event"]))  #477 205

x_train = train.loc[:, ~df.columns.isin(['No','Event', "Time_Event", "Color","Crcen>5a","ICH_Event_CR", "AgeCoxRe"])]
x_test  =  test.loc[:, ~df.columns.isin(['No','Event', "Time_Event", "Color","Crcen>5a","ICH_Event_CR",  "AgeCoxRe"])]


#############################LOGISTIC REGRESSION########################
f1s , rec, acc, prec = 0,0,0,0

for j in range(1):
    train, test = train_test_split(df, test_size=0.3, stratify=df["Event"])

    x_train = train.loc[:, ~df.columns.isin(['No', 'Event', "Time_Event", "Color", "Crcen>5a", "ICH_Event_CR", "AgeCoxRe"])]
    x_test = test.loc[:, ~df.columns.isin(['No', 'Event', "Time_Event", "Color", "Crcen>5a", "ICH_Event_CR", "AgeCoxRe"])]

    model = LogisticRegression(solver='liblinear', C=3 ,class_weight = "balanced")
    model.fit(x_train, train["Event"])
    train_predictions = model.predict(x_test)
    f1s += metrics.f1_score(test["Event"], model.predict(x_test))
    prec += metrics.precision_score(test["Event"], model.predict(x_test))
    acc += metrics.accuracy_score(test["Event"], model.predict(x_test))
    rec += metrics.recall_score(test["Event"], model.predict(x_test))

print(f1s/1000, prec/1000,acc/1000,rec/1000)
cm = confusion_matrix(test["Event"], model.predict(x_test))
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.show()

print(classification_report(test["Event"], model.predict(x_test)))

# import statsmodels.api as sm
# logit_model=sm.Logit(train["Event"],x_train )
# result=logit_model.fit(method='newton')
# print(result.summary())
# predicted = result.predict(train["Event"])
# # we dont need all the params
# exit()
#################################### PCA ##########################


df_z_scaled = train.loc[:, ~train.columns.isin(['No', "Time_Event", "Color","Crcen>5a","ICH_Event_CR", "Surgery", "AgeCoxRe"])]
df_z = df_z_scaled.copy()
#apply normalization techniques
for column in df_z_scaled.columns:
    if column == "AgeatDiagnosis" or  column == "Lokalization CCM":
          df_z_scaled[column] = (df_z_scaled[column] -
                             df_z_scaled[column].mean()) / df_z_scaled[column].std()
test_scaled = test.loc[:, ~train.columns.isin(['No', "Time_Event", "Color","Crcen>5a","ICH_Event_CR", "Surgery", "AgeCoxRe"])]

for column in test_scaled.columns:
    if column == "AgeatDiagnosis" or  column == "Lokalization CCM":
          test_scaled[column] = (test_scaled[column] -
                             test_scaled[column].mean()) /test_scaled[column].std()


pca = PCA(n_components =7)

X_pca = pca.fit_transform(df_z_scaled.loc[:, ~df_z_scaled.columns.isin(["Event"])])

from sklearn.cluster import KMeans


inertias = []

LABEL_COLOR_MAP = {0 : 'orange',
                   1 : 'blue',
                   2 :   'purple'
                   }

#################################### Kmeans#####################################
for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(X_pca)
    inertias.append(kmeans.inertia_)

plt.plot(range(1,11), inertias, marker='o')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

kmeans = KMeans(n_clusters=2)
kmeans.fit(X_pca)
label_color = [LABEL_COLOR_MAP[l] for l in kmeans.labels_]

sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], c =df_z_scaled["Event"], alpha=0.8,style = df_z_scaled["Event"] )
plt.show()
predictions  = kmeans.predict(pca.fit_transform(test_scaled.loc[:, ~test_scaled.columns.isin(["Event"])]))

# print(list(predictions))
# print(list(test_scaled["Event"]))
# good_event,bad_event,good_no, bad_no=0,0,0,0
#
#
# for  i in range(len(predictions)):
#     if  list(test_scaled["Event"])[i] == 0 :
#         if predictions[i]  == 0 :
#             good_event +=1
#         else:
#             bad_event +=1
#     elif list(test_scaled["Event"])[i] == 1:
#                 if predictions[i]  == 1 or predictions[i]  == 2:
#                     good_no += 1
#                 else:
#                     bad_no += 1
#
# print(good_event, "bad ", bad_event )
# print(good_no, "bad ", bad_no )


import statsmodels.formula.api as smf
import statsmodels.api as sm

print("################################## GLM ################################## ")
################################## GLM ##################################
means = []
for j in range(1):
    train, test = train_test_split(df, test_size=0.3, stratify=df["Event"])

    x_train = train.loc[:, ~df.columns.isin(['No', 'Event', "Time_Event", "Color", "Crcen>5a", "ICH_Event_CR", "AgeCoxRe"])]
    x_test = test.loc[:, ~df.columns.isin(['No', 'Event', "Time_Event", "Color", "Crcen>5a", "ICH_Event_CR", "AgeCoxRe"])]

    train = train.dropna(subset = ["Time_Event"])
    test = test.dropna(subset = ["Time_Event"])

    x_train = train.loc[:, ~df.columns.isin(['No', 'Event',  "Color", "Crcen>5a", "ICH_Event_CR", "AgeCoxRe"])]
    # model = smf.glm(formula = '''Time_Event ~AgeatDiagnosis+Surgery+Sex+Lokalization_CCM+
    #                           Loc_Brainstem+AgeCoxRe+Art_Hypertension+Diabetes+Hyperlipidemia+Smoking+Obesity_BMI+
    #          MOP_ICH+CRcen1a+CRcen2a+CRcen3a+CRcen4a+CRcen5a
    #
    # ''',
    #                 data = train,
    #                 family = sm.families.Poisson())
    # result = model.fit()
    # Display and interpret results

    model = smf.glm(formula='''Time_Event ~ Surgery+Lokalization_CCM+
                              Loc_Brainstem+Art_Hypertension+Diabetes+Hyperlipidemia+Smoking+Obesity_BMI+
             MOP_ICH-1
    
    ''',
                    data=train,
                    family=sm.families.Gamma())

    # Fit the model

    # Fit the model
    result = model.fit(method="lbfgs", class_weight = "balanced" , C=10)
    # Display and interpret results
    #print(result.summary())
    # Estimated default probabilities
    predictions = result.predict(test)

    import statistics

    means.append(statistics.mean(abs(list(predictions) - test["Time_Event"])))

print(statistics.mean(means))
plt.hist(means)
plt.show()
print(result.summary())