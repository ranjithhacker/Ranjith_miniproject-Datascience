# Ranjith_miniproject-Datascience

# IMPORT THE LIBRARIES
import warnings
warnings.filterwarnings('ignore')


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


from google.colab import files
uploaded=files.upload()

**READING THE DATASET AND CHECKING COUNT OF NULL VALUES:**

# READ THE DATASET
df = pd.read_csv('full_data.csv')
print(f'The dataset:\n{df.head()}')

df.describe()

# ANLAYSING THE DATASET
print(f'Random rowss:\n{df.sample(6)}')
print(f'\nThe brain stroke dataset includes the following\n: {df.columns}')
print(f'\nTotal Rows and Columns = {df.shape}')
print(f'\nStatistics: \n{df.describe()}')
print(f'\Info: \n{df.info()}')
print(f'\nChecking for NULL values: \n{df.isnull().sum()}')
print(f'\nData proportions: {df["gender"].unique()}')
print(f'{df["gender"].value_counts()}')
print ("\nMissing values: ", df.isnull().sum().values.sum())

# ANALYSING WITH VISUALISATION
# declaring data
stroke = df.loc[df['stroke']==1]
# PIE-PLOT
data = stroke['gender'].value_counts()
theLabels = ['Male', 'Female']
explode = [0, 0.1]
myColors = ['lightskyblue','lightpink']



# plotting data on chart
plt.pie(data, labels=theLabels,explode=explode,autopct='%.1f%%', colors=myColors, shadow=True)
plt.title('Gender Pie Chart (with stroke)')
plt.legend(loc='upper left')
plt.show()

# HISTOGRAM
sns.histplot(data=df,x='age',hue='stroke',palette='crest')
plt.title('Age distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# SUBPLOTS
fig, axes = plt.subplots(2, 2, figsize=(12, 10), edgecolor = "black")
df.plot(kind="hist", y="age", bins=70, color="b", ax=axes[0][0], edgecolor = "black")
df.plot(kind="hist", y="bmi", bins=100, color="r", ax=axes[0][1], edgecolor = "black")
df.plot(kind='scatter', x='age', y='avg_glucose_level', color='green', ax=axes[1][0], title="Age vs. avg_glucose_level")
df.plot(kind='scatter', x='bmi', y='avg_glucose_level', color='red', ax=axes[1][1], title="bmi vs. avg_glucose_level")
plt.show()

# HEATMAP
plt.figure(figsize=(10,10))
plt.title('Correlation Heatmap')
sns.heatmap(df.corr(),annot=True);
plt.yticks(rotation=0)
plt.xticks(rotation=0)
plt.show()


# SMOKING STATUS
print(f'\nSmoking status of people with Brain stroke:\n{df["smoking_status"].unique()}')
smoking_status=['formerly smoked','never smoked','smokes','Unknown']
formerly_smoked=len(stroke[stroke['smoking_status']==smoking_status[0]])
never_smoked=len(stroke[stroke['smoking_status']==smoking_status[1]])
smokes=len(stroke[stroke['smoking_status']==smoking_status[2]])
unknown=len(stroke[stroke['smoking_status']==smoking_status[3]])
print('Never smoked:',never_smoked)
print('Formerly smoked:',formerly_smoked)
print('Smoked:',smokes)
print('Unknown:',unknown)
smoke_stroke = [formerly_smoked, never_smoked, smokes, unknown]

# BARPLOT
sns.barplot(y=smoke_stroke, x=smoking_status, palette='Pastel1')
plt.title("Stroke / Smoking Status")
plt.show()

# LABEL ENCODING
from sklearn.preprocessing import LabelEncoder 
lb = LabelEncoder()
# similar to smoking_status in IRIS dataset
df['ever_married'] = lb.fit_transform(df['ever_married'])
df['work_type'] = lb.fit_transform(df['work_type'])
df['Residence_type'] = lb.fit_transform(df['Residence_type'])
df['smoking_status'] = lb.fit_transform(df['smoking_status'])
df['gender'] = lb.fit_transform(df['gender'])



# DIVIDE DATASET INTO INDEPENDENT AND DEPENDENT VARIABLE
x = df.drop(['stroke'], axis=1)
y = df["stroke"]
print(f'y = \n{y}')
# SPLITTING x AND y INTO TRAIN AND TEST DATASET
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .3)
print(f"\nx_train shape: {x_train.shape}")
print(f"x_test shape: {x_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")
# IMPORT THE MODEL/ALGORITHM - LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression 
lr = LogisticRegression(max_iter=1000)
# TRAIN THE MODEL
lr.fit(x_train,y_train) 
# EVALUATION
from sklearn.metrics import mean_absolute_error, r2_score , confusion_matrix, accuracy_score, classification_report
print("\nMODEL EVALUATION: ")
print("Mean_absolute_error = ", mean_absolute_error(lr_y_pred, y_test)) 
print("The classification report is:\n",classification_report(y_test,lr_y_pred))
print("r2_score = ", r2_score(lr_y_pred, y_test)) 
print("Accuracy of LOGISTIC REGRESSION MODEL in percentage (%): ", (accuracy_score(y_test, lr_y_pred))*100)



# DIVIDE DATASET INTO INDEPENDENT AND DEPENDENT VARIABLE
x = df.drop(['stroke'], axis=1)
y = df["stroke"]
print(f'y = \n{y}')
# SPLITTING x AND y INTO TRAIN AND TEST DATASET
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .3)
print(f"\nx_train shape: {x_train.shape}")
print(f"x_test shape: {x_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")
# IMPORT THE MODEL/ALGORITHM - LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression 
lr = LogisticRegression(max_iter=1000)
# TRAIN THE MODEL
lr.fit(x_train,y_train) 
# EVALUATION
from sklearn.metrics import mean_absolute_error, r2_score , confusion_matrix, accuracy_score, classification_report
print("\nMODEL EVALUATION: ")
print("Mean_absolute_error = ", mean_absolute_error(lr_y_pred, y_test)) 
print("The classification report is:\n",classification_report(y_test,lr_y_pred))
print("r2_score = ", r2_score(lr_y_pred, y_test)) 
print("Accuracy of LOGISTIC REGRESSION MODEL in percentage (%): ", (accuracy_score(y_test, lr_y_pred))*100)

# IMPORT THE MODEL/ALGORITHM - BERNOULLINB
from sklearn.naive_bayes import BernoulliNB 
bnb = BernoulliNB()

# TRAIN THE MODEL
bnb.fit(x_train,y_train) 

# PREDICTIONION
bnb_y_pred = bnb.predict(x_test) 

# EVALUATION
from sklearn.metrics import mean_absolute_error, r2_score , confusion_matrix, accuracy_score, classification_report
print("\nMODEL EVALUATION: ")
cm = confusion_matrix(y_test,bnb_y_pred)
print("Consfusion Matrix-\n", confusion_matrix(y_test, bnb_y_pred)) 
print("Mean_absolute_error = ", mean_absolute_error(bnb_y_pred, y_test)) 
print("The classification report is:\n",classification_report(y_test, bnb_y_pred))
print("Accuracy of NAIVE BAYES- BERNOULLI MODEL in percentage (%): ", (accuracy_score(y_test, bnb_y_pred))*100)
sns.heatmap(cm, annot=True)
plt.show()

# IMPORT THE MODEL/ALGORITHM - DECISION TREE
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()

# TRAIN THE MODEL
dtc.fit(x_train,y_train) 

# PREDICTIONION
dtc_y_pred = dtc.predict(x_test) 

# EVALUATION
from sklearn.metrics import mean_absolute_error, r2_score , confusion_matrix, accuracy_score, classification_report
print("\nMODEL EVALUATION: ")
cm = confusion_matrix(y_test,dtc_y_pred)
print("Consfusion Matrix-\n", confusion_matrix(y_test, dtc_y_pred)) 
print("Mean_absolute_error = ", mean_absolute_error(dtc_y_pred, y_test)) 
print("The classification report is:\n",classification_report(y_test,dtc_y_pred))
print("Accuracy of DECISION TREE CLASSIFIER in percentage (%): ", (accuracy_score(y_test, dtc_y_pred))*100)
sns.heatmap(cm, annot=True)
plt.show()


# IMPORT THE MODEL/ALGORITHM - GAUSSIAN NB
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

# TRAIN THE MODEL
gnb.fit(x_train,y_train) 

# # PREDICTION
gnb_y_pred = gnb.predict(x_test) 
# print(f"First 5 actual values -\n{y_test.values[:5]},\nFirst 5 predicted values -\n{gnb_y_pred[:5]}")

# EVALUATION
from sklearn.metrics import mean_absolute_error, r2_score , confusion_matrix, accuracy_score, classification_report
print("\nMODEL EVALUATION: ")
cm = confusion_matrix(y_test,gnb_y_pred)
print("Consfusion Matrix-\n", confusion_matrix(y_test, gnb_y_pred)) 
print("Mean_absolute_error = ", mean_absolute_error(gnb_y_pred, y_test)) 
print("The classification report is:\n",classification_report(y_test,gnb_y_pred))
print("Accuracy of NAIVE BAYES- GAUSSIAN MODEL in percentage (%): ", (accuracy_score(y_test, gnb_y_pred))*100)
sns.heatmap(cm, annot=True)
plt.show()

# IMPORT THE MODEL/ALGORITHM
from sklearn.svm import SVC
svm = SVC()

# TRAIN THE MODEL
svm.fit(x_train,y_train) 

# PREDICTION
svm_y_pred = svm.predict(x_test) 

# EVALUATION
from sklearn.metrics import mean_absolute_error, r2_score , confusion_matrix, accuracy_score, classification_report
print("\nMODEL EVALUATION: ")
cm = confusion_matrix(y_test,svm_y_pred)
print("Consfusion Matrix-\n", confusion_matrix(y_test, svm_y_pred)) 
print("Mean_absolute_error = ", mean_absolute_error(svm_y_pred, y_test)) 
print("The classification report is:\n",classification_report(y_test,svm_y_pred))
print("Accuracy of SUPPORT VECTOR in percentage (%): ", (accuracy_score(y_test, svm_y_pred))*100)
sns.heatmap(cm, annot=True)
plt.show()


# ---------------------PERFORMANCE ANALYSIS OF DIFFERENT MODELS -------------------
dataPerf = pd.DataFrame(data={'Model': ['LogisticRegression', 'BernoulliNB', 'Decision Tree Classifier', 'GaussianNB','K-Nearest Neighbours Classifier', 'Random Forest', 'SVM'], 'Score': [lr.score(x_test, y_test), bnb.score(x_test, y_test), dtc.score(x_test, y_test), gnb.score(x_test, y_test), knn.score(x_test, y_test), rfc.score(x_test, y_test), svm.score(x_test, y_test)]})

plt.figure(figsize=(12, 8))
sns.barplot(x="Model", y="Score", data=dataPerf, palette="magma")
plt.title('Performance analysis of different Models')
plt.show()


![image](https://github.com/Naveen3640/BRAIN_STROKE-PREDICTION/assets/95179990/8b4047d5-6490-4d2a-b6a5-36df0581a36e)
CONCLUSION: 

WITH THIS PROJECT WE CAN PREDICT BRAIN STROKE WITH DIFFERENT ALGORITHMS AND 
VERIFY WHICH ALGORITHM IS VERY EFFFICIENT IN PREDICTING THE BRAIN STROKE BY VISUALIZING ACCURACY OF EACH ALGORITHM.

                                                                  Accuracy Score: 95.38461%
