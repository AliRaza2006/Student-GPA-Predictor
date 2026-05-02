from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import KFold
import random

path="Academic.csv"
df=pd.read_csv(path)
copy=df.copy()

copy.info()
print()
print("Null data before filling null: ")
print(copy.isna().sum())
print()
print(f"Duplicated data before removing the duplicates: {copy.duplicated().sum()}")
print()
objcols=copy.select_dtypes(include=['object','string']).columns
numcols=copy.select_dtypes(include='number').columns
copy[numcols]=copy[numcols].apply(pd.to_numeric,errors='coerce')
copy[numcols]=copy[numcols].fillna(copy[numcols].mean())
copy[objcols]=copy[objcols].fillna(copy[objcols].mode().iloc[0])
print("Null Data after filling null: ")
print(copy.isna().sum())
print()
copy=copy.drop_duplicates(keep='first')
print(f"Duplicated data after before removing the duplicates: {copy.duplicated().sum()}")
print()

num_data = copy.select_dtypes(include='number')
plt.figure(figsize=(10,6))
sns.heatmap(num_data.corr(), annot=True, cmap="coolwarm")
plt.show()

print()

n = len(numcols)
cols = 3
rows = (n // cols) + (n % cols > 0)
fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
axes = axes.flatten()
for i, col in enumerate(numcols):
    sns.boxplot(x=copy[col], ax=axes[i])
    axes[i].set_title(col)
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)
plt.tight_layout(pad=5)
plt.show()

le=LabelEncoder()
copy['Part-Time Job']=le.fit_transform(copy['Part-Time Job'])
pd.options.display.max_columns=None

copy=pd.get_dummies(copy,columns=['Major'],drop_first=True)

bool_cols = copy.select_dtypes(include='bool').columns #i did this because the columns got converted to bool data type by one hot encoding so for linear regression i had to convert them to int data type
copy[bool_cols] = copy[bool_cols].astype(int)
print(copy.head())

scaler=StandardScaler()
Y = copy.pop('College GPA')
X = copy
scaler.fit(X)
X=scaler.transform(X)
X = pd.DataFrame(X,columns=copy.columns)
kf=KFold(n_splits=5,shuffle=True,random_state=random.randint(1,100))
model=LinearRegression()
r2_scores=[]
mae=[]
rmse=[]
for trainindex,testindex in kf.split(X):
    xtrain,xtest=X.iloc[trainindex],X.iloc[testindex]
    ytrain,ytest=Y.iloc[trainindex],Y.iloc[testindex]
    model.fit(xtrain,ytrain)
    ypred=model.predict(xtest)
    r2_scores.append(r2_score(ytest,ypred))
    mae.append(mean_absolute_error(ytest,ypred))
    rmse.append(root_mean_squared_error(ytest,ypred))

for i in range(0,len(r2_scores)):
    print(f"R2 Score in iteration {i+1}: {r2_scores[i]}")
    print(f"Mean absolute error in iteration {i+1}: {mae[i]}")
    print(f"Root mean squared error in iteration {i+1}: {rmse[i]}")
    print()

print(f"R2 mean: {np.mean(r2_scores)}")
print(f"Mean absolute error: {np.mean(mae)}")
print(f"Root mean squared error: {np.mean(rmse)}")

# print(model.coef_)
print()

print("Predict your Gpa by entering your information: ")
print()
study_week=float(input("How many hours do you study in a week:  "))
attendance=float(input("What is your attendance percentage: "))
major=input("Whats your major(Business/Science/Arts/Engineering): ")
high_gpa=float(input("what was your gpa in high school: "))
extra=int(input("Number of extracurricular activities: "))
part_time=int(input("Do you have any part time job(1:yes, 0:No): "))
library=float(input("How many hours do you spend in library in a week: "))
online_course_work=float(input("Online coursework hours: "))
sleep=float(input("How many hours do you sleep: "))

newXtest=pd.DataFrame([{
    "Study Hours per Week":study_week,
    "Attendance Rate":attendance,
    "Major_Business":0,
    "Major_Engineering":0,
    "Major_Science":0,
    "High School GPA":high_gpa,
    "Extracurricular Activities":extra,
    "Part-Time Job":part_time,
    "Library Usage per Week":library,
    "Online Coursework Engagement":online_course_work,
    "Sleep Hours per Night":sleep,
}])
if major.lower() == "business":
    newXtest["Major_Business"] = 1
elif major.lower() == "engineering":
    newXtest["Major_Engineering"] = 1
elif major.lower() == "science":
    newXtest["Major_Science"] = 1
newXtest = newXtest.reindex(columns=X.columns, fill_value=0)

newXtest = pd.DataFrame(
    scaler.transform(newXtest),
    columns=X.columns
)
print(f"\nCollege Gpa: {model.predict(newXtest)[0]}")