
# Machine Learning Practice using Titanic example
## Orders
### 1. 데이터를 불러온다(pandas의 read_csv로)
### 2. 데이터 info를 보고 null 값에 대해 전처리
### 3. 데이터 문자열들을 categorize
### 4. data exploration
### 5. 남은 데이터 문자열 숫자형을 categorize 
### 6. 1-5과정에 포함되는 preprocessing 함수들을 만들기


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline 
#주피터 노트북에서 show안에도 그림 볼 수 있게 해줌
```

### 1. 데이터를 불러온다(pandas의 read_csv로)


```python
titanic_df = pd.read_csv('./titanic_train.csv')
titanic_df.head()
print(titanic_df.info())
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 12 columns):
    PassengerId    891 non-null int64
    Survived       891 non-null int64
    Pclass         891 non-null int64
    Name           891 non-null object
    Sex            891 non-null object
    Age            714 non-null float64
    SibSp          891 non-null int64
    Parch          891 non-null int64
    Ticket         891 non-null object
    Fare           891 non-null float64
    Cabin          204 non-null object
    Embarked       889 non-null object
    dtypes: float64(2), int64(5), object(5)
    memory usage: 83.6+ KB
    None
    

### 2. 데이터 info를 보고 null 값에 대해 전처리


```python
titanic_df['Age'].fillna(titanic_df['Age'].mean(), inplace = True)
titanic_df['Cabin'].fillna('N', inplace=True)
titanic_df['Embarked'].fillna('N', inplace=True)
titanic_df.isnull().sum().sum() #행별로 확인, 다시 행을 다 합쳐서 확인
```




    0



### 3. 데이터 문자열들을 classification


```python
# 문자열 확인(분류가능)
titanic_df['Sex'].value_counts()
titanic_df['Embarked'].value_counts()
titanic_df['Cabin'].value_counts()
 # cabin 같은 경우 앞자리가 선실의 레벨을 나타내기에 첫 글자만 따온다
titanic_df['Cabin'] = titanic_df['Cabin'].str[:1]
print(titanic_df['Cabin'].head(3))
```

    0    N
    1    C
    2    N
    Name: Cabin, dtype: object
    

### 4. data exploration


```python
titanic_df.groupby(['Sex', 'Survived'])['Survived'].count()
# 남자가 더 많이 죽음. 시각화를 해보자(seaborn을 사용해서)
# seaborn 같은 경우 세련된 비주얼, 쉬운 API, 편리한 판다스 연동
sns.barplot(x='Sex', y='Survived', data=titanic_df)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2914c1ec240>




![png](02_titanic_prediction_files/02_titanic_prediction_9_1.png)



```python
# Pclass와 성별을 살펴보면서 부에 따른 생존을 살펴보자
sns.barplot(x='Pclass', y='Survived', hue='Sex', data=titanic_df)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2914c29aba8>




![png](02_titanic_prediction_files/02_titanic_prediction_10_1.png)



```python
# 입력 age에 따라 구분 값을 반환하는 함수 설정. DataFrame의 apply lambda 식에 사용
def get_category(age):
    cat = ''
    if age <= -1: cat = 'Unknown'
    elif age <= 5: cat = 'Baby'
    elif age <= 12: cat = 'Child'
    elif age <= 18: cat = 'Teenager'
    elif age <= 25: cat = 'Student'
    elif age <= 35: cat = 'Young Adult'
    elif age <= 60: cat = 'Adult'
    else : cat = 'Elderly'

    return cat

# 막대 그래프의 크기 figure를 더 크게 설정
plt.figure(figsize=(10, 6))

# X축의 값을 순차적으로 표시하기 위한 설정
group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Elderly']

# lambda 식에 위에서 생성한 get_category() 함수를 반환값으로 지정
# get_category(X)는 입력값으로 'Age' 칼럼 값을 받아서 해당하는 cat 반환
titanic_df['Age_cat'] = titanic_df['Age'].apply(lambda x : get_category(x))
sns.barplot(x='Age_cat', y='Survived', hue='Sex', data=titanic_df, order=group_names)
titanic_df.drop('Age_cat', axis=1, inplace=True)
```


![png](02_titanic_prediction_files/02_titanic_prediction_11_0.png)


### 5. 남은 데이터 문자열 숫자형을 categorize
labelEncoder를 사용. encode_feature()함수를 새로 생성


```python
from sklearn import preprocessing

def encode_feature(dataDF):
    features = ['Cabin', 'Sex', 'Embarked']
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(dataDF[feature])
        dataDF[feature] = le.transform(dataDF[feature])

    return dataDF
titanic_df = encode_feature(titanic_df)
titanic_df.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>1</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>7</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>0</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>0</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>7</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



### 6. 1-5과정에 포함되는 preprocessing 함수들을 만들기


```python
# null 처리 함수
def fillna(df):
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    df['Cabin'].fillna('N', inplace=True)
    df['Embarked'].fillna('N', inplace=True)
    df['Fare'].fillna(0, inplace=True)
    return df

# 머신러닝 알고리즘에 불필요한 속성 제거
def drop_feature(df):
    df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
    return df

# 레이블 인코딩 수행
def format_features(df):
    df['Cabin'] = df['Cabin'].str[:1]
    features = ['Cabin', 'Sex', 'Embarked']
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df[feature])
        df[feature] = le.transform(df[feature])
    return df

# 앞에서 설정한 데이터 전처리 함수 호출
def transform_features(df):
    df = fillna(df)
    df = drop_feature(df)
    df = format_features(df)
    return df

#이제 y에다가는 survived를 넣고, X에는 전처리 된 변수들을 넣는다
titanic_df = pd.read_csv('./titanic_train.csv')
y_titanic_df = titanic_df['Survived']
X_titanic_df = titanic_df.drop('Survived', axis=1)

X_titanic_df = transform_features(X_titanic_df)
```
