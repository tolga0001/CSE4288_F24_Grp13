import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import seaborn as sns
from scipy import stats

def visualize(df):
    #Graph age distribution
    plt.hist(df['age'], bins='auto', color='skyblue', edgecolor='black')
    plt.title('Age Distribution')
    plt.xlabel('age')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    #Graph gpa distribution
    plt.hist(df['gpa'], bins=5, color='salmon', edgecolor='black')
    plt.title('gpa Distribution')
    plt.xlabel('GPA')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    #Graph academic year distribution
    plt.hist(df['academic_year'], bins=4, color='green', edgecolor='black')
    plt.title('academic year Distribution')
    plt.xlabel('Academic Year')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    #Graph course distribution
    course_counts = df['course'].value_counts().sort_values(ascending=False)
    plt.bar(course_counts.index, course_counts.values, color='yellow')
    plt.xlabel('Course')
    plt.ylabel('Frequency')
    plt.title('Course Frequency')
    plt.xticks(course_counts.index)
    plt.show()

    print("\nSkewness of columns\n")
    print(f"\ngender: {df['gender'].skew()}")
    print(f"\nage: {df['age'].skew()}")
    print(f"\ncourse: {df['course'].skew()}")
    print(f"\nacademic year: {df['academic_year'].skew()}")
    print(f"\ngpa: {df['gpa'].skew()}")

    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.figure(figsize=(10, 10))
    plt.show()

    #check outliers(only necessary for age in this dataset due to its size)
    df['age_zscore'] = zscore(df['age'])
    plt.figure(figsize=(10, 10))
    sns.histplot(df['age_zscore'], kde=True, label='Z-score of age', color='green', stat='density')
    plt.title('Z-score Distribution for Age')
    plt.xlabel('Z-score')
    plt.ylabel('Density')
    plt.show()
    df.drop(columns=['age_zscore'], inplace=True)


def encode(df):

    #Encode answer columns
    yes_no_map = {'Yes': 1, 'No': 0, 'yes': 1, 'no': 0}
    df['marital_status'] = df['marital_status'].map(yes_no_map)
    df['depression'] = df['depression'].map(yes_no_map)
    df['anxiety'] = df['anxiety'].map(yes_no_map)
    df['panic_attack'] = df['panic_attack'].map(yes_no_map)
    df['treatment'] = df['treatment'].map(yes_no_map)

    #Encode gender with one-hot encoding
    gender_map = {'Male': 1, 'Female': 0}
    df['gender'] = df['gender'].map(gender_map)

    #Encode academic_year
    year_mapping = {"year 1": 1,"Year 1": 1, "year 2": 2,"Year 2": 2,"year 3": 3, "Year 3": 3,"Year 4": 4, "year 4": 4}
    df['academic_year'] = df['academic_year'].map(year_mapping)

    #Encode course with label encoding
    label_encoder = LabelEncoder()
    df['course'] = label_encoder.fit_transform(df['course'])

    # Scale age column
    scaler = MinMaxScaler()

    #Encode gpa
    gpa_mapping = {
        '3.50 - 4.00 ': 5,
        '3.50 - 4.00': 5,
        '3.00 - 3.49': 4,
        '2.50 - 2.99': 3,
        '2.00 - 2.49': 2,
        '0 - 1.99': 1
    }

    df['gpa'] = df['gpa'].map(gpa_mapping)

    #Normalize to [0,1]
    df['gpa'] = scaler.fit_transform(df[['gpa']])
    #Normalize to [0,1]
    df['age'] = scaler.fit_transform(df[['age']])

    #Encode date        --> Proves unnecessary from sample data
    #df['date'] = df['date'].str.split(' ').str[0]
    #df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')

    #df['year'] = df['date'].dt.year
    #df['month'] = df['date'].dt.month
    #df['day'] = df['date'].dt.day
    #df.drop(columns=['date'], inplace=True)
    df.drop(columns=['date'], inplace=True)

    print(f"\nNull cell count after encoding: \n{df.isnull().sum()}")
    return df

def fill_null(df):
    #check for missing data and fill missing with mean (This dataset only has one row with age missing)
    print(f"\nNull cell count: \n{df.isnull().sum()}")
    df["age"] = df["age"].fillna(df["age"].mean())
    print(df)
    print(f"\nNull cell count after fill: \n{df.isnull().sum()}")
    return df

def data_preprocessing(df):
    # check for missing data and fill missing with mean (This dataset only has one row with age missing)
    fill_null(df)
    encode(df)
    visualize(df)

    return df

if __name__ == '__main__':
    pd.set_option('display.max_columns', 20)

    df = pd.read_csv('student_mental_health.csv')
    print(df)


    #Rename columns for better handling
    df.rename(columns={'Timestamp': 'date', 'Choose your gender': 'gender', 'Age': 'age',
                       'What is your course?': 'course', 'Your current year of Study': 'academic_year',
                       'What is your CGPA?': 'gpa', 'Marital status': 'marital_status',
                       'Do you have Depression?': 'depression', 'Do you have Anxiety?': 'anxiety',
                       'Do you have Panic attack?': 'panic_attack',
                       'Did you seek any specialist for a treatment?': 'treatment'}, inplace=True)

    df = data_preprocessing(df)
    print(f"\nResult dataset:\n{df}")
    print(df.describe())

