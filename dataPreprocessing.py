import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
import seaborn as sns
from scipy import stats

def visualize(df):
    #Graph age distribution
    plt.hist(df['age'], bins=8, color='skyblue', edgecolor='black')
    plt.title('Age Distribution')
    plt.xlabel('age')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    #Graph gpa distribution
    plt.hist(df['gpa'], bins='auto', color='salmon', edgecolor='black')
    plt.title('gpa Distribution')
    plt.xlabel('GPA')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    plt.hist(df['sleep_quality'], bins=5, color='salmon', edgecolor='black')
    plt.title('sleep quality Distribution')
    plt.xlabel('Sleep Quality')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    plt.hist(df['engagement'], bins=5, color='salmon', edgecolor='black')
    plt.title('Academic Engagement Distribution')
    plt.xlabel('Engagement')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    plt.hist(df['stress_level'], bins=5, color='salmon', edgecolor='black')
    plt.title('Academic Stress Level Distribution')
    plt.xlabel('Stress Level')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    plt.hist(df['study_hours'], bins=19, color='salmon', edgecolor='black')
    plt.title('Study Hours Per Week Distribution')
    plt.xlabel('GPA')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    plt.hist(df['symptom'], bins=7, color='salmon', edgecolor='black')
    plt.title('Mental health problem Symptoms per week')
    plt.xlabel('Symptoms per week')
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
    plt.figure(figsize=(10, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
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

    #Normalize to [0,1]
    df['gpa'] = scaler.fit_transform(df[['gpa']])

    df['academic_year'] = scaler.fit_transform(df[['academic_year']])

    df['course'] = scaler.fit_transform(df[['course']])

    #df['age'] = scaler.fit_transform(df[['age']])
    df['age'] = scaler.fit_transform(df[['age']])

    df['engagement'] = scaler.fit_transform(df[['engagement']])

    df['stress_level'] = scaler.fit_transform(df[['stress_level']])

    df['study_hours'] = scaler.fit_transform(df[['study_hours']])

    df['sleep_quality'] = scaler.fit_transform(df[['sleep_quality']])

    df['symptom'] = scaler.fit_transform(df[['symptom']])

    #Encode date handle two different date types
    df['date_1'] = df['date'].where(df['date'].str.contains('/'), None)
    df['date_2'] = df['date'].where(df['date'].str.contains('-'), None)

    df['date_1'] = pd.to_datetime(df['date_1'], errors='coerce', dayfirst=True)

    df['date_2'] = pd.to_datetime(df['date_2'], errors='coerce')

    df['date'] = df['date_1'].fillna(df['date_2'])

    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month

    df['year'] = label_encoder.fit_transform(df['year'])
    df['month'] = label_encoder.fit_transform(df['month'])

    df.drop(columns=['date', 'date_1', 'date_2'], inplace=True)
    df.drop(columns=['month', 'year'], inplace=True)

    print(f"\nNull cell count after encoding: \n{df.isnull().sum()}")
    return df

def fill_null(df):
    #check for missing data and fill missing with mean (This dataset only has one row with age missing)
    print(f"\nNull cell count: \n{df.isnull().sum()}")
    df["age"] = df["age"].fillna(df["age"].mean())
    df["gpa"] = df["gpa"].fillna(df["gpa"].mean())
    print(df)
    print(f"\nNull cell count after fill: \n{df.isnull().sum()}")
    return df

def data_preprocessing(df):
    # check for missing data and fill missing with mean (This dataset only has one row with age missing)
    fill_null(df)
    encode(df)
    #visualize(df)

    return df

if __name__ == '__main__':
    pd.set_option('display.max_columns', 20)

    #df = pd.read_csv('student_mental_health.csv')
    df = pd.read_csv('mentalhealth_dataset.csv')
    print(df)


    #Rename columns for better handling
    df.rename(columns={'Timestamp': 'date', 'Gender': 'gender', 'Age': 'age',
                       'Course': 'course', 'YearOfStudy': 'academic_year',
                       'CGPA': 'gpa', 'Depression': 'depression', 'Anxiety': 'anxiety',
                       'PanicAttack': 'panic_attack','SpecialistTreatment': 'treatment',
              'SymptomFrequency_Last7Days': 'symptom', 'SleepQuality': 'sleep_quality',
                       'StudyStressLevel': 'stress_level', 'StudyHoursPerWeek': 'study_hours', 'AcademicEngagement': 'engagement',},
              inplace=True)

    #Drop repeated column
    df.drop(columns=['HasMentalHealthSupport'], inplace=True)

    df = data_preprocessing(df)
    print(f"\nResult dataset:\n{df}")
    print(df.describe())
    visualize(df)

    df.to_csv('processed_data.csv', index=False)


