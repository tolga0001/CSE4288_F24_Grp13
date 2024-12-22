import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
import seaborn as sns
from scipy import stats

def visualize(df):
    # Graph numeric column distributions
    numeric_columns = ['age', 'gpa', 'sleep_quality', 'engagement', 'stress_level', 'study_hours', 'symptom']
    for col in numeric_columns:
        plt.hist(df[col], bins='auto', color='skyblue', edgecolor='black')
        plt.title(f'{col.capitalize()} Distribution')
        plt.xlabel(col.capitalize())
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

    # Graph academic year distribution (after encoding)
    if 'academic_year' in df.columns:
        plt.hist(df['academic_year'], bins=4, color='green', edgecolor='black')
        plt.title('Academic Year Distribution')
        plt.xlabel('Academic Year')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

    # Graph course frequency
    if 'course' in df.columns and not np.issubdtype(df['course'].dtype, np.number):
        course_counts = df['course'].value_counts()
        plt.bar(course_counts.index, course_counts.values, color='yellow')
        plt.xlabel('Course')
        plt.ylabel('Frequency')
        plt.title('Course Frequency')
        plt.xticks(rotation=45)
        plt.show()

    # Correlation matrix (for numeric data only)
    numeric_df = df.select_dtypes(include=[np.number])  # Select only numeric columns
    correlation_matrix = numeric_df.corr()
    plt.figure(figsize=(10, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

    # Skewness of numeric columns
    print("\nSkewness of numeric columns:")
    for col in numeric_df.columns:
        print(f"{col}: {numeric_df[col].skew()}")



from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def encode(df):
    # Map gender to numeric
    gender_map = {'Male': 1, 'Female': 0}
    df['gender'] = df['gender'].map(gender_map)

    # Encode academic_year
    year_mapping = {"year 1": 1, "Year 1": 1, "year 2": 2, "Year 2": 2, "year 3": 3, "Year 3": 3, "Year 4": 4,
                    "year 4": 4}
    df['academic_year'] = df['academic_year'].map(year_mapping)

    # Encode course with label encoding
    label_encoder = LabelEncoder()
    df['course'] = label_encoder.fit_transform(df['course'])

    # Normalize numeric columns
    scaler = MinMaxScaler()
    numeric_columns = ['age', 'gpa', 'engagement', 'stress_level', 'study_hours', 'sleep_quality', 'symptom']
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

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
    df = fill_null(df)
    df = encode(df)
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
    df.drop(columns=['HasMentalHealthSupport', 'date'], inplace=True)

    df = data_preprocessing(df)
    print(f"\nResult dataset:\n{df}")
    print(df.describe())

    df.to_csv('processed_data.csv', index=False)


