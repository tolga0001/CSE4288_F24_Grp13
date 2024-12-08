import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Data Preprocessing and Exploratory Data Analysis
class StudentMentalHealthAnalysis:
    def __init__(self, file_path):
        # Read the CSV file
        self.df = pd.read_csv(file_path)
        
        # Data Cleaning
        self.clean_data()
        
    def clean_data(self):
        # Rename columns for easier handling
        self.df.columns = [
            'timestamp', 'gender', 'age', 'course', 'study_year', 
            'cgpa', 'marital_status', 'depression', 'anxiety', 
            'panic_attack', 'sought_treatment'
        ]
        
        # Handle missing values
        # Replace empty strings with NaN
        self.df = self.df.replace(r'^\s*$', np.nan, regex=True)
        
        # Fill missing age with median
        self.df['age'] = pd.to_numeric(self.df['age'], errors='coerce')
        self.df['age'].fillna(self.df['age'].median(), inplace=True)
        
        # Encode categorical variables
        categorical_columns = [
            'gender', 'course', 'study_year', 'cgpa', 
            'marital_status', 'depression', 'anxiety', 
            'panic_attack', 'sought_treatment'
        ]
        
        # One-hot encode categorical columns
        for col in categorical_columns:
            if col in ['depression', 'anxiety', 'panic_attack', 'sought_treatment']:
                # Binary encoding for mental health related columns
                self.df[col] = (self.df[col] == 'Yes').astype(int)
            elif col == 'gender':
                self.df[col] = (self.df[col] == 'Female').astype(int)
    
    def basic_statistics(self):
        """Generate basic statistical overview of the dataset"""
        print("Dataset Basic Statistics:")
        print("\nTotal number of students:", len(self.df))
        print("\nGender Distribution:")
        print(self.df['gender'].value_counts(normalize=True))
        
        print("\nMental Health Conditions:")
        conditions = ['depression', 'anxiety', 'panic_attack', 'sought_treatment']
        for condition in conditions:
            print(f"{condition.replace('_', ' ').title()} Rate: {self.df[condition].mean()*100:.2f}%")
        
        print("\nAge Statistics:")
        print(self.df['age'].describe())
    
    def visualize_mental_health_by_gender(self):
        """Create visualizations of mental health conditions by gender"""
        plt.figure(figsize=(15,10))
        
        # Mental health conditions
        conditions = ['depression', 'anxiety', 'panic_attack', 'sought_treatment']
        
        for i, condition in enumerate(conditions, 1):
            plt.subplot(2, 2, i)
            # Group by gender and calculate condition rate
            condition_by_gender = self.df.groupby('gender')[condition].mean()
            
            condition_by_gender.plot(kind='bar')
            plt.title(f'{condition.replace("_", " ").title()} Rate by Gender')
            plt.ylabel('Proportion')
            plt.xlabel('Gender (0=Male, 1=Female)')
            plt.xticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig('mental_health_by_gender.png')
        plt.close()
    
    def visualize_mental_health_by_course(self):
        """Analyze mental health conditions by course"""
        # Group courses with low frequency
        course_counts = self.df['course'].value_counts()
        rare_courses = course_counts[course_counts < 5].index
        self.df['course_grouped'] = self.df['course'].apply(
            lambda x: x if x not in rare_courses else 'Other Courses'
        )
        
        plt.figure(figsize=(15,10))
        
        # Mental health conditions
        conditions = ['depression', 'anxiety', 'panic_attack', 'sought_treatment']
        
        for i, condition in enumerate(conditions, 1):
            plt.subplot(2, 2, i)
            # Calculate condition rate by course
            condition_by_course = self.df.groupby('course_grouped')[condition].mean().sort_values(ascending=False)
            
            condition_by_course.plot(kind='bar')
            plt.title(f'{condition.replace("_", " ").title()} Rate by Course')
            plt.ylabel('Proportion')
            plt.xlabel('Course')
            plt.xticks(rotation=90)
        
        plt.tight_layout()
        plt.savefig('mental_health_by_course.png')
        plt.close()
    
    def correlation_analysis(self):
        """Perform correlation analysis of mental health variables"""
        # Select relevant numeric columns
        correlation_columns = ['age', 'depression', 'anxiety', 'panic_attack', 'sought_treatment']
        correlation_matrix = self.df[correlation_columns].corr()
        
        plt.figure(figsize=(10,8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Between Mental Health Variables')
        plt.tight_layout()
        plt.savefig('mental_health_correlation.png')
        plt.close()
    
    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        self.basic_statistics()
        self.visualize_mental_health_by_gender()
        self.visualize_mental_health_by_course()
        self.correlation_analysis()
        print(self.df)

# Initialize and run analysis
analysis = StudentMentalHealthAnalysis('Student Mental health.csv')
analysis.run_full_analysis()