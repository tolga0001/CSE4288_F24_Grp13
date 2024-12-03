import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Load the data into a pandas DataFrame
data = pd.read_csv("Student Mental health.csv")  # Replace 'data.csv' with your actual file path

# Display the first few rows of the dataframe
print(data.head())

# Clean column names (if necessary)
data.columns = data.columns.str.strip()  # Removing extra spaces if any

# Basic statistics: Count of people with Depression, Anxiety, Panic Attack
depression_count = data['Do you have Depression?'].value_counts().get('Yes', 0)
anxiety_count = data['Do you have Anxiety?'].value_counts().get('Yes', 0)
panic_attack_count = data['Do you have Panic attack?'].value_counts().get('Yes', 0)

# Count of people seeking specialist treatment
seeking_specialist_count = data['Did you seek any specialist for a treatment?'].value_counts().get('Yes', 0)

# Display the results
print(f"People with Depression: {depression_count}")
print(f"People with Anxiety: {anxiety_count}")
print(f"People with Panic Attacks: {panic_attack_count}")
print(f"People who sought specialist treatment: {seeking_specialist_count}")

# Example of filtering rows with specific conditions
depression_data = data[data['Do you have Depression?'] == 'Yes']
anxiety_data = data[data['Do you have Anxiety?'] == 'Yes']

# Show how many students have both depression and anxiety
both_depression_anxiety = len(depression_data[depression_data['Do you have Anxiety?'] == 'Yes'])
print(f"People with both Depression and Anxiety: {both_depression_anxiety}")

# Plotting All Graphs in One Figure
fig, axs = plt.subplots(2, 2, figsize=(14, 10))  # 2 rows, 2 columns

# First plot: Count of People with Depression
axs[0, 0].bar(data['Do you have Depression?'].value_counts().index,
              data['Do you have Depression?'].value_counts().values,
              color=['skyblue', 'salmon'])
axs[0, 0].set_title('Count of People with Depression')
axs[0, 0].set_xlabel('Depression Status')
axs[0, 0].set_ylabel('Count')

# Second plot: Count of People with Anxiety
axs[0, 1].bar(data['Do you have Anxiety?'].value_counts().index,
              data['Do you have Anxiety?'].value_counts().values,
              color=['lightgreen', 'orange'])
axs[0, 1].set_title('Count of People with Anxiety')
axs[0, 1].set_xlabel('Anxiety Status')
axs[0, 1].set_ylabel('Count')

# Third plot: People Seeking Specialist Treatment
axs[1, 0].bar(data['Did you seek any specialist for a treatment?'].value_counts().index,
              data['Did you seek any specialist for a treatment?'].value_counts().values,
              color=['purple', 'yellow'])
axs[1, 0].set_title('People Seeking Specialist Treatment')
axs[1, 0].set_xlabel('Seeking Treatment')
axs[1, 0].set_ylabel('Count')

# Fourth plot: Mental Health by Course
mental_health_by_course = data.groupby('What is your course?').agg({
    'Do you have Depression?': lambda x: (x == 'Yes').sum(),
    'Do you have Anxiety?': lambda x: (x == 'Yes').sum()
})
mental_health_by_course.plot(kind='bar', ax=axs[1, 1])
axs[1, 1].set_title('Mental Health by Course')
axs[1, 1].set_xlabel('Course')
axs[1, 1].set_ylabel('Count')


data['Depression_flag'] = data['Do you have Depression?'].apply(lambda x: 1 if x == 'Yes' else 0)
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Age', y='Depression_flag', data=data)
plt.title('Age vs Depression')
plt.xlabel('Age')
plt.ylabel('Depression (Yes=1, No=0)')
plt.show()

# Heatmap of correlation between variables (numeric)
correlation_matrix = data[['Age', 'Depression_flag', 'Anxiety_flag']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# Adjust layout and show all graphs at once
plt.tight_layout()
plt.show()  # Display all plots together


# Eğer 'Date' veya benzeri bir tarih sütunu varsa, zamanla depresyon oranını analiz edebiliriz
data['Date'] = pd.to_datetime(data['Date'])  # Eğer tarih varsa
monthly_depression = data.groupby(data['Date'].dt.to_period('M'))['Do you have Depression?'].apply(lambda x: (x == 'Yes').mean())

# Zamanla değişimi görmek için grafik
plt.figure(figsize=(10, 6))
monthly_depression.plot(kind='line', marker='o', color='blue')
plt.title('Monthly Depression Rate')
plt.xlabel('Month')
plt.ylabel('Depression Rate')
plt.grid(True)
plt.show()
