import pandas as pd

# Assuming the 'data' DataFrame already contains the age column
# Example: data['Age'] contains the age of the students
data = pd.read_csv("Student Mental health.csv")  # Replace 'data.csv' with your actual file path
# Define the age bins and the labels
age_bins = [18, 24, 30, 40, 50, 60]  # Age ranges
age_labels = ['18-24', '25-30', '31-40', '41-50', '51-60']  # Corresponding labels

# Create a new column 'Age_group' with the binned values
data['Age_group'] = pd.cut(data['Age'], bins=age_bins, labels=age_labels, right=False)

# Display the new 'Age_group' column
print(data[['Age', 'Age_group']].head())

# Optionally, you can visualize the distribution of mental health issues across age groups
import matplotlib.pyplot as plt

# Plot the distribution of depression based on age groups
age_group_counts = data.groupby('Age_group')['Do you have Depression?'].value_counts().unstack()
age_group_counts.plot(kind='bar', stacked=True, figsize=(10, 6), color=['skyblue', 'salmon'])
plt.title('Depression by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.show()
