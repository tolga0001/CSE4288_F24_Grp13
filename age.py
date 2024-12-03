import sys

import pandas as pd
import os
#replace actual dataset with updated one that the Health_Score column is added
# first run combinedClasses.py and replace the fileName
updated_file_name = "Updated_Student_Mental_Health_Weighted.csv"
# Check if the file exists
if not os.path.exists(updated_file_name):
    print("First run 'combinedClasses.py' to generate the updated dataset.")
    sys.exit() #exit the script
data = pd.read_csv(updated_file_name)  # Replace 'data.csv' with your actual file path
print(data)
def categorize_health_score(score):
    if score == 0:
        return 'Low'
    elif score == 1 or score == 2:
        return 'Moderate'
    else:
        return 'High'

data['Health_Status'] = data['Health_Score'].apply(categorize_health_score)

# Show the new Health_Status column
print(data[['Health_Score', 'Health_Status']].head())
import matplotlib.pyplot as plt
# Example: Visualize health status by age group
age_group_counts = data.groupby('Age_group')['Health_Status'].value_counts().unstack()
age_group_counts.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Mental Health Status by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.show()
