import pandas as pd
import matplotlib.pyplot as plt

# Load the data into a pandas DataFrame
data = pd.read_csv("processed_data_2.csv")  # Replace with your actual file path

# Calculate the probabilities (frequencies) for each condition
prob_depression = (data['depression_label']).mean()
prob_anxiety = (data['anxiety_label']).mean()
prob_stress = (data['stress_label']).mean()

# Calculate the inverse of the frequency (the rarer, the higher the weight)
w_depression = 1 / prob_depression if prob_depression != 0 else 0
w_anxiety = 1 / prob_anxiety if prob_anxiety != 0 else 0
w_panic_attack = 1 / prob_stress if prob_stress != 0 else 0

# Function to create a weighted health score
def health_score(row):
    score = 0
    # Apply the weight for each condition
    score += w_depression * (row['depression_label'])
    score += w_anxiety*(row['anxiety_label'])
    score += w_panic_attack*(row['stress_label'])
    return score

# Apply the function to create a new 'Health_Score' column
data['Health_Score'] = data.apply(health_score, axis=1)

# Display the new health score column
#print(data[['Do you have Depression?', 'Do you have Anxiety?', 'Do you have Panic attack?', 'Health_Score']].head())

# Example analysis: Distribution of health scores
health_score_counts = data['Health_Score'].value_counts().sort_index()
print(health_score_counts)
data.to_csv("Updated_Student_Mental_Health_Weighted.csv", index=False)
# Visualization of the health score distribution
health_score_counts.plot(kind='bar', color='skyblue', figsize=(8, 5))
plt.title('Distribution of Health Scores (Weighted)')
plt.xlabel('Health Score')
plt.ylabel('Count')
