import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

def visualize(df):
    df.hist(bins='auto', figsize=(20, 20))
    plt.tight_layout()
    plt.show()

def encode(df):

    #one hot encoding
    df = pd.get_dummies(df, columns=['age'])
    df = pd.get_dummies(df, columns=['scholarship'])
    df = pd.get_dummies(df, columns=['gender'])

    #label encode depression label etc
    label_encoder = LabelEncoder()
    df['depression_label'] = label_encoder.fit_transform(df['depression_label'])
    df['stress_label'] = label_encoder.fit_transform(df['stress_label'])
    df['anxiety_label'] = label_encoder.fit_transform(df['anxiety_label'])
    df['academic_year'] = label_encoder.fit_transform(df['academic_year'])
    df['gpa'] = label_encoder.fit_transform(df['gpa'])


    #Frequency encode
    university_counts = df['university'].value_counts()
    df['university'] = df['university'].map(university_counts)
    department_counts = df['department'].value_counts()
    df['department'] = df['department'].map(department_counts)

    #z scale
    standard_scaler = StandardScaler()
    df['depression_value'] = standard_scaler.fit_transform(df[['depression_value']])
    df['stress_value'] = standard_scaler.fit_transform(df[['stress_value']])
    df['anxiety_value'] = standard_scaler.fit_transform(df[['anxiety_value']])


    #print(df)
    #print(df.columns)
    return df


if __name__ == '__main__':
    pd.set_option('display.max_columns', 70)

    df = pd.read_csv('mentalhealth_dataset_2.csv')

    #Rename columns
    df.rename(columns={
        '2. In a semester, how often have you been unable to stop worrying about your academic affairs? ': 'worry_per_week',
        '1. In a semester, how often you felt nervous, anxious or on edge due to academic pressure? ': 'anxious_per_week',
        '1. Age': 'age',
        '2. Gender': 'gender',
        '3. University': 'university',
        '6. Current CGPA': 'gpa',
        '5. Academic Year': 'academic_year',
        '7. Did you receive a waiver or scholarship at your university?': 'scholarship',
        '4. Department': 'department',
        '3. In a semester, how often have you had trouble relaxing due to academic pressure? ': 'relaxing_trouble',
        '4. In a semester, how often have you been easily annoyed or irritated because of academic pressure?': 'easily_annoyed',
        '5. In a semester, how often have you worried too much about academic affairs? ': 'worried_too_much',
        '6. In a semester, how often have you been so restless due to academic pressure that it is hard to sit still?': 'restless_academic_pressure',
        '7. In a semester, how often have you felt afraid, as if something awful might happen?': 'afraid_of_awful',
        'Anxiety Value': 'anxiety_value',
        'Anxiety Label': 'anxiety_label',
        '1. In a semester, how often have you felt upset due to something that happened in your academic affairs? ': 'upset_due_to_academic',
        '2. In a semester, how often you felt as if you were unable to control important things in your academic affairs?': 'unable_to_control_academic',
        '3. In a semester, how often you felt nervous and stressed because of academic pressure? ': 'nervous_and_stressed',
        '4. In a semester, how often you felt as if you could not cope with all the mandatory academic activities? (e.g, assignments, quiz, exams) ': 'unable_to_cope_academic_activities',
        '5. In a semester, how often you felt confident about your ability to handle your academic / university problems?': 'confident_academic_ability',
        '6. In a semester, how often you felt as if things in your academic life is going on your way? ': 'academic_life_on_track',
        '7. In a semester, how often are you able to control irritations in your academic / university affairs? ': 'control_academic_irritations',
        '8. In a semester, how often you felt as if your academic performance was on top?': 'top_academic_performance',
        '9. In a semester, how often you got angered due to bad performance or low grades that is beyond your control? ': 'angered_by_low_performance',
        '10. In a semester, how often you felt as if academic difficulties are piling up so high that you could not overcome them? ': 'overwhelmed_by_academic_difficulties',
        'Stress Value': 'stress_value',
        'Stress Label': 'stress_label',
        '1. In a semester, how often have you had little interest or pleasure in doing things?': 'little_interest_pleasure',
        '2. In a semester, how often have you been feeling down, depressed or hopeless?': 'feeling_depressed_or_hopeless',
        '3. In a semester, how often have you had trouble falling or staying asleep, or sleeping too much? ': 'sleep_trouble',
        '4. In a semester, how often have you been feeling tired or having little energy? ': 'feeling_tired_or_energy',
        '5. In a semester, how often have you had poor appetite or overeating? ': 'poor_appetite_or_overeating',
        '6. In a semester, how often have you been feeling bad about yourself - or that you are a failure or have let yourself or your family down? ': 'feeling_bad_about_self',
        '7. In a semester, how often have you been having trouble concentrating on things, such as reading the books or watching television? ': 'concentration_trouble',
        "8. In a semester, how often have you moved or spoke too slowly for other people to notice? Or you've been moving a lot more than usual because you've been restless? ": 'slow_movement_or_restlessness',
    '9. In a semester, how often have you had thoughts that you would be better off dead, or of hurting yourself? ': 'thoughts_of_hurt_or_death',
    'Depression Value': 'depression_value',
    'Depression Label': 'depression_label'
    }, inplace = True)

    numeric = df.select_dtypes(include=['number']).columns
    non_numeric = df.select_dtypes(exclude=['number']).columns


    df = encode(df)

    visualize(df)

    visualize(df)

    #Check for null
    print(df.isnull().any())
    print(df.columns)

    df.to_csv('processed_data_2.csv', index=False)