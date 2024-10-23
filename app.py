from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__, template_folder='templates')
app.config['TEMPLATES_AUTO_RELOAD'] = True

data = pd.read_csv("./SALARY_PREDICTION.csv")
le_skills = LabelEncoder()
data['Skills'] = le_skills.fit_transform(data['Skills'])
le_interest = LabelEncoder()
data['Interest'] = le_interest.fit_transform(data['Interest'])
le_work_env = LabelEncoder()
data['Work_Environment'] = le_work_env.fit_transform(data['Work_Environment'])

X = data[['Years_Experience', 'Skills', 'Interest', 'Work_Environment', 'Projects']]
y = data['Salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

@app.route('/', methods=['GET', 'POST'])
def predict_salary():
    if request.method == 'POST':
        years_experience = float(request.form['years_experience'])
        skills = request.form['skills']
        interest = request.form['interest']
        work_environment = request.form['work_environment']
        projects = int(request.form['projects'])
        
        skill_map = dict(zip(le_skills.classes_, range(len(le_skills.classes_))))
        interest_map = dict(zip(le_interest.classes_, range(len(le_interest.classes_))))
        work_env_map = dict(zip(le_work_env.classes_, range(len(le_work_env.classes_))))

        skills_encoded = skill_map.get(skills, -1)
        interest_encoded = interest_map.get(interest, -1)
        work_env_encoded = work_env_map.get(work_environment, -1)

        if skills_encoded == -1 or interest_encoded == -1 or work_env_encoded == -1:
            return render_template('index.html', error="One or more input values are not recognized.")

        input_data = np.array([[
            years_experience,
            skills_encoded,
            interest_encoded,
            work_env_encoded,
            projects
        ]])
        input_data_scaled = scaler.transform(input_data)
        
        prediction = model.predict(input_data_scaled)[0]
        return render_template('result.html', prediction=prediction)
    
    return render_template('index.html')
if __name__ == '__main__':
    app.run(debug=True)
