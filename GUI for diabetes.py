import tkinter as tk
from tkinter import messagebox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib


data = pd.read_csv(r'C:\Users\STUDENT\Desktop\Saksham Data Science\diabetes.csv')


features = data.drop('Outcome', axis=1)
target = data['Outcome']


X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = LogisticRegression()
model.fit(X_train, y_train)


joblib.dump(model, 'diabetes_model.pkl')
joblib.dump(scaler, 'scaler.pkl')


model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')


class DiabetesPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Diabetes Predictor")

        self.labels = []
        self.entries = []
        for i, col in enumerate(features.columns):
            label = tk.Label(root, text=col)
            label.grid(row=i, column=0, padx=10, pady=5)
            self.labels.append(label)

            entry = tk.Entry(root)
            entry.grid(row=i, column=1, padx=10, pady=5)
            self.entries.append(entry)

        self.predict_button = tk.Button(root, text="Predict", command=self.predict_diabetes)
        self.predict_button.grid(row=len(features.columns), column=0, columnspan=2, pady=10)

    def predict_diabetes(self):
        
        input_data = []
        for entry in self.entries:
            value = entry.get()
            if value == "":
                messagebox.showerror("Input Error", "Please fill all the fields")
                return
            input_data.append(float(value))

       
        input_data = scaler.transform([input_data])

        
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][prediction]

        
        result = "Diabetes" if prediction == 1 else "No Diabetes"
        messagebox.showinfo("Prediction Result", f"Prediction: {result}\nProbability: {probability:.2f}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DiabetesPredictorApp(root)
    root.mainloop()
