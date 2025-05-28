📉 Customer Churn Prediction using Machine Learning
This project was developed during my internship at CodSoft, where I worked as a Machine Learning Intern. The objective was to create a machine learning model that can accurately predict whether a customer is likely to churn (i.e., stop using the service), enabling companies to take proactive steps for retention.

🧠 Problem Statement
Customer retention is critical for subscription-based businesses. Identifying customers at risk of churning allows businesses to offer tailored incentives, improve services, and minimize revenue loss. This project aims to use machine learning to classify customers based on historical behavior and service usage.

📊 Dataset
Source: Typically sourced from telecom customer databases

Size: ~7,000+ records

Features:

Demographic: gender, SeniorCitizen, Partner, Dependents

Services signed up: PhoneService, InternetService, StreamingTV, etc.

Account info: Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges

Target: Churn (Yes/No)
🔧 Workflow
graph LR
    A[Data Preprocessing] --> B[Exploratory Data Analysis]
    B --> C[Model Building]
    C --> D[Model Evaluation]
    D --> E[Results & Deployment]

1. Data Preprocessing
Handled missing values (e.g., blank TotalCharges)

Converted categorical variables using Label Encoding and One-Hot Encoding

Normalized numerical features (MonthlyCharges, TotalCharges) using MinMaxScaler

2. Exploratory Data Analysis (EDA)
Churn distribution analysis

Correlation heatmap

Feature-wise comparison between churned and retained customers

3. Model Building
Trained and evaluated several machine learning models:

Logistic Regression

Random Forest

Gradient Boosting

4. Model Evaluation
Used the following metrics to assess model performance:

Accuracy

Precision

Recall

F1-Score

Confusion Matrix
📈 Results
Model             	Accuracy	Precision (0/1) 	Recall (0/1)	 F1-Score (0/1)
Logistic Regression 81.6%     	-                	  -                  	-
Random Forest      	86.75%	    -                 -                 -
Gradient Boosting  	86.6%     	0.88 / 0.76	    0.96 / 0.47   	0.92 / 0.58
Note:

"0" = Not Churn, "1" = Churn

Detailed classification report for Gradient Boosting:

text
Classification Report (Gradient Boosting):
              precision    recall  f1-score   support

           0       0.88      0.96      0.92      1607
           1       0.76      0.47      0.58       393

    accuracy                           0.87      2000
   macro avg       0.82      0.72      0.75      2000
weighted avg       0.86      0.87      0.85      2000
✅ The Random Forest Classifier performed the best overall and was selected for the final deployment phase.

📽️ Demo Video
🎥 Check out the working demo of the model in action:
[Attach or link your compressed video here]

🧰 Tools & Technologies
Python

Pandas, NumPy

Scikit-learn

Matplotlib, Seaborn

Jupyter Notebook

📌 Key Takeaways
Customer churn is a classification problem with real-world business value

Feature engineering and data preprocessing are critical for model performance

Recall is a key metric when predicting churn, as missing a potential churn is more costly than a false positive

Visualizations are vital to understand business behavior and improve communication with stakeholders

✅ Internship Info
🏢 Organization: CodSoft
💼 Role: Machine Learning Intern
📅 Duration: [Include if you'd like, e.g., May 2025]
📫 Let’s Connect
Feel free to connect with me on LinkedIn or reach out if you're working on similar ML projects!

🔗 Hashtags
#CustomerChurn #ChurnPrediction #MachineLearning #CodSoftInternship #Python #RandomForest #MLProject #DataScience #BusinessIntelligence #LinkedInProjects



