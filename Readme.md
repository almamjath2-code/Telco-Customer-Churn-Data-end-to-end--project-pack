# 📊 Customer Churn Prediction Project
# 📌 Overview

This project predicts customer churn for a telecommunications company using Machine Learning.

Customer churn refers to customers who stop using a company’s services.
The goal is to identify such customers early so the business can take retention actions.

The project includes data preprocessing, exploratory data analysis (EDA), feature engineering, model building, and deployment using AWS and Docker.

# 🌐 Live Demo

# 👉 Customer Churn Prediction App
http://customer-churn-app.s3-website.ap-south-1.amazonaws.com

# 🐳 Dockerized Application

This project is fully containerized using Docker to ensure:

Consistent runtime environment
Easy deployment
No dependency issues
Cloud-ready application
📦 Run using Docker
# 1️⃣ Build Docker Image
docker build -t customer-churn-app .
# 2️⃣ Run Docker Container
docker run -p 8000:8000 customer-churn-app
# 📄 Sample Dockerfile
FROM python:3.10

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
# ☁️ Deployment (AWS + Docker)
# 🌐 Frontend
# Hosted on AWS S3 Static Website Hosting
Built using HTML, CSS, JavaScript
# ⚙️ Backend API
Built using FastAPI / Flask
Containerized using Docker
Deployed on AWS Elastic Beanstalk / EC2
🚀 Deployment Flow
GitHub Repository
      ↓
Docker Image Build
      ↓
AWS Elastic Beanstalk / EC2
      ↓
Backend API Running in Container
      ↓
Frontend (S3) calls API
# 🗂 Dataset

The dataset contains customer details such as demographics, account information, and usage behavior.

Key Features:
Feature	Description
Age	Age of the customer
Gender	Customer gender
Tenure	Number of months stayed
MonthlyCharges	Monthly billing amount
TotalCharges	Total billing amount
Contract	Contract type
PaymentMethod	Payment method
Churn	Target variable
🛠 Tools & Technologies
Python
Pandas & NumPy
Matplotlib & Seaborn
Scikit-learn
 FastAPI
 Joblib
Docker
AWS S3
AWS Elastic Beanstalk
# 🧹 Data Preprocessing
Missing value handling
Encoding categorical variables
Feature scaling
Train-test split
# 📊 Exploratory Data Analysis (EDA)
Churn distribution analysis
Correlation heatmap
Customer behavior analysis
Contract vs churn insights
# 🤖 Machine Learning Models
Model	Accuracy	Precision	Recall	F1-Score
Random Forest (Test)	0.76	0.68	0.56	0.61
Random Forest (Train)	0.75	0.67	0.55	0.60

# Best Model: Random Forest Classifier

# 🚀 Web Application

Interactive app to predict whether a customer will churn or not based on input features.

# 📈 Project Workflow
Data collection
Data preprocessing
Exploratory Data Analysis
Feature engineering
Model training
Model saving
Docker containerization
AWS deployment
Web app integration
# 💡 Key Insights
Month-to-month contracts have highest churn
High monthly charges increase churn
Longer tenure reduces churn probability
Payment method impacts retention
# ⚡ Future Improvements
Add XGBoost / LightGBM models
Improve Docker optimization (multi-stage build)
Add CI/CD pipeline using GitHub Actions
Deploy using Kubernetes (advanced)
Add monitoring dashboard (CloudWatch)
# 📁 Project Structure
customer_churn_project/
│
├── data/
├── notebooks/
├── app.py
├── model.pkl
├── requirements.txt
├── Dockerfile
└── README.md
# 🤝 Contributing
Fork repository
Create feature branch
Commit changes
Push branch
Create Pull Request
# 📜 License

MIT License

# 👨‍💻 Author

Amjath

GitHub: https://github.com/almamjath2-code
Email: almamjath2@gmail.com