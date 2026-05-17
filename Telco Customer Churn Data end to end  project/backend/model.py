import pandas as pd
import joblib

#1. Load the training data
train_comodel  = joblib.load("x_train_columns.pkl")
model  = joblib.load("customer_churn_model.pkl")

#2. New customer data
new_customer_data = {
    "Age": [28],
    "Gender": ["female"],
    "Tenure": [17],
    "MonthlyCharges": [121.38],
    "Contract": ["Month-to-month"],
    "PaymentMethod": ["Mailed check"],
    "TotalCharges": [2061.9]
}
#3. is have the leble encoded ender the 4 data fram  like this "car_age": car_age,


#4. Create a DataFrame for the new customer data and /
#not ender one hot and bynary encoded t in the data frame 
# when you write the same in to left age and for all i mean what is have in data colum name  that is write
input_df = pd.DataFrame([{
    "Age":new_customer_data['Age'][0],#[0]this is use for ender code new_customer_data['Age'] output 0    35 /
    "Tenure": new_customer_data['Tenure'][0],#but ender code new_customer_data['Age'][0] get output 35
    "MonthlyCharges": new_customer_data['MonthlyCharges'][0],
    "TotalCharges": new_customer_data['TotalCharges'][0]#new_customer_data this already a data frame in 2that reasone use [0]
}])


#5.
# Create all Category columns = 0
for col in train_comodel:  #one hot methot
    if col.startswith("Gender_") or col.startswith("Contract_") or col.startswith("PaymentMethod_"):
        input_df[col] = 0



# Convert list → single value
gender = new_customer_data['Gender'][0]
contract = new_customer_data['Contract'][0]
payment = new_customer_data['PaymentMethod'][0]


#5.1
Gender_col = "Gender_" + gender
if Gender_col in input_df.columns:
    input_df[Gender_col] = 1

#5.2
Contract_col = "Contract_" + contract
if Contract_col in input_df.columns:
    input_df[Contract_col] = 1

#5.3
PaymentMethod_col = "PaymentMethod_" + payment
if PaymentMethod_col in input_df.columns:
    input_df[PaymentMethod_col] = 1


#  # Set correct Category values = 1
# #5.1
# Gender_col = "Gender_" + new_customer_data['Gender']
# if Gender_col in input_df.columns:
#     input_df[Gender_col] = 1

# #5.2
# Contract_col = "Contract_" + new_customer_data['Contract']
# if Contract_col in input_df.columns:
#     input_df[Contract_col] = 1

# #5.3
# PaymentMethod_col = "PaymentMethod_" + new_customer_data['PaymentMethod']
# if PaymentMethod_col in input_df.columns:
#     input_df[PaymentMethod_col] = 1

# this full of 5 and 5.1,5.2,5.3 is same code 

# # Set correct Category values = 1
# for col in train_comodel:
#     if col.startswith("Gender_"):
#         Gender_col = "Gender_" + new_customer_data['Gender']
#         if Gender_col in input_df.columns:
#             input_df[Gender_col] = 1
#     elif col.startswith("Contract_"):
#         Contract_col = "Contract_" + new_customer_data['Contract']
#         if Contract_col in input_df.columns:
#             input_df[Contract_col] = 1
#     elif col.startswith("PaymentMethod_"):
#         PaymentMethod_col = "PaymentMethod_" + new_customer_data['PaymentMethod']
#         if PaymentMethod_col in input_df.columns:
#             input_df[PaymentMethod_col] = 1

#

#7. Match training columns
input_df = input_df.reindex(columns=train_comodel, fill_value=0)

#8. Prediction
prediction = model.predict(input_df)[0]

#this say the probability of the customer churning, which can be useful for understanding the confidence of the prediction.
# (optional probability)
if hasattr(model, "predict_proba"):
    prob = model.predict_proba(input_df)[0][1] #model.predict_proba only have classification
    print(f"Churn Probability: {prob:.1f}")#1f  is saying (.)0.6 tf 2f is saying (.)0.60 tf 3f is saying (.)0.600

print(f"Prediction (0=No, 1=Yes): {prediction}")