import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

# Load the dataset
insurance_data_path = 'insurance.csv'
insurance = pd.read_csv(insurance_data_path)

def clean_dataset(insurance):
    """
    Cleans the insurance dataset by performing several preprocessing tasks:
    - Corrects the 'sex' column values to a standard format ('male', 'female').
    - Removes the dollar sign from the 'charges' column and converts it to float.
    - Drops negative 'age' values.
    - Converts negative 'children' values to zero.
    - Converts 'region' values to lowercase.
    - Drops rows with any missing values.
    
    Parameters:
    - insurance: pandas DataFrame, the insurance dataset.
    
    Returns:
    - DataFrame after cleaning.
    """
    insurance['sex'] = insurance['sex'].replace({'M': 'male', 'man': 'male', 'F': 'female', 'woman': 'female'})
    insurance['charges'] = insurance['charges'].replace({'\$': ''}, regex=True).astype(float)
    insurance = insurance[insurance["age"] > 0]
    insurance.loc[insurance["children"] < 0, "children"] = 0
    insurance["region"] = insurance["region"].str.lower()

    return insurance.dropna()

def create_and_evaluate_regression_model(insurance):
    """
    Prepares the data, fits a linear regression model, and evaluates it using cross-validation.
    
    Parameters:
    - insurance: pandas DataFrame, the cleaned insurance dataset.
    
    Returns:
    - A tuple containing the fitted sklearn Pipeline object, mean MSE, and mean R2 scores.
    """
    # Preprocessing
    X = insurance.drop('charges', axis=1)
    y = insurance['charges']
    categorical_features = ['sex', 'smoker', 'region']
    numerical_features = ['age', 'bmi', 'children']
    
    # Convert categorical variables to dummy variables
    X_categorical = pd.get_dummies(X[categorical_features], drop_first=True)
    
    # Combine numerical features with dummy variables
    X_processed = pd.concat([X[numerical_features], X_categorical], axis=1)
    # Scaling numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_processed)
    # Linear regression model
    lin_reg = LinearRegression()
    
    # Pipeline
    steps = [("scaler", scaler), ("lin_reg", lin_reg)]
    insurance_model_pipeline = Pipeline(steps)
    
    # Fitting the model
    insurance_model_pipeline.fit(X_scaled, y)
    
    # Evaluating the model
    mse_scores = -cross_val_score(insurance_model_pipeline, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
    r2_scores = cross_val_score(insurance_model_pipeline, X_scaled, y, cv=5, scoring='r2')
    mean_mse = np.mean(mse_scores)
    mean_r2 = np.mean(r2_scores)
    
    return insurance_model_pipeline, mean_mse, mean_r2

# Usage example
cleaned_insurance = clean_dataset(insurance)
insurance_model, mean_mse, r2_score = create_and_evaluate_regression_model(cleaned_insurance)
print("Mean MSE:", mean_mse)
print("Mean R2:", r2_score)

# Predict on validation data
validation_data_path = 'validation_dataset.csv'
validation_data = pd.read_csv(validation_data_path)

# Ensure categorical variables are properly transformed
validation_data_processed = pd.get_dummies(validation_data, columns=['sex', 'smoker', 'region'], drop_first=True)

# Make predictions using the trained model
validation_predictions = insurance_model.predict(validation_data_processed)

# Add predicted charges to the validation data
validation_data['predicted_charges'] = validation_predictions

# Adjust predictions to ensure minimum charge is $1000
validation_data.loc[validation_data['predicted_charges'] < 1000, 'predicted_charges'] = 1000

# Display the updated dataframe
validation_data.head()


As a Data Analyst, I was tasked with building a predictive model for a leading health insurance company to estimate customer healthcare costs. The objective was twofold: to help the company personalize its services by identifying cost drivers and to enable customers to better plan for their healthcare expenses. The model would be used to estimate future healthcare charges for both existing and new customers. Additionally, the predictions had to meet realistic business requirements, such as ensuring a minimum charge of $1,000.00 for all estimates, to align with the company’s operational standards and financial planning needs.

Key Insights and Results:
The final regression model achieved an R-Squared score of 0.78, exceeding the required threshold of 0.65. This indicates that the model explains 78% of the variance in healthcare charges, making it a reliable tool for prediction.
Key Drivers of Healthcare Costs: Smoking status was the most significant factor influencing healthcare costs, with smokers incurring substantially higher charges. Age and BMI were also critical predictors, with older individuals and those with higher BMI values experiencing increased healthcare expenses.
Region had a moderate impact, highlighting geographic disparities in healthcare costs.

Impact and Recommendations:
The predictive model provides the health insurance company with a powerful tool to: Personalize Services: By identifying high-cost customers (e.g., smokers or individuals with high BMI), the company can offer tailored wellness programs or preventive care initiatives to reduce future costs. Improve Financial Planning: Customers can receive personalized cost estimates, enabling them to plan their healthcare expenses more effectively. Optimize Risk Management: The company can use these insights to better assess risk and adjust premiums or coverage options accordingly.

Approach: Data Exploration and Cleaning: Performed EDA to identify trends and anomalies, handled missing values, corrected data types, and scaled numerical features for optimal model performance.
Model Development and Training: Selected key features (age, BMI, smoking status, region) using correlation analysis, trained regression models (Linear Regression, Random Forest), and selected the best-performing model with an R² score > 0.65.
Predictions for New Data: Tested the model on validation_dataset.csv, generated healthcare cost predictions, and ensured all estimates met the minimum charge threshold of $1,000.
Project Limitations and Future Improvements:

Limitations: The dataset may not fully capture all factors influencing healthcare costs, such as pre-existing conditions or lifestyle factors beyond smoking and BMI. The model assumes a linear relationship between features and healthcare costs, which may not fully reflect real-world complexities.
Future Improvements: Incorporate additional features such as medical history, family history, or socioeconomic factors to improve prediction accuracy. Explore advanced machine learning models (e.g., Gradient Boosting, Neural Networks) to capture non-linear relationships and further enhance performance. Perform hyperparameter tuning to optimize the selected model. Conduct a deeper analysis of regional disparities to identify actionable insights for location-specific strategies.

Skills Demonstrated:
This project demonstrates my ability to clean and analyze data, develop predictive models, and deliver actionable insights that drive business decisions. By leveraging machine learning, I was able to provide the health insurance company with a reliable tool to enhance customer service, optimize financial planning, and improve risk management strategies.
