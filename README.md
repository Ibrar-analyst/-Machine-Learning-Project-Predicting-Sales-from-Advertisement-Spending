# -Machine-Learning-Project-Predicting-Sales-from-Advertisement-Spending

## Overview

This is a machine learning project built using **Linear Regression** to predict product sales based on the advertisement spending on **TV**, **Radio**, and **Newspaper**. The goal is to showcase how advertising budget allocation affects sales performance. I used the `advertising.csv` dataset to train a linear regression model and make predictions.

Additionally, I built an **interactive Streamlit app** that allows users to input their advertisement budgets (TV, Radio, Newspaper) and get the predicted sales.

## Dataset

- The dataset used is the **Advertising Dataset**, which contains information about the advertising budget (TV, Radio, Newspaper) for a product, and the resulting **Sales**.
  
- Columns:
  - `TV`: Advertising budget spent on TV.
  - `Radio`: Advertising budget spent on Radio.
  - `Newspaper`: Advertising budget spent on Newspaper.
  - `Sales`: Total sales as a result of advertising.

## Technologies

- **Python** (v3.8+)
- **Libraries**:
  - `pandas` – For data handling and processing
  - `scikit-learn` – For building the Linear Regression model
  - `matplotlib` and `seaborn` – For visualizing the data and model performance
  - `numpy` – For mathematical operations
  - **Streamlit** – For building the interactive app

## How to Run the Project

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/sales-prediction-ml-project.git
    ```

2. Install the dependencies:
    ```bash
    cd sales-prediction-ml-project
    pip install -r requirements.txt
    ```

3. Run the Streamlit app:
    ```bash
    streamlit run streamlit_app/app.py
    ```

4. Alternatively, run the Jupyter Notebook to train and test the model:
    ```bash
    jupyter notebook notebooks/sales_prediction.ipynb
    ```

## Model Evaluation

- **Model used**: Linear Regression
- **Evaluation Metrics**: 
    - R² Score: A measure of how well the model fits the data.
    - RMSE: Root Mean Squared Error, to evaluate the model's prediction accuracy.

### Visualizations:
- **Regression Plot**: Visualizes the relationship between ad spending and sales.
- **Predicted vs Actual**: Compares the predicted sales with the actual sales for validation.

## Future Improvements

- Try other models such as **Ridge** or **Lasso** regression for comparison.
- Implement a **cross-validation** to improve model generalization.
- Use **interactive visualizations** to present results dynamically.
- 🔧 1. Experimenting with Advanced Models:

Exploring Ridge and Lasso Regression for better regularization.

Trying tree-based models like Random Forest and XGBoost to capture non-linear relationships.

Testing Neural Networks for even more complex predictions.

🔧 2. Hyperparameter Tuning & Cross-Validation:

Implementing GridSearchCV and RandomizedSearchCV for hyperparameter optimization.

Using K-fold Cross Validation for more reliable model performance estimates.

🔧 3. Interactive Visualizations:

Adding dynamic, interactive visualizations in Streamlit to better understand how TV, Radio, and Newspaper budgets impact sales.

Integrating Plotly for scatter plots, regression lines, and bar charts for feature importance.

🔧 4. Real-Time Data Integration:

Connecting to live advertising data sources (Google Ads, Facebook Ads API) to provide real-time predictions.

Creating a pipeline to automatically pull in new data and update predictions accordingly.

🔧 5. Deploying the Model:

Deploying the model with Streamlit Sharing, Heroku, or AWS for accessible web-based prediction services.

Building a robust API to interact with the model from other applications or platforms.

🔧 6. Adding More Predictive Features:

Including additional features like seasonality, geographical region, and past sales data to improve the model’s accuracy.

🔧 7. Model Explainability:

Adding tools like SHAP or LIME to make model predictions more interpretable and user-friendly.

🔧 8. User Authentication and Reports:

Enabling user authentication to provide personalized predictions for different users.

Integrating automated email reports with prediction summaries for businesses and stakeholders.

## License

This project is open-source and available under the MIT License.
