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

## License

This project is open-source and available under the MIT License.
