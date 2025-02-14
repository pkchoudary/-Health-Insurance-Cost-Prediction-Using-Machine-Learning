# Health Insurance Cost Prediction Using Machine Learning

## Overview
This project aims to predict health insurance costs based on user demographics and medical information using machine learning techniques. The model is trained on a dataset containing various attributes such as age, BMI, smoking habits, and region to estimate insurance charges.

## Features
- Data preprocessing and feature engineering
- Exploratory data analysis (EDA)
- Multiple regression models for cost prediction
- Model evaluation and performance comparison
- Deployment using a web interface (optional)

## Technologies Used
- Python
- Pandas, NumPy (Data Handling)
- Matplotlib, Seaborn (Data Visualization)
- Scikit-Learn (Machine Learning)
- Flask or Streamlit (Optional for Deployment)

## Dataset
The dataset used in this project consists of the following columns:
- `age`: Age of the insured person
- `sex`: Gender (Male/Female)
- `bmi`: Body Mass Index (BMI)
- `children`: Number of dependents
- `smoker`: Whether the person is a smoker (Yes/No)
- `region`: Residential area (Northeast, Northwest, Southeast, Southwest)
- `charges`: Medical costs billed by health insurance

## Installation
To set up the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/health-insurance-prediction.git
   cd health-insurance-prediction
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Load and preprocess the dataset.
2. Perform exploratory data analysis (EDA) to understand patterns.
3. Train various regression models, such as:
   - Linear Regression
   - Decision Tree Regression
   - Random Forest Regression
   - Gradient Boosting Regression
4. Evaluate model performance using metrics like Mean Squared Error (MSE) and R-squared score.
5. (Optional) Deploy the model using Flask or Streamlit for user interaction.

## Model Training
Run the following command to train the model:
```bash
python train.py
```

## Deployment (Optional)
To deploy the model using Streamlit:
```bash
streamlit run app.py
```

## Results
The best-performing model is evaluated based on:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R-squared Score (R²)

## Future Improvements
- Adding more features such as lifestyle factors, medical history, and location-specific costs.
- Experimenting with deep learning models.
- Implementing an API for real-time predictions.

## License
This project is open-source and available under the MIT License.

## Contact
For any questions or contributions, please reach out .

