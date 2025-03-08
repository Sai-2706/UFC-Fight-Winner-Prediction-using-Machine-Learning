# UFC-Fight-Winner-Prediction-using-Machine-Learning

Project Overview:
This project aims to predict the winner of UFC fights using machine learning techniques. Initially, an input model with fewer features achieved **100% accuracy**, but it was likely overfitting. To improve real-world predictive power, additional features were incorporated, resulting in a **70% accuracy model**. Our goal is to enhance this accuracy by integrating new features like **odds-related metrics** and optimizing the model further.

Dataset:
The dataset contains historical UFC fight data, including fighter statistics, past performance metrics, and betting odds. Feature engineering is applied to create derived features such as:
- **UFC Fight Winner PredictionSignificant\_Strike\_Accuracy\_diff**: Measures striking accuracy difference between two fighters.
- **Odds\_diff**: Computes the difference in betting odds, which is expected to be a strong predictor of fight outcomes.
- Other fight-related statistics, including takedowns, strikes, and submissions.

Machine Learning Model:
- **Model Used**: LightGBM (LGBMClassifier)
- **Training Approach**:
  - Data preprocessing and feature engineering.
  - Train-test split for model evaluation.
  - Hyperparameter tuning for optimization.
- **Results**: The model currently achieves **\~70% accuracy** on unseen data.

Future Improvements:
- Further tuning of LGBM hyperparameters.
- Feature selection and dimensionality reduction to remove irrelevant data.
- Experimentation with other ensemble methods (e.g., XGBoost, Random Forest) to compare performance.
- Enhanced feature engineering using domain knowledge to extract more meaningful fight-related insights.

Conclusion: 
This project provides a solid foundation for predicting UFC fight winners using machine learning. By incorporating odds-based features and optimizing the model further, we aim to achieve higher accuracy and better generalization for unseen fights.
