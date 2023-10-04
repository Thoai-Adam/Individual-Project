<<<<<<< HEAD
=======
![image](https://github.com/hitzuMan/Individual-Project/assets/118470135/19e8c3ab-e0cf-4045-81e8-25ce133ae4d3)
>>>>>>> 53e11e2b83a8b8778e570ec02ef29093157da0ef




# Breast Cancer Prediction Project

## Abstract

Breast cancer is a critical healthcare concern, and early detection plays a pivotal role in improving patient outcomes. This project presents a comprehensive approach to breast cancer prediction using machine learning techniques. The goal is to develop an accurate classification model capable of distinguishing between malignant and benign tumors based on various diagnostic features.

### Dataset

I utilized a dataset containing clinical and diagnostic information for breast cancer cases. This dataset comprises features such as mean radius, mean texture, mean smoothness, and more, which are used for classification.

### Data Dictionary

| **Column Name**    | **Data Type**   | **Description**                                 | **Measurement**               |
|--------------------|-----------------|-------------------------------------------------|------------------------------|
| concave points_mean        | Numeric         | Mean radius of the tumor                       | Numeric (millimeters)        |
| radius_worst      | Numeric         | Mean texture of the tumor                      | Numeric (units)              |
| perimeter_worst    | Numeric         | Mean perimeter of the tumor                    | Numeric (millimeters)        |
| concave points_worst          | Numeric         | Mean area of the tumor                         | Numeric (square millimeters) |
| diagnosis          | Categorical     | Diagnosis of tumor                             | Categorical (Malignant 'M', Benign 'B') |

### Methodology

I employed a multi-step approach to build an effective breast cancer prediction model:

1. **Data Preprocessing**: I conducted thorough data preprocessing, including handling missing values, encoding categorical data, and scaling features to ensure data quality and model compatibility.

2. **Feature Selection**: I selected a subset of informative features to reduce dimensionality and improve model efficiency. Key features included concave points_mean, radius_worst, perimeter_worst, and concave points_worst.

3. **Model Building**: Several machine learning algorithms were explored, including Logistic Regression, Decision Trees, Random Forests, K-Nearest Neighbors, and XGBoost. Hyperparameter tuning and cross-validation were performed to optimize model performance.

4. **Evaluation**: Models were evaluated using accuracy. The best-performing model was selected based on this metrics.

### Results

Our breast cancer prediction model achieved an accuracy rate of 95% on the test dataset, demonstrating its efficacy in early cancer detection. 

### Conclusion

This project underscores the significance of machine learning in breast cancer prediction and early detection. Our results highlight the potential for improved patient outcomes through accurate and timely diagnosis. I believe that the insights and methodologies presented here can contribute to ongoing efforts in the field of medical data analysis and healthcare.

For detailed implementation and code, please refer to the project repository.

---

