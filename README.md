# knn
# 🌸 K-Nearest Neighbors (KNN) Classification - Iris Dataset

This project is part of the **AI & ML Internship Task 6**. The objective is to understand and implement the K-Nearest Neighbors (KNN) algorithm for classification using the famous Iris dataset.

---

## 📌 Task Objective

- Implement KNN using Scikit-learn.
- Normalize features for better performance.
- Evaluate model performance using accuracy and confusion matrix.
- Visualize accuracy for different values of K.
- (Optional) Visualize decision boundaries.

---

## 📁 Dataset Used

- **Name**: Iris Dataset  
- **Source**: [UCI Machine Learning Repository](https://www.kaggle.com/datasets/uciml/iris)  
- **Features**:
  - Sepal Length
  - Sepal Width
  - Petal Length
  - Petal Width
- **Target**: Species (Setosa, Versicolor, Virginica)

---

## 🛠 Tools & Libraries Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

---

## 📊 How It Works

1. Loaded and cleaned the Iris dataset.
2. Scaled the feature values using `StandardScaler`.
3. Split the dataset into training and testing sets.
4. Trained KNN classifiers for different values of **K (1 to 10)**.
5. Selected the best K value based on accuracy.
6. Evaluated performance using:
   - Accuracy score
   - Confusion matrix
   - Classification report
7. Visualized:
   - K vs Accuracy plot
   - (Optional) KNN decision boundaries for 2D feature space

---

## 📈 Output Example

- Best K value: **1**
- Accuracy: **100%**
- Confusion Matrix:

