# Phone Usage Behavioral Analysis Using Clustering Algorithm

This project explores mobile phone usage behavior across different age groups using machine learning techniques. By analyzing behavioral metrics such as app usage time, screen-on time, battery drain, and data consumption, users are clustered based on their phone addiction severity. The goal is to uncover insights into usage patterns and determine if age is a significant factor in phone addiction.

## 🔍 Project Objectives

- Analyze behavioral data to detect patterns in smartphone usage.
- Apply clustering algorithms (K-Means, DBSCAN) to segment users by behavior.
- Use classification models (Decision Tree, Random Forest) to predict addiction severity.
- Test the correlation between age and phone usage metrics using statistical methods.

## 🧠 Technologies Used

- **Python**
- **Pandas**, **NumPy**
- **Scikit-learn**
- **Matplotlib**, **Seaborn**
- **Pickle**
- **Jupyter Notebook**

## 📊 Key Findings

- Clustering revealed distinct user behavior groups.
- Age showed minimal correlation with phone addiction.
- Behavioral features were more predictive than demographic data.
- Decision Tree and Random Forest classifiers achieved 100% accuracy on the dataset.

## 📁 Project Structure

📦 **phone-usage-behavior-clustering**  
├── 📂 [models/](./models) – Saved models (.pkl files)  
├── 📂 [data/](./user_behavior_dataset.csv) – Dataset (CSV)  
├── 📜 [Streamlit app code](./app.py) – Streamlit User Interface  
├── 📜 [README.md](./README.md) – Project overview  
├── 📜 [requirements.txt](./requirements.txt) – Dependencies  
└── 📜 [phone_behavior_analysis.ipynb](./phone_behavior_analysis.ipynb) – Main analysis notebook  


## 📌 Getting Started

1. Clone the repo  
   ```bash
   git clone https://github.com/yourusername/phone-usage-behavior-clustering.git
   ```
2. Install dependencies
```bash
  pip install -r requirements.txt
```
3. Run the notebook
Open phone_behavior_analysis.ipynb in Jupyter and follow the cells step by step.

##  📈 Results and Visualizations
All visual outputs (correlation heatmaps, PCA, confusion matrix) are available in the notebook and appendices.

## 📚 References
- Liu, X. et al. (2022). Smartphone usage and mental health.
- Zhou, H. et al. (2021). Behavioral clustering using K-means and DBSCAN.
- Smith, A. et al. (2020). Digital habits by generation.

## 🤝 Contribution
Pull requests and suggestions are welcome! Feel free to fork this repository and improve on the analysis or modeling process.

## 📬 Contact
For questions or collaborations, reach out to me on X: https://x.com/ADominion76816

License: MIT
