# Bank Customer Segmentation 🏦

This project focuses on segmenting bank customers based on their transactional behavior using Python and unsupervised machine learning.

---

## 📊 Project Goals
- Identify customer segments based on transaction amount, age, and account balance.
- Provide insights into customer behaviors across time, gender, and location.
- Visualize findings through intuitive plots.


---

## 🧱 Dataset Info
- **Rows**: 1,048,567
- **Columns**: 9
- Features include:
  - TransactionID, CustomerID, DOB, Gender, Location
  - Account Balance, Transaction Date, Time, Amount (INR)

---

## 🔍 Key Analyses
1. **Customer Demographics**
   - Age distribution & binning
   - Gender-based transaction analysis

2. **Transaction Patterns**
   - Time-slot activity (Morning, Afternoon, Evening, Night)
   - Location-based frequency (Top 10 cities)

3. **Clustering**
   - KMeans clustering based on Age, Account Balance, and Transaction Amount
   - Visual interpretation of clusters and segment meaning

---

## 🧠 Tools & Technologies
- **Python**: pandas, matplotlib, scikit-learn
- **Jupyter/VSCode**: for development
- **GitHub**: for version control and sharing

---
## 🔧 Setup

To install required packages, run:

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

Once dependencies are installed, you can run the clustering analysis using:

```bash
python scripts/customer_segmentation.py
```

Generated visualizations and reports can be found in the `reports/` folder.

---

## 📁 Folder Structure

```
Bank_Customer_Segmentation/
├── data/                  # Raw transaction data
├── reports/               # Plots and analysis outputs
├── scripts/               # Main Python scripts (e.g., customer_segmentation.py)
├── requirements.txt       # Python package dependencies
└── README.md              # Project overview
```


---

## 🙋🏻‍♀️ Author
- **Yelin Lee**  
- Graduate Student | Aspiring Data Analyst | Passionate about BI & Customer Insights  
- [LinkedIn](https://www.linkedin.com/in/your-link)
