 📊 Customer Churn Prediction Web App

![banner_image](https://github.com/user-attachments/assets/574e8564-947d-4910-a94d-4a6346d1775c)


---

*Table of Contents*

* [Overview](#overview)
* [Features](#features)
* [Demo](#demo)
* [Technologies Used](#technologies-used)
* [Installation & Setup](#installation--setup)
* [Usage](#usage)
* [Screenshots](#screenshots)
* [Model Details](#model-details)
* [Future Improvements](#future-improvements)
* [License](#license)

---

*Overview*

This project is a **Customer Churn Prediction System** designed to predict which customers are likely to stop using a service.
It helps businesses, such as banks, telecoms, and subscription services, identify high-risk customers and take proactive retention measures.

The app is built with **Python**, **Streamlit**, and a **machine learning model (XGBoost)** trained on real customer data.

---

*Features*

* Predicts churn probability for individual customers
* Input forms for customer demographic and account details
* Displays actionable messages based on prediction:

   ✅ Likely to stay
   ⚠️ Likely to churn
* Feature importance visualization (optional with matplotlib)
* Scalable and lightweight web interface using Streamlit

---

 *Demo*

* Live App URL: https://futureml02-yyhd3ezukbep4bfsbfpunb.streamlit.app/

<img width="916" height="965" alt="image" src="https://github.com/user-attachments/assets/ee08dda5-6158-46cb-882b-7b993980c705" />

---

*Technologies Used*

* **Python 3.13**
* **Streamlit** – for web app frontend
* **NumPy & Pandas** – data handling
* **Scikit-learn** – preprocessing & model evaluation
* **XGBoost** – predictive modeling
* **Matplotlib** – feature visualization

---

 **Installation & Setup**

To run this project locally:

1. **Clone the repo:**

```bash
git clone https://github.com/YOUR_USERNAME/churn-webapp.git
cd churn-webapp
```

2. **Create a virtual environment:**

```bash
python -m venv venv
```

3. **Activate the environment:**

* **Windows:**

```bash
venv\Scripts\activate
```

* **Mac/Linux:**

```bash
source venv/bin/activate
```

4. **Install dependencies:**

```bash
python -m pip install -r requirements.txt
```

5. **Run the app locally:**

```bash
streamlit run app.py
```

---

## **Usage**

1. Open the app in your browser (local or deployed).
2. Fill in the customer details: gender, age, tenure, balance, number of products, credit card status, activity status, and estimated salary.
3. Click **Predict** to see if the customer is likely to churn.

---

## **Screenshots**

![input_form](https://github.com/user-attachments/assets/912a8bbb-a53d-4fe2-b25a-95898075a892)


![prediction_output](https://github.com/user-attachments/assets/03087d6e-6908-4435-bf10-9c2f5f70b5ba)



---

## **Model Details**

* **Model Type:** XGBoost Classifier
* **Training Dataset:** : https://www.kaggle.com/datasets/blastchar/telco-customer-churn
* **Metrics:**

  * ROC-AUC: 0.848
  * Precision, Recall, F1-score: See notebook or dashboard
* **Features Used:** gender, age, tenure, balance, number of products, credit card status, activity status, estimated salary

---

## **Future Improvements**

* Add more features to improve prediction accuracy
* Include interactive visualization of churn probabilities
* Integrate with a database for real-time prediction on new customer data
* Add user authentication for secure access

---

## **License**

This project is licensed under the MIT License.


Do you want me to do that next?
