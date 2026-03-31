# &#x20;Air Quality Index (AQI) Predictor

### Machine Learning College Project

!\[Python](https://img.shields.io/badge/Python-3.8+-blue)
!\[ML](https://img.shields.io/badge/ML-Scikit--Learn-orange)
!\[Status](https://img.shields.io/badge/Status-Complete-green)



## &#x20;Problem Statement

Air pollution is a major health crisis in India and globally. Predicting the **Air Quality Index (AQI)** accurately helps governments, citizens, and health agencies take preventive action before pollution levels become dangerous.

This project uses **Machine Learning** to predict AQI values based on key pollutant and weather parameters.



## &#x20;Objective

* Predict the AQI value based on pollutant levels and weather data
* Compare multiple ML models and select the best performing one
* Visualize key patterns in air quality data



## &#x20;Dataset Features

|Feature|Description|
|-|-|
|PM2\_5|Fine particulate matter (µg/m³)|
|PM10|Coarse particulate matter (µg/m³)|
|NO2|Nitrogen Dioxide (µg/m³)|
|SO2|Sulfur Dioxide (µg/m³)|
|CO|Carbon Monoxide (mg/m³)|
|Ozone|Ground-level Ozone (µg/m³)|
|Temperature|Ambient Temperature (°C)|
|Humidity|Relative Humidity (%)|
|Wind\_Speed|Wind Speed (m/s)|
|**AQI**|**Target Variable**|



## &#x20;ML Models Used

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor ✅ *(Best Performer)*



## &#x20;Evaluation Metrics

* **R² Score** (accuracy of prediction)
* **MAE** – Mean Absolute Error
* **RMSE** – Root Mean Squared Error



## &#x20;How to Run

### 1\. Clone the Repository

```bash
git clone https://github.com/YOUR\_USERNAME/air-quality-predictor.git
cd air-quality-predictor
```

### 2\. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3\. Run the Project

```bash
python aqi\_predictor.py
```

\---

## &#x20;Project Structure

```
air\_quality\_predictor/
├── aqi\_predictor.py      # Main ML script
├── aqi\_data.csv          # Generated dataset
├── eda\_plots.png         # EDA visualizations
├── model\_results.png     # Model comparison plots
├── requirements.txt      # Dependencies
└── README.md             # Project documentation
```



## &#x20;AQI Categories

|AQI Range|Category|Health Impact|
|-|-|-|
|0 - 50|Good|Minimal|
|51 - 100|Moderate|Acceptable|
|101 - 150|Unhealthy (Sensitive)|Risk for sensitive groups|
|151 - 200|Unhealthy|Everyone may be affected|
|201 - 300|Very Unhealthy|Health alert|
|301+|Hazardous|Emergency conditions|



## &#x20;Future Scope

* Deploy as a web app using **Streamlit**
* Use real-time API data (OpenWeatherMap, CPCB India)
* Implement deep learning (LSTM) for time-series AQI prediction
* Add city-wise AQI maps using Folium



## ABOUT ME

**YATHAM JATHINDRA REDDY**

COLLAGE :VIT BHOPAL 

COURSE  :CSE AI AND ML

YEAR    :2025 - 29


## &#x20;License

MIT License

