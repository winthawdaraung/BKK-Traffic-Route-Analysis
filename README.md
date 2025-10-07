# 🚌 Bangkok Bus Travel Time & Route Analysis

### 📍 Project Overview
This project aims to **estimate bus travel time and best route recommendation** for Bangkok's public bus routes using **GPS probe data**, **traffic congestion analysis**, and **machine learning models**.

It builds a complete **data analytics pipeline** — from raw data cleaning to predictive modeling and visualization — to support **route optimization** and **public transport efficiency improvement**.

---

## 🚀 Objectives
- Predict **bus travel speed and total travel time** along each route
- Estimate **waiting time** for passengers at stops
- Recommend the **fastest or most efficient bus route** between selected origin and destination
- Visualize **traffic congestion** and **travel performance** on an interactive dashboard

---

## 🗂️ Project Structure

```
BKK-TRAFFIC-ROUTE-ANALYSIS/
│
├── .devcontainer/
│   └── devcontainer.json                    # Dev container configuration
│
├── pages/                                   # Dashboard application
│   ├── main_page.py                        # Dashboard main interface
│   ├── route_analysis.py                   # Route travel time and congestion map
│   └── requirements.txt                    # Dashboard dependencies
│
├── BKK-Traffic-Route-Analysis-Notebooks/   # Analysis notebooks
│   ├── DataCleaning.ipynb                  # Data filtering and validation
│   ├── Processing.ipynb                    # Feature engineering & congestion clustering
│   ├── MODEL_Training.ipynb                # ML model training and evaluation
│   ├── BusTravelTime.ipynb                 # Travel time and waiting time estimation
│   ├── Visualization.ipynb                 # Map and dashboard visualization
│   └── bus_routes_scraping.ipynb           # Extract route geometry from OSM
│
├── BKK BUS TRAVEL TIME ANALYSIS AND OPTIMIZATION.pdf  # Final project report
│
└── README.md                                # This file
```

---

## ⚙️ Data Pipeline

### **1. Data Cleaning** (`DataCleaning.ipynb`)
- Remove invalid data points and duplicates
- Filter out unrealistic speeds (below 0 or above 150 km/h)
- Keep only records within Bangkok boundaries
- **Result:** Clean, structured GPS dataset for analysis

---

### **2. Feature Engineering** (`Processing.ipynb`)

**Derived attributes:**
- **Temporal:** `hour`, `day_of_week`, `is_weekend`, `is_rush_hour`
- **Spatial:** `lat_grid`, `lon_grid`, `distance_from_center`
- **Traffic context:** `distance_to_congestion`, `congestion_avg_speed`

**Key techniques:**
- **HDBSCAN clustering** to detect congestion zones dynamically
- **KDTree spatial queries** for fast nearest-congestion lookups

---

### **3. Model Training** (`MODEL_Training.ipynb`)

Each route has its own ML model to capture unique traffic behavior.

**Evaluation Metrics:**
- Mean Absolute Error (MAE)
- R² score
- 5-fold Cross Validation

Models and feature sets are saved as `.pkl` files for later inference.

---

### **4. Travel Time Estimation** (`BusTravelTime.ipynb`)

**Steps:**
1. Compute segment distance between consecutive coordinates using **Haversine formula**
2. Predict segment speed using the trained ML model
3. Calculate travel time per segment:
   ```
   t_segment = (distance_segment / predicted_speed) × 60
   ```
4. Sum across all segments to get **total route travel time**
5. Estimate **waiting time** from distance and predicted speed

---

### **5. Route Calculation**

1. Identify bus routes near **origin** and **destination** (within 0.5 km)
2. If both are on the same route → direct route
3. If not, find **transfer routes** using nearby route intersections
4. Merge both route legs and add transfer penalty (~7 min)
5. Compute total predicted time:
   ```
   Total = Travel Time + Waiting Time + Transfer Time
   ```
6. Recommend the route with **shortest total travel time**

---

### **6. Visualization & Dashboard** (`Visualization.ipynb` + `pages/`)

Built with **Folium** and **Plotly** for interactive visualization:
- Congestion heatmaps (HDBSCAN clusters)
- Route travel speed and delay visualization
- Dashboard comparison of routes by time, congestion, and reliability

---

## 🎯 Why Separate Model per Route?

Each bus route has **distinct traffic patterns**:
- Different areas (CBD vs suburban)
- Different average speeds and stop densities
- Unique relationships between features (rush hour impact, congestion distance)

**Benefits:**
- ✅ **Improved prediction accuracy**
- ✅ **Better interpretability** (feature importance per route)
- ✅ **Flexibility** for retraining specific routes later

---

## 📊 Notebooks Workflow

| Order | Notebook | Purpose |
|-------|----------|---------|
| 1 | `DataCleaning.ipynb` | Clean and validate raw GPS data |
| 2 | `Processing.ipynb` | Engineer features and detect congestion zones |
| 3 | `bus_routes_scraping.ipynb` | Scrape bus route geometries from OpenStreetMap |
| 4 | `MODEL_Training.ipynb` | Train and evaluate ML models per route |
| 5 | `BusTravelTime.ipynb` | Estimate travel and waiting times |
| 6 | `Visualization.ipynb` | Generate maps and visualizations |

---

## 🖥️ Dashboard

The interactive dashboard is built with **Streamlit** and located in the `pages/` folder:

- **`main_page.py`** - Main dashboard interface with route selection
- **`route_analysis.py`** - Detailed route analysis with congestion maps

### Running the Dashboard

```bash
cd pages/
pip install -r requirements.txt
streamlit run main_page.py
```

---

## 🧰 Tools & Libraries

| Category | Tools |
|----------|-------|
| **Programming** | Python 3.10+ |
| **Data Handling** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn, XGBoost |
| **Clustering** | HDBSCAN |
| **Geospatial** | GeoPandas, Shapely, KDTree, Folium |
| **Visualization** | Matplotlib, Plotly, Streamlit |
| **Data Sources** | OpenStreetMap, Longdo Traffic API |

---

## 🗺️ Visualization Highlights

- **Folium Heatmap:** Congestion zones by severity
- **Route Map:** Color-coded speed and time per segment
- **Dashboard:** Route comparison charts (speed, delay, total time)

---

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/BKK-Traffic-Route-Analysis.git
   cd BKK-Traffic-Route-Analysis
   ```

2. **Install dependencies:**
   ```bash
   pip install -r pages/requirements.txt
   ```

3. **Run Jupyter notebooks:**
   ```bash
   jupyter notebook
   ```
   Follow the notebooks in order (see Notebooks Workflow above)

4. **Launch the dashboard:**
   ```bash
   cd pages/
   streamlit run main_page.py
   ```

---

## 📈 Results

The system successfully:
- ✅ Predicts bus travel times with high accuracy (low MAE, high R²)
- ✅ Identifies congestion hotspots in Bangkok
- ✅ Recommends optimal routes considering traffic conditions
- ✅ Provides interactive visualizations for decision-making

---

## 📄 Documentation

- **`BKK BUS TRAVEL TIME ANALYSIS AND OPTIMIZATION.pdf`** - Comprehensive final presentation slides

---

## 🏁 Summary

This project demonstrates how **machine learning and spatial analytics** can improve **public transport performance analysis**.

By combining GPS data, congestion detection, and predictive models, the system estimates travel and waiting times, identifies congestion zones, and recommends efficient bus routes — supporting **smarter and more reliable urban mobility** in Bangkok.

---


