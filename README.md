# 🛡️ Security Intelligence Data Pipeline
### *Unified Open Data Engine for AI-Driven Safety Insights*

**Author:** [Anthony Ikenna Ogbonna (OGBmetrix)](https://github.com/OGBmetrix)  
**Core Language:** Python  
**Data Format:** Parquet (optimized analytics dataset)  

---

## 🚀 Overview
The **Security Intelligence Data Pipeline** is a modular data engineering project that unifies multiple public safety and law enforcement datasets into one analytics-ready source.  
It serves as the data backbone for an **AI-powered Security Chatbot** — capable of delivering real-time threat insights, safety breakdowns, and policing intelligence for any given location.

---

## 🎯 Core Objective
To transform raw Toronto Open Data files into a **clean, unified security intelligence dataset**, enabling:
- Real-time **threat-level estimation** by neighbourhood or division  
- **Data-driven policing analytics** and public safety visualization  
- **Resource and efficiency analysis** across divisions  
- **AI chatbot integration** for interactive insights  

---

## 🧩 Data Sources Integrated
The pipeline merges more than **30 open datasets** across four categories:

| Layer | Example Datasets | Focus |
|-------|------------------|-------|
| **Traffic & Collisions** | CYCLIST_KSI, PEDESTRIAN_KSI, FATALS_KSI, TRAFFIC_COLLISIONS | Killed or Seriously Injured (KSI) metrics, collision density |
| **Crimes & Incidents** | Assault, Robbery, Theft Over, Break & Enter, Major Crime Indicators | Violent & property crime patterns |
| **Police & Operations** | Patrol Zones, Arrested & Charged Persons, Dispatch Calls, Search of Persons | Police response levels, operational coverage |
| **Budget & Resources** | Gross Operating Budget, Expenditures by Division, Personnel by Rank | Financial allocation and staffing ratios |

---

## ⚙️ Pipeline Features
- 🧠 **Automatic schema detection**  
  Detects and aligns location, date, and coordinate fields across datasets.
- 🌍 **Unified spatial model**  
  Standardizes all data under `location_name` and `year` for clean merging.
- 🤖 **Smart dataset classification**  
  Auto-categorizes files into *traffic, crime, operations,* or *budget* families.
- 📈 **Weighted Security Index**  
  Generates a normalized `security_score` (0–1) combining crime, collision, and police presence indicators.
- 🪶 **Audit report generation**  
  Automatically outputs summary stats, missing values, and merge validation.

---

## 🧠 Data Output
Final output: **`security_master.parquet`**

| Column | Description |
|---------|--------------|
| `location_name` | Unified spatial identifier (neighbourhood/division) |
| `year` | Year of observation |
| `violent_crime_count` | Sum of violent incidents (assault, robbery, shootings, etc.) |
| `property_crime_count` | Sum of property-related incidents |
| `total_crime_count` | Combined total of all reported crimes |
| `crime_rate_per_1000` | Normalized crime rate |
| `ksi_total` | Total killed or seriously injured (traffic) |
| `total_collisions` | Total recorded traffic collisions |
| `police_presence_index` | Combined measure of dispatch calls and arrests |
| `gross_budget` | Division-level total budget |
| `total_personnel` | Total police personnel per division |
| `spending_efficiency` | Ratio of enforcement activity to expenditure |
| `security_score` | Weighted safety index (0 = safe, 1 = high risk) |

---

## 📁 Repository Structure
