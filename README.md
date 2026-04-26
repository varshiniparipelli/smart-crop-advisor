# Smart Crop Advisor
An "AI-powered crop recommendation system" that helps farmers and agricultural planners choose the most suitable crop based on soil nutrients, environmental conditions, and resource availability.
Features
1.Crop Recommendation (ML-based)
  * Predicts best crop using N, P, K, temperature, humidity, pH, and rainfall
  * ~92% accuracy using Random Forest model
2.Soil Health Analysis
  * Soil nutrient matching score
  * Soil type & texture insights
  * Organic Matter (OM) analysis
3.Pest & Disease Risk Index
  * Predicts pest/disease risk (Low → Very High)
  * Suggests preventive measures
4.Water Requirement Planner
  * Compares crop water needs vs available resources
  * Suggests irrigation strategies
5.Advanced Visualizations
  * Interactive graphs using Plotly
  * Top crop predictions
  * Soil & environmental analysis charts
6.Multi-language Support**
  * Supports 10+ languages for accessibility
7.Smart Alert System**
  * Real-time alerts based on predictions and risks

# Tech Stack

* Frontend: Streamlit
* Backend: Python
* Machine Learning:Scikit-learn (Random Forest)
* Data Processing: Pandas, NumPy
* Visualization:Plotly

# Project Structure

smart-crop-advisor -> app.py -> data.py -> generate_and_train.py  -> model --> crop_model.pkl -> requirements.txt -> prediction_history.json

# Installation & Setup
1. Clone the repository
```bash
git clone https://github.com/varshiniparipelli/smart-crop-advisor.git
cd smart-crop-advisor
```
2. Install dependencies
```bash
pip install -r requirements.txt
```
3. Run the application
   
streamlit run app.py

# How It Works

1. User inputs:
   * Soil nutrients (N, P, K)
   * Temperature, humidity
   * pH level and rainfall
   * Soil type & water source
2. Machine Learning model predicts:
   * Best crop
   * Confidence score
3. System provides:
   * Soil match analysis
   * Pest risk index
   * Water requirement insights
   * Smart recommendation

# Use Cases

* Farmers for crop planning
* Agricultural students & researchers
* Smart farming applications
* Government/agri advisory systems

# Future Improvements

* Real-time weather API integration
* Location-based crop suggestions
* Plant disease detection using images
* Mobile app version


## Acknowledgements

* Inspired by smart agriculture solutions
* Built using open-source tools and datasets

