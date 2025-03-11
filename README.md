# Project-4 | Machine Learning in Healthcare – Predicting Diabetes and Heart Disease Risks 
Group 1 Members: Maha Pentakota, Sapir Madar Coulson, Michael Villeda, Ana Garcia, Daniel Grimm 

### Link to Streamlit Web URL to run predictive model: https://project-4-diabetes.streamlit.app/

## Project Proposal: Machine Learning in Healthcare – Predicting Diabetes

For our project, we developed a machine learning model to analyze and predict the risks of diabetes based on factors derived from a survey of over 250,000 individuals by the CDC. Using the common factors from this dataset we can have a new individual answer the same questions in an efficient manner and train the model using these answers. The dataset that was utilized was sourced from reputable organizations and agencies allowing public use. The model aims to visualize trends between symptoms and characteristics to deliver an unbiased and accurate result according to the data provided. The results provided may reveal patterns that aid in early detection.  However, consulting a medical professional is always advised.

## Data Sources

Publicly available datasets utilized included :
- https://gis.cdc.gov/grasp/diabetes/diabetesatlas-surveillance.html 
- https://www.nature.com/articles/s41598-021-90406-0 
- https://diabetesatlas.org/data/en/  
- https://www.cdc.gov/diabetes/php/data 
- https://www.who.int/data/gho/info/gho-odata-api 
- https://risktest-api.diabetes.org/#:~:text=The%20Risk%20Test%20API%20uses,will%20expire%20aŌer%2060%20minutes. 

Our primary dataset comes from the 2015 Behavioral Risk Factor Surveillance System provided by the CDC.  This dataset has over 250,000 records of individuals being asked about their health.
diabetes_binary_health_indicators_BRFSS2015.csv

In each dataset, the following symptoms are indicators of Prediabetes, Diabetes, or Heart Disease:
High Blood Pressure		
- High Cholesterol
- Cholesterol Check
- BMI
- Smoking Habits
- Alcohol Consumption
- History of Stroke
- Heart Disease Attacks
- Physical Activity
- Consumption of Fruits
- Consumption of Vegetables
- Accessibility to Healthcare
- General Health (scale of 1-excellent, to 5-poor)
- Sex
- Age
- Education
- Income

For our calculations to predict the risk of diagnosis, we will examine all the following variables except for income and education. While we acknowledge that socioeconomic factors can influence general health outcomes, for our project, we have excluded Income and Education from our predictive calculations as they don’t contribute meaningfully to the prediction model.  

In addition, our datasets contain records that indicate whether a patient has been diagnosed with diabetes. These records were derived from a variety of trusted data sources, including:
- Centers for Disease Control and Prevention (CDC)
- Indian Health Service (IHS)
- Agency for Healthcare Research and Quality (AHRQ)
- U.S. Census Bureau
- Published Research Studies
- Estimated percentages and the total number of people with diabetes and prediabetes were derived from the following sources:
- National Health and Nutrition Examination Survey (NHANES)
- National Health Interview Survey (NHIS)
- IHS National Data Warehouse (NDW)
- Behavioral Risk Factor Surveillance System (BRFSS)
- United States Diabetes Surveillance System (USDSS)
- U.S. Resident Population Estimates

## Data Clean-Up
Given the BRFSS vast data, there were 330 columns of patient data, date and time surveys were taken, and other more unrelated data according to the CDC and diabetesatlas.org, we focused on 22 distinguishable factors best to predict the risk of a diagnosis of diabetes 

## Machine Learning Predictive Model
The main file in the repository for the predictive model is the app.py file.  In this file, the below steps are taken to create and run this predictive machine learning model:
- The CSV dataset mentioned above is red into a Pandas dataframe
- The column headers are renamed into human-readable formats to be used later as user selections
- Categories for Age and Sex are created to be used in selectable radio buttons
- Streamlit is called, user inputs dictionaries are created, and fields are built for users to input Height and Weight (to calculate BMI), age, general health rating, gender, and selectable boxes with categories derived directly from the columns of the dataset
- These user input boxes, checkboxes, and radio buttons are organized with for loops across multiple columns in the Streamlit page building to make the GUI pleasant and legible
- A web URL is created by Streamlit from this code which any user can load.  A cloud-based virtualization is created by Streamlit to run the dependencies and properly load the code from the app.py file.  This Streamlit URL updates automatically if the app.py file is changed
- When the user enters values in these boxes, checks the checkboxes, and clicks on a value in the radio button, they are stored in the preloaded dictionaries as values to be used in the prediction model training
- When the user clicks the “Click here to begin training the prediction model” button, selected and entered dataset features are loaded as the X variable in the model training, and the target (y) variable is set to the Diabetes column showing if the person polled No Diabetes (0), or has prediabetes OR diabetes (1)
- This data is trained and split
- The model is defined with Keras layers, relu and sigmoid activation functions, and an adam optimizer
- The model begins to run, and each Epoch row as it runs is displayed in the Streamlit text box at the bottom of the GUI screen
- When the model finishes going through the Epochs and the model is evaluated, a prediction is made with the user-inputted data compared against the dataset
- If the predicted value is 1 (the user does have a high risk of having prediabetes and diabetes), a message of High Risk of Diabetes is displayed.  If the predicted value is 0 (the user does not have a risk of having diabetes), a message of Low Risk of Diabetes is displayed

## Tableau and Data Visualization
We loaded several of the datasets we sourced into Tableau in order to generate visualizations that properly display the relationship of some of the more severe factors as they relate to the liklihood of a Diabetes diagnosis (smoking, high blood pressure, high cholesterol, etc.)

## Libraries and Data Models Used
- Spark (reading in the dataset and creating Pandas dataframe)
- Streamlit (Python web app)
- Tableau (Data Visualization)
- Tensorflow/Keras (Predictive Machine Learning Model Training)
- Powerpoint (Presentation)

