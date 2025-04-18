# Assignment 11.1 - What Drives the Price of a Car?

Goal is to understand what factors make a car more or less expensive. As a result of our analysis, we should provide clear recommendations to our client -- a used car dealership -- as to what consumers value in a used car.

[Link to Vehicles Dataset](https://github.com/atewari-bot/vehicle-value-prediction/blob/main/data/vehicles.csv)

[Link to Jupyter Notebook](https://github.com/atewari-bot/vehicle-value-prediction/blob/main/vehicle_value_prediction.ipynb)

## Business Understanding

The objective is to develop a preidction regression model to estimate the price of used cars based on relevant features which are key value parameters for the customers. Price of a used car could vary based on make, model, year, engine type, condition, fuel efficiency etc.

This involves identifying and quantifying key predictors by using exploratory data analysis and feature engineering methodologies. The goal is to improve pricing accuracy based on insights provided by data and help in strategic data-driven decision making.

## Data Understanding (EDA)

This is the first step of Exploratory Data Analysis (EDA)

* Dataset size is <b>426880 X 18</b>.
* Many columns have missing values.
* Size column have <b>~72%+</b> missing values.

### Understanding Data via visualization

![Image](/images/top10_selling_car.png)

**Key Takeaways:** 
* Ford, Chevrolet, Toyata are the top selling cars in the order.

![Image](/images/top10_selling_car_by_model.png)

**Key Takeaways:** 
* Ford F-150 is top selling model of the car.

![Image](/images/top10_selling_car_by_year.png)

**Key Takeaways:** 
* Most numbers of car were sold for the year 2018.

![Image](/images/price_vs_odometer_scatter.png)

**Key Takeaways:** 
* Price and odometer readings are ngeatively correlated.

![Image](/images/count_plot_by_category_features.png)

**Key Takeaways:** 
* Most important car features which are highly corrleated with most number of cars are as follows:
  * Most Valued Fuel Type: Gas
  * Most Valued Vehicle Type: Sedan
  * Most Valued Transmission: Automatic
  * Most Valued Condition: Good
  * Most Valued Title Status: Clean
  * Most Valued Color: White
  * Most Valued Cyclinders: Zero cylinders (Electric or Hydrogen fuel vehicles)
  * Top car seeling state: CA

![Image](/images/price_vs_category_features.png)

**Key Takeaways:** 
* Most important car features which are highly corrleated with price of cars are as follows:
  * Fuel Type: Diesel and Other(Electric or Hydrogen Fuel)
  * Vehicle Type: Chasis-cab
  * Transmission Type: Other Type Transmission
  * Condition: Good
  * Title Status: Lien
  * Vehicle Color: White
  * Cyclinder configuration: 8 Cyclinders
  * State: West Virginia

## Data Preparation (EDA)

Data preperation is the next step of Exploratory Data Analysis (EDA)

### Data Type Transformation

* Performed data type transformation from <b>Object</b> data type to <b>String</b> type.

### Drop Columns/Rows

* Dropped <b>id</b> column, as it is not useful for our analysis.
* Dropped <b>size</b> column as <b>~72%</b> of the values are missing.

### Data Transformation

* Added new columns based on features interaction
  * <b>vehicle_age:</b> Absolute age of the vehicle in years.
  * <b>log_vehicle_age:</b> Vehicle does not decline linearly. So, log of age represents depreciation more accurately.
  * <b>vehicle_age_odometer_ratio:</b> (Vehicle wear rate) Ratio of odometer and vehicle age ratio
  * <b>vehicle_age_condition:</b> Helps identify if older cars are in surprisingly good/bad condition.
  * <b>title_good_condition:</b> Clean + good condition = better resale value.

* Feature Imputation/Transformation

  <table>
      <tr>
          <th>Feature Name</th>
          <th>Missing Values (%)</th>
          <th>Imputation/Transformation Techinque</th>
          <th>Description</th>
      </tr>
          <td>year</td>
          <td>0.282281</td>
          <td>
            <ul>
              <li>VIN based imputation</li>
              <li>Rows dropped</li>
              <li>Outlier Removal</li>
              <li>Datatype conversion</li>
            <ul>
          </td>
          <td>
            <ul>
              <li>Used Python package <b>vin</b> to impute missing values.</li>
              <li>Dropped 192 reamining rows with NaN value.</li>
              <li>Applied upper bound outlier removal at 99 percentile.</li>
              <li>Converted <b>year</b> column datatype from float to int</li>
            </ul>
          </td>
      <tr>
          <td>manufacturer</td>
          <td>4.133714</td>
          <td>
            <ul>
              <li>VIN based imputation</li>
              <li>JamesSteinEncoder Encoder</li>
            </ul>
          </td>
          <td>
            <ul>
              <li>Used Python package <b>vin</b> to impute missing values.</li>
              <li>Applied JamesSteinEncoder with RandomForestRegressor encoder</li>
            </ul>
          </td>
      </tr>
      <tr>
          <td>model</td>
          <td>1.236179</td>
          <td>
            <ul>
              <li>VIN based imputation.</li>
              <li>Contextual imputation.</li>
              <li>JamesSteinEncoder Encoder</li>
            </ul>
          </td>
          <td>
            <ul>
              <li>Used Python package <b>vin</b> to impute misisng values.</li>
              <li>Used manufacturer and ambigous model name to impute model with accurate values.</li>
              <li>Applied JamesSteinEncoder with RandomForestRegressor encoder</li>
            </ul>
          </td>
      </tr>
      <tr>
          <td>condition</td>
          <td>40.785232</td>
          <td>
            <ul>
              <li>Ordinal Encoding</li>
            </ul>
          </td>
          <td>
            <ul>
              <li>Applied ordinal encoding with order of worst to best</li>
            </ul>
          </td>
      </tr>
      <tr>
          <td>cylinders</td>
          <td>41.622470</td>
          <td>
            <ul>
              <li>Contextual Imputation</li>
              <li>Conditional Imputation</li>
              <li>IterativeImputer based Imputation</li>
            </ul>
          </td>
          <td>
            <ul>
              <li>Converted cylinders to their numeric representation to represent their ordinal importance.</li>
              <li>Electric vehicle should have 0 cylinders</li>
              <li>Performed imputation using IterativeImputer with RandomForestRegressor</li>
            </ul>
          </td>
      </tr>
      </tr>
          <td>fuel</td>
          <td>0.705819</td>
          <td>
            <ul>
              <li>OHE</li>
            </ul>
          </td>
          <td>
            <ul>
              <li>Applied one-hot encoding</li>
            </ul>
          </td>
      <tr>
          <td>odometer</td>
          <td>1.030735</td>
          <td>
            <ul>
              <li>Dropped rows.</li>
              <li>Removed Outliers.</li>
            </ul>
          </td>
          <td>
            <ul>
              <li>Dropped 4331 rows with NaN values.</li>
              <li>Applied upper bound outlier removal at 99 percentile.</li>
            </ul>
          </td>
      </tr>
      <tr>
          <td>title_status</td>
          <td>1.930753</td>
          <td>
            <ul>
              <li>Ordinal Encoding</li>
            </ul>
          </td>
          <td>
            <ul>
              <li>Applied ordinal encoding with order of worst to best</li>
            </ul>
          </td>
      </tr>
      <tr>
          <td>transmission</td>
          <td>0.598763</td>
          <td>
            <ul>
              <li>OHE</li>
            </ul>
          </td>
          <td>
            <ul>
              <li>Applied one-hot encoding</li>
            </ul>
          </td>
      </tr>
      <tr>
          <td>VIN</td>
          <td>37.725356</td>
          <td>
            <ul>
              <li>Column Dropped</li>
            </ul>
          </td>
          <td>
            <ul>
              <li>Dropped <b>VIN</b> column after apply VIN based imputation.</li>
            </ul>
          </td>
      </tr>
      <tr>
          <td>drive</td>
          <td>30.586347</td>
          <td>
            <ul>
              <li>VIN based imputation</li>
              <li>Ordinal Encoding</li>
            </ul>
          </td>
          <td> 
            <ul>
              <li>Used Python package <b>vin</b> to impute misisng values.</li>
              <li>Applied ordinal encoding with order of worst to best</li>
            </ul>
          </td>
      </tr>
      <tr>
          <td>type</td>
          <td>21.752717</td>
          <td>
            <ul>
              <li>VIN based imputation.</li>
              <li>Contextual imputation.</li>
              <li>Ordinal Encoding</li>
            </ul>
          </td>
          <td>
            <ul>
              <li>Used Python package <b>vin</b> to impute missing values.</li>
              <li>Imputed similar types to one using contextual/domain understanding.</li>
              <li>Example: <b>sedan</b> and <b>sedan/saloon</b> are same type of vehicle</li>
              <li>Applied ordinal encoding with order of worst to best</li>
            </ul>
          </td>
      </tr>
      <tr>
          <td>paint_color</td>
          <td>30.501078</td>
          <td>
            <ul>
              <li>Target based Encoding</li>
            </ul>
          </td>
          <td>
            <ul>
              <li>Applied JamesSteinEncoder with RandomForestRegressor encoder</li>
            </ul>
          </td>
      </tr>
      <tr>
          <td>price</td>
          <td>NA</td>
          <td>
            <ul>
              <li>Outlier removal/Mean imputation</li>
              <li>IterativeImputer based Imputation</li>
            </ul>
          </td>
          <td>
            <ul>
              <li>Applied IQR to identify outliers and imputed them with median of the price column.</li>
              <li>Imputed rows where price=0 with median where condition != NaN and title_status != Nan</li>
              <li>Dropped rows where price=0 and (condition=nan or title_status=nan)</li>
            </ul>
          </td>
      </tr>
      <tr>
          <td>region</td>
          <td>NA</td>
          <td>
            <ul>
              <li>Target based Encoding</li>
            </ul>
          </td>
          <td>
            <ul>
              <li>Applied JamesSteinEncoder with RandomForestRegressor encoder</li>
            </ul>
          </td>
      </tr>
      <tr>
          <td>state</td>
          <td>NA</td>
          <td>
            <ul>
              <li>Target based Encoding</li>
            </ul>
          </td>
          <td>
            <ul>
              <li>Applied JamesSteinEncoder with RandomForestRegressor encoder</li>
            </ul>
          </td>
      </tr>
  </table>

  ### Data Visualization

  ![Image](/images/vehicle_price_distribution.png)

  **Key Takeaways:**  
  * Difference in mean and median of vehicle price indicates some level of variance in the dataset.

## Modeling

* Shuffled the clean dataset and divide into 2 parts (9:1 ratio)
  * First data set us used for model training and testing
  * Second data set would be never seen dataset and would be used to validate model.
* Split first data set into train and test set using scikit-learn train_test_split() method
* Trained 8 types of models based on different regression algorithms.
  * LinearRegression
  * Ridge Regression
  * Lasso Regression
  * ElasticNet
  * XGBoostRegressor
  * HistGradientBoostingRegressor
  * DecisionTreeRegressor
  * RandomForestRegressor
* Performed hyperparameters tuning of the models using GridSearchCV to perform cross-validation and evaulate best performing model.

## Evaluation

### Summary of Model Evaluation

#### Original Model Evaluation
* Defined model evaluation method which will perform multiple tasks as below:
  * Calculate MSE, RMSE, MAE, R2 score for train and test target variable.
  * Plot line chart for actual and predicted target price values.
  * Calculate permutation importance of feature and plot a bar graph in the order of importance of features.
* Perform model evalaution for each of the 4 models based on error metrics.
* Plot line chart of MSE train and MSE test for each of the models trained.

#### Evaluation Adjustment
* Did few adjustments during multiple iterations of fixing issues and fine tuning model.
  * Error Metrics
    * Added Adjusted R2 Score
  * Hyperparameter Tuning
    * Used <b>Precompute</b> and <b>warm_start</b> parameters to reduce model training time for ElasticNet from 33 mins to 45 seconds.
  * Data Leakage Issue
    * Price imputation cause data leakage to the model. Due to which we saw R2 score of 0.99.
  * Model Accuracy Improvement
    * Orginally tried 4 regression models (LinearRegression, Ridge, Lasso and ElasticNet) and R2 score was 0.53
    * Tried 4 additional models (XGBoostRegressor, HistGradientBoostingRegressor, DecisionTreeRegressor and RandomForestRegressor) which improved R2 score to 0.82 and 0.8 for test data respectively.

#### Models Performance Metrics


| Metric             | Linear Regressor | Ridge Regressor | Lasso Regressor | ElasticNet Regressor | XGBRegressor | HistGradientBoosting Regressor | DecisionTreeRegressor | RandomForestRegressor |
|--------------------|------------------|------------------|------------------|------------------------|--------------|-------------------------------|------------------------|------------------------|
| MSE Train          | 4.441425e+07     | 4.956479e+07     | 4.955961e+07     | 4.955967e+07           | 1.827579e+07 | 3.049972e+07                  | 1.538371e+07           | 2.666361e+06           |
| MSE Test           | 4.399326e+07     | 4.912720e+07     | 4.912108e+07     | 4.912123e+07           | 2.305160e+07 | 3.105107e+07                  | 3.089348e+07           | 1.743582e+07           |
| RMSE Train         | 6.664402e+03     | 7.040226e+03     | 7.039859e+03     | 7.039863e+03           | 4.275020e+03 | 5.522655e+03                  | 3.922208e+03           | 1.632900e+03           |
| RMSE Test          | 6.632742e+03     | 7.009080e+03     | 7.008643e+03     | 7.008654e+03           | 4.801209e+03 | 5.572349e+03                  | 5.558190e+03           | 4.175622e+03           |
| MAE Train          | 4.331177e+03     | 4.727050e+03     | 4.724849e+03     | 4.724744e+03           | 2.654963e+03 | 3.450958e+03                  | 2.136398e+03           | 8.328990e+02           |
| MAE Test           | 4.309288e+03     | 4.706869e+03     | 4.704857e+03     | 4.704759e+03           | 2.915236e+03 | 3.487311e+03                  | 3.056769e+03           | 2.167751e+03           |
| R2 Train           | 7.156924e-01     | 6.827224e-01     | 6.827556e-01     | 6.827552e-01           | 8.830117e-01 | 8.047630e-01                  | 9.015247e-01           | 9.829319e-01           |
| R2 Test            | 7.187060e-01     | 6.858795e-01     | 6.859187e-01     | 6.859177e-01           | 8.526075e-01 | 8.014587e-01                  | 8.024664e-01           | 8.885149e-01           |
| Adjusted R2 Train  | 7.156688e-01     | 6.826961e-01     | 6.827292e-01     | 6.827288e-01           | 8.830020e-01 | 8.047468e-01                  | 9.015165e-01           | 9.829305e-01           |
| Adjusted R2 Test   | 7.186125e-01     | 6.857751e-01     | 6.858143e-01     | 6.858133e-01           | 8.525585e-01 | 8.013927e-01                  | 8.024007e-01           | 8.884779e-01           |

#### Models Loss Function Visualization

  ##### Mean Squared Error (MSE)

  ![Image](/images/model_mse_evaluation.png)

  ##### Root Mean Squared Error (RMSE)

  ![Image](/images/model_rmse_evaluation.png)

  ##### Mean Absolute Error (MAE)

  ![Image](/images/model_mae_evaluation.png)

  ##### R2 Score

  ![Image](/images/model_r2_evaluation.png)

  **Key Takeaways:**
  * RandomForestRegressor have the least value for loss function.
  * RandomForestRegressor have best R2 score value of ~0.89 for test and validation data set. This mean model is capturing 89% of the patterns that determine car prices based on the features provided (like age, mileage, condition, etc.).

#### Actual Vs Predicted Price Visualization

  ##### Linear Regression

  ![Image](/images/predicted_vs_actual_price_for_LinearRegression.png)

  ##### Ridge Regression

  ![Image](/images/predicted_vs_actual_price_for_Ridge.png)

  ##### Lasso Regression

  ![Image](/images/predicted_vs_actual_price_for_Lasso.png)

  ##### ElasticNet Regression

  ![Image](/images/predicted_vs_actual_price_for_ElasticNet.png)

  ##### XGBRegressor

  ![Image](/images/predicted_vs_actual_price_for_XGBRegressor.png)

  ##### HistGradientBoostingRegressor

  ![Image](/images/predicted_vs_actual_price_for_HistGradientBoostingRegressor.png)

  ##### DecisionTreeRegressor

  ![Image](/images/predicted_vs_actual_price_for_DecisionTreeRegressor.png)

  ##### RandomForestRegressor

  ![Image](/images/predicted_vs_actual_price_for_RandomForestRegressor.png)

#### Most Imprtant Feature Selection Visualization

  * Most important features as per best model based on RandomForestRegressor are as follows:

    | Features                   |   Permutation Importance Mean |   Permutation Importance Std |
    |----------------------------|-------------------------------|------------------------------|
    | model_encoded              |                   0.506307    |                  0.00192967  |
    | log_vehicle_age            |                   0.231115    |                  0.000989822 |
    | odometer                   |                   0.15158     |                  0.000844946 |
    | vehicle_drive              |                   0.0596375   |                  0.00036149  |
    | vehicle_age_odometer_ratio |                   0.0545906   |                  0.00036105  |
    | vehicle_type               |                   0.0542512   |                  0.000322732 |
    | vehicle_condition          |                   0.0404631   |                  0.00034835  |
    | manufacturer_encoded       |                   0.0397567   |                  0.000264473 |
    | num_cylinders              |                   0.0360877   |                  0.000351327 |
    | fuel_diesel                |                   0.0238663   |                  0.000322384 |
    | state_encoded              |                   0.0227645   |                  0.000298867 |
    | region_encoded             |                   0.0224454   |                  0.000365618 |
    | fuel_gas                   |                   0.0213357   |                  0.000261419 |
    | paint_color_encoded        |                   0.0204294   |                  0.000230751 |
    | transmission_other         |                   0.011612    |                  0.000200928 |
    | transmission_automatic     |                   0.00733399  |                  0.000224791 |
    | vehicle_title_status       |                   0.00677997  |                  0.000101094 |
    | fuel_other                 |                   0.00335671  |                  3.52621e-05 |
    | transmission_manual        |                   0.00167926  |                  4.037e-05   |
    | fuel_hybrid                |                   0.00017335  |                  6.66447e-06 |
    | fuel_electric              |                   8.05131e-05 |                  9.45524e-06 |
    | title_good_condition       |                   0           |                  0           |
    | vehicle_age_condition      |                   0           |                  0           |
    | vehicle_age                |                   0           |                  0           |

  ##### Linear Regression

  ![Image](/images/perm_importance_LinearRegression.png)

  ##### Ridge Regression

  ![Image](/images/perm_importance_Ridge.png)

  ##### Lasso Regression

  ![Image](/images/perm_importance_Lasso.png)

  ##### ElasticNet Regression

  ![Image](/images/perm_importance_ElasticNet.png)

  ##### XGBRegressor

  ![Image](/images/perm_importance_XGBRegressor.png)

  ##### HistGradientBoostingRegressor

  ![Image](/images/perm_importance_HistGradientBoostingRegressor.png)

  ##### DecisionTreeRegressor

  ![Image](/images/perm_importance_DecisionTreeRegressor.png)

  ##### RandomForestRegressor

  ![Image](/images/perm_importance_RandomForestRegressor.png)

## Deployment

* Utilized joblib library to dump and load model
* Used validation data set (unseen data) to validate model performance and observed consistency with training results.

### Post-deployment Models Loss Function Visualization

  ##### Mean Squared Error (MSE)

  ![Image](/images/model_mse_evaluation_for_validation.png)

  ##### Mean Absolute Error (MAE)

  ![Image](/images/model_mae_evaluation_for_validation.png)

  ##### R2 Score

  ![Image](/images/model_r2_evaluation_for_validation.png)


