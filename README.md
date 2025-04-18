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
  * ~~<b>title_good_condition:</b> Clean + good condition = better resale value.~~

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
              <li>Dropped encoded feature after imputing cylinders based on manufacturer.</li>
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
              <li><s>OHE</s></li>
              <li>Dropped</li>
            </ul>
          </td>
          <td>
            <ul>
              <li><s>Applied one-hot encoding</s></li>
              <li>Dropped the feature</li>
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
              <li><s>Ordinal Encoding</s></li>
              <li>Dropped the feature</li>
            </ul>
          </td>
          <td>
            <ul>
              <li><s>Applied ordinal encoding with order of worst to best</s></li>
            </ul>
          </td>
      </tr>
      <tr>
          <td>transmission</td>
          <td>0.598763</td>
          <td>
            <ul>
              <li><s>OHE</s></li>
            </ul>
          </td>
          <td>
            <ul>
              <li><s>Applied one-hot encoding</s></li>
              <li>Dropped the feature</li>
            </ul>
          </td>
      </tr>
      <tr>
          <td>VIN</td>
          <td>37.725356</td>
          <td>
            <ul>
              <li>Dropped the feature</li>
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
              <li><s>Target based Encoding</s></li>
              <li><Dropped</li>
            </ul>
          </td>
          <td>
            <ul>
              <li><s>Applied JamesSteinEncoder with RandomForestRegressor encoder</s></li>
              <li><Dropped the feature</li>
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
              <li><s>Target based Encoding</s></li>
              <li><Dropped</li>
            </ul>
          </td>
          <td>
            <ul>
              <li><s>Applied JamesSteinEncoder with RandomForestRegressor encoder</s></li>
              <li><Dropped the feature</li>
            </ul>
          </td>
      </tr>
      <tr>
          <td>state</td>
          <td>NA</td>
          <td>
            <ul>
              <li><s>Target based Encoding</s></li>
              <li><Dropped</li>
            </ul>
          </td>
          <td>
            <ul>
              <li><s>Applied JamesSteinEncoder with RandomForestRegressor encoder</s></li>
              <li><Dropped the feature</li>
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
    * Used <b>Early Stopping</b> parameter for HistGradientBoostingRegressor.
  * Data Leakage Issue
    * Price imputation cause data leakage to the model. Due to which we saw R2 score of 0.99.
  * Model Accuracy Improvement
    * Orginally tried 4 regression models (LinearRegression, Ridge, Lasso and ElasticNet) and R2 score was 0.53
    * Tried 4 additional models (XGBoostRegressor, HistGradientBoostingRegressor, DecisionTreeRegressor and RandomForestRegressor) which improved R2 score to 0.82 and 0.8 for test data respectively.

#### Models Performance Metrics

| Metric              | Linear Regressor | Ridge Regressor | Lasso Regressor | ElasticNet Regressor | XGBRegressor | Hist GradientBoosting Regressor | DecisionTreeRegressor | RandomForestRegressor |
|---------------------|------------------|------------------|------------------|-----------------------|--------------|-------------------------------|------------------------|------------------------|
| MSE Train           | 47601929.10      | 49643882.38      | 49643882.84      | 49643885.75           | 13351305.52  | 20897676.92                  | 13121555.02            | 2757472.66             |
| MSE Test            | 48617621.72      | 50745609.62      | 50745595.17      | 50745702.40           | 24333799.65  | 30935852.03                  | 31571833.46            | 19984805.11            |
| RMSE Train          | 6899.42          | 7045.84          | 7045.84          | 7045.84               | 3653.94      | 4571.40                      | 3622.37                | 1660.56                |
| RMSE Test           | 6972.63          | 7123.60          | 7123.59          | 7123.60               | 4932.93      | 5562.00                      | 5618.88                | 4470.44                |
| MAE Train           | 4533.00          | 4727.06          | 4727.06          | 4726.99               | 2199.54      | 2826.55                      | 1883.84                | 775.18                 |
| MAE Test            | 4569.37          | 4768.14          | 4768.13          | 4768.08               | 2769.99      | 3299.13                      | 2931.57                | 2086.49                |
| R2 Train            | 0.70             | 0.68             | 0.68             | 0.68                  | 0.91         | 0.87                         | 0.92                   | 0.98                   |
| R2 Test             | 0.69             | 0.68             | 0.68             | 0.68                  | 0.84         | 0.80                         | 0.80                   | 0.87                   |
| Adjusted R2 Train   | 0.70             | 0.68             | 0.68             | 0.68                  | 0.91         | 0.87                         | 0.92                   | 0.98                   |
| Adjusted R2 Test    | 0.69             | 0.68             | 0.68             | 0.68                  | 0.84         | 0.80                         | 0.80                   | 0.87                   |

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
  * RandomForestRegressor have best R2 score value of ~0.87 for test and validation data set. This mean model is capturing 87% of the patterns that determine car prices based on the features provided.

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

  * A permutation importance mean of 0 means that shuffling that feature had no effect on the model’s performance — at all.
  * A permutation importance standard deviation (std) of 0 means that across all the permutations (random shuffles) done during the importance calculation, the feature’s impact on the model’s performance was exactly the same every time.
  * Most important features as per best model based on RandomForestRegressor are as follows:

    | Features                   |   Permutation Importance Mean |   Permutation Importance Std |
    |----------------------------|-------------------------------|------------------------------|
    | model_encoded              |                          0.71 |                            0 |
    | odometer                   |                          0.12 |                            0 |
    | vehicle_age                |                          0.1  |                            0 |
    | log_vehicle_age            |                          0.1  |                            0 |
    | num_cylinders              |                          0.08 |                            0 |
    | vehicle_type               |                          0.08 |                            0 |
    | vehicle_drive              |                          0.07 |                            0 |
    | vehicle_age_odometer_ratio |                          0.06 |                            0 |
    | vehicle_age_condition      |                          0.05 |                            0 |
    | vehicle_condition          |                          0.02 |                            0 |

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


