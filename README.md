# Assignment 11.1 - Will the Customer Accept the Coupon?

Goal is to understand what factors make a car more or less expensive. As a result of our analysis, we should provide clear recommendations to our client -- a used car dealership -- as to what consumers value in a used car.

[Link to Coupons Dataset](https://github.com/atewari-bot/vehicle-value-prediction/blob/main/data/vehicles.csv)

[Link to Jupyter Notebook](https://github.com/atewari-bot/vehicle-value-prediction/blob/main/vehicle_value_prediction.ipynb)

## Business Understanding

The objective is to develop a preidction regression model to estimate the price of used cars based on relevant features which are key value parameters for the customers. Price of a used car could vary based on make, model, year, engine type, condition, fuel efficiency etc.

This involves identifying and quantifying key predictors by using exploratory data analysis and feature engineering methodologies. The goal is to improve pricing accuracy based on insights provided by data and help in strategic data-driven decision making.

## Data Understanding (EDA)

This is the first step of Exploratory Data Analysis (EDA)

* Dataset size is <b>426880 X 18</b>.
* Many columns have missing values.
* Size column have <b>~72%+</b> missing values.

## Data Preparation (EDA)

This is the next step of Exploratory Data Analysis (EDA)

### Data Type Transformation

* Performed data type transformation from <b>Object</b> data type to <b>String</b> type.

### Drop Columns/Rows

* Dropped <b>id</b> column, as it is not useful for our analysis.
* Dropped <b>size</b> column as <b>~72%</b> of the values are missing.

### Handle Missing Values and Data Transformation

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
            <li>Used Python package <b>vin</b>.</li>
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
            <li>Used Python package <b>vin</b></li>
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
          <li>Used Python package <b>vin</b> to impute missing values.</li>
          <li>Imputed similar types to one using contextual/domain understanding.</li>
          <li>Example: <b>sedan</b> and <b>sedan/saloon</b> are same type of vehicle</li>
          <li>Applied ordinal encoding with order of worst to best</li>
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
            <li>Performed imputation using IterativeImputer with RandomForestRegressor</li>
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

## Modeling

## Evaluation

## Deployment
