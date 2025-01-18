from flask import Flask, request, render_template
import pandas as pd
import joblib
from sqlalchemy import create_engine
from urllib.parse import quote

app = Flask(__name__)

# Load pre-trained models
processed1 = joblib.load('processed1.joblib')
svd_fit = joblib.load('truncated_svd.joblib')
kmeans_model = joblib.load('model.pkl')


def preprocess_data(df):
    # Aggregating data by Customer
    aggregated_data = df.groupby('Customer').agg({
        'Customer Lifetime Value': 'mean',
        'Total Claim Amount': 'sum',
        'Number of Policies': 'sum',
        'Monthly Premium Auto': 'mean',
        'Income': 'mean',
        'Response': lambda x: (x == 'Yes').sum(),
        'Number of Open Complaints': 'mean',
        'Months Since Last Claim': 'mean',
        'State': 'first',
        'Coverage': 'first',
        'Education': 'first',
        'EmploymentStatus': 'first',
        'Vehicle Class': 'first'
    }).reset_index()

    # Renaming columns
    aggregated_data.columns = [
        'Customer', 'Avg_Lifetime_Value', 'Total_Claim_Amount', 'Total_Policies',
        'Avg_Monthly_Premium', 'Avg_Income', 'Total_Positive_Responses',
        'Avg_Open_Complaints', 'Avg_Months_Since_Claim', 'State', 'Coverage',
        'Education', 'EmploymentStatus', 'Vehicle_Class'
    ]

    # Feature Engineering
    high_value_threshold = 10000
    premium_affordability_low = 50
    premium_affordability_high = 150

    aggregated_data['High_Value_Customer'] = (aggregated_data['Avg_Lifetime_Value'] > high_value_threshold).astype(int)
    aggregated_data['Cross_Sell_Potential'] = (aggregated_data['Total_Policies'] > 2).astype(int)
    aggregated_data['Engaged_Customer'] = (aggregated_data['Total_Positive_Responses'] > 0).astype(int)
    aggregated_data['Premium_Affordability'] = pd.cut(
        aggregated_data['Avg_Income'] / aggregated_data['Avg_Monthly_Premium'],
        bins=[-1, premium_affordability_low, premium_affordability_high, float('inf')],
        labels=['Low', 'Medium', 'High']
    )
    claim_amount_mean = aggregated_data['Total_Claim_Amount'].mean()
    claim_amount_std = aggregated_data['Total_Claim_Amount'].std()
    aggregated_data['Risk_Category'] = pd.cut(
        aggregated_data['Total_Claim_Amount'],
        bins=[-float('inf'), claim_amount_mean - claim_amount_std, claim_amount_mean + claim_amount_std, float('inf')],
        labels=['Low', 'Medium', 'High']
    )
    aggregated_data['Retention_Risk'] = pd.cut(
        aggregated_data['Avg_Open_Complaints'],
        bins=[-float('inf'), 0.5, 2.5, float('inf')],
        labels=['Low', 'Medium', 'High']
    )
    aggregated_data['Recency_Since_Last_Policy'] = df.groupby('Customer')['Months Since Policy Inception'].max().reset_index(drop=True)
    aggregated_data['Potential_Revenue_Increase'] = aggregated_data['Cross_Sell_Potential'] * aggregated_data['Avg_Monthly_Premium']

    # Prepare data for preprocessing
    final_data = aggregated_data.drop(columns=['Customer'])

    # Apply preprocessing
    insurance_clean = pd.DataFrame(processed1.transform(final_data), columns=processed1.get_feature_names_out())

    # Apply SVD
    pca_res = pd.DataFrame(svd_fit.transform(insurance_clean))

    # Apply KMeans clustering
    cluster_labels = kmeans_model.predict(pca_res)
    
    # Dictionary to map cluster labels to descriptive names
    cluster_label_names = {
          0: 'Balanced Portfolio',
          1: 'Moderate Income, Lower Cross-Sell Potential',
          2: 'Stable Revenue Contributors',
          3: 'High Income, Moderate Engagement'
    }

    # Combine results and assign cluster labels
    final_result = pd.concat([aggregated_data, pd.Series(cluster_labels, name='Cluster')], axis=1)
    final_result['Cluster_Label'] = final_result['Cluster'].map(cluster_label_names)

    return final_result


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process_csv():
    # Get user inputs for DB credentials
    db_user = request.form['db_user']
    db_password = request.form['db_password']
    db_name = request.form['db_name']
    
    # Create SQLAlchemy engine dynamically
    engine = create_engine(f"mysql+pymysql://{db_user}:{quote(db_password)}@localhost/{db_name}")
    
    if 'file' not in request.files:
        return render_template('errors.html', error='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('errors.html', error='No selected file')

    if file and file.filename.endswith('.csv'):
        df = pd.read_csv(file)

        try:
            processed_data = preprocess_data(df)
            
            # Display first 100 rows and save the rest to the database
            first_100 = processed_data.head(100)
            rest_data = processed_data.iloc[100:]
            
            # Save the rest to the database
            rest_data.to_sql('processed_insurance_data_new_label', engine, if_exists='replace', index=False)
            
            # Pass data to result.html and show success message
            message = "The first 100 rows are displayed, and the rest of the data has been successfully saved to the database."
            return render_template('result.html', tables=[first_100.to_html(classes='data', header="true")], message=message)
        except Exception as e:
            return render_template('errors.html', error=str(e))
    else:
        return render_template('errors.html', error='Invalid file format. Please upload a CSV file.')


if __name__ == '__main__':
    app.run(debug=True)

