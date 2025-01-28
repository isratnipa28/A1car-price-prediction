import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import joblib  # For loading the saved model

# Initialize the app
app = dash.Dash(__name__)

# Load the saved model
loaded_model = joblib.load('random_forest_model.pkl')  # Make sure the model path is correct

# Feature names that the model expects
feature_names = ['year', 'km_driven', 'owner', 'mileage', 'engine', 'max_power', 'seats']

# App layout
# App layout
app.layout = html.Div([
    # Title and Instructions
    html.H1("Israt's Car Prediction System", style={'fontWeight': 'bold'}),
    
    # Instructions with non-bold H2
    html.H2("Please put mileage, maximum power, and year to predict the car price. If you don't know the values, you can use the default value.", 
            style={'whiteSpace': 'pre-line'}),

    # Input Fields for Mileage, Max Power, Year with added spacing
    html.Label("Enter Mileage:"),
    dcc.Input(id='mileage', type='number', value=19.39, step=0.01),  # Default value 19.39
    html.Br(), html.Br(), html.Br(),  # Added 3 line spaces between inputs
    
    html.Label("Enter Max Power:"),
    dcc.Input(id='max_power', type='number', value=82.4, step=0.1),  # Default value 82.4
    html.Br(), html.Br(), html.Br(),  # Added 3 line spaces between inputs
    
    html.Label("Enter Year:"),
    dcc.Input(id='year', type='number', value=2023, step=1),  # Default value 2023
    html.Br(), html.Br(), html.Br(),  # Added 3 line spaces between inputs

    
    # Predict Button
    html.Button('Predict', id='predict-button', n_clicks=0),
    
    # Add some space between the button and the result
    html.Br(),

    # Predicted Price Output (with currency in Baht)
    html.Div(id='predicted-price', style={'fontSize': '20px', 'fontWeight': 'bold'})
])


# Callback to update prediction
# Callback to update prediction
@app.callback(
    Output('predicted-price', 'children'),
    Input('predict-button', 'n_clicks'),
    Input('mileage', 'value'),
    Input('max_power', 'value'),
    Input('year', 'value')
)
def update_prediction(n_clicks, mileage, max_power, year):
    if n_clicks > 0:
        # Prepare the input data in the expected format
        km_driven = 0  # Replace with actual data or default value
        owner = 1       # Replace with actual data or default value
        engine = 1199   # Replace with actual data or default value
        seats = 5       # Replace with actual data or default value
        
        input_data = pd.DataFrame([[year, km_driven, owner, mileage, engine, max_power, seats]], columns=feature_names)
        
        # Make prediction using the loaded model
        predicted_price = loaded_model.predict(input_data)
        
        # Convert predicted price to Baht (example conversion rate)
        conversion_rate = 30  # Adjust this conversion rate as needed
        price_in_baht = predicted_price[0] * conversion_rate
        
        # Return the formatted predicted price
        return f"Predicted Car Price: {price_in_baht:,.2f}baht"  # Format as currency with two decimals

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)


