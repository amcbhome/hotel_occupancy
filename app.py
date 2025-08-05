import pandas as pd
import numpy as np
from datetime import date, timedelta

# --- Parameters for Data Generation ---
start_date = date(2023, 1, 1)
end_date = date(2025, 12, 31)
date_range = [start_date + timedelta(days=d) for d in range((end_date - start_date).days + 1)]

# --- Generate Independent Variables ---
data = {
    'Date': date_range,
    'Day_of_Week': [d.isoweekday() for d in date_range],
    'Month': [d.month for d in date_range],
}

df = pd.DataFrame(data)

# Feature Engineering
df['Is_Weekend'] = df['Day_of_Week'].apply(lambda x: 1 if x in [6, 7] else 0)

# Simulate holiday periods (e.g., Christmas, New Year, Easter, Summer)
df['Is_Holiday_Period'] = 0
df['Is_Holiday_Period'] = df.apply(lambda row: 1 if (row['Month'] == 12 and row['Date'].day >= 20) or \
                                                 (row['Month'] == 1 and row['Date'].day <= 5) or \
                                                 (row['Month'] in [7, 8]) else 0, axis=1)

# Simulate nearby events
event_dates = np.random.choice(df['Date'], size=50, replace=False)
df['Nearby_Event_Attendees'] = 0
for d in event_dates:
    df.loc[df['Date'] == d, 'Nearby_Event_Attendees'] = np.random.randint(1000, 50000)

# Simulate average room rate with a slight positive correlation to demand
df['Average_Room_Rate'] = 150 + df['Is_Weekend'] * 20 + df['Is_Holiday_Period'] * 50 + \
                           (df['Nearby_Event_Attendees'] > 0) * 30 + np.random.normal(0, 10, len(df))
df['Average_Room_Rate'] = df['Average_Room_Rate'].round(2)

# --- Generate Dependent Variable (Occupancy_Rate) using a linear model with noise ---
# Coefficients (Beta values)
beta_0 = 0.60  # Baseline occupancy
beta_weekend = 0.15
beta_holiday = 0.20
beta_event = 0.000002
beta_rate = -0.001 # A slight negative effect of price

# Model: Occupancy = B0 + B1*Weekend + B2*Holiday + B3*EventAttendees + B4*RoomRate + noise
df['Occupancy_Rate'] = beta_0 + \
                       beta_weekend * df['Is_Weekend'] + \
                       beta_holiday * df['Is_Holiday_Period'] + \
                       beta_event * df['Nearby_Event_Attendees'] + \
                       beta_rate * df['Average_Room_Rate'] + \
                       np.random.normal(0, 0.05, len(df)) # Add some random noise

# Ensure occupancy rate is within a plausible range (0 to 1)
df['Occupancy_Rate'] = df['Occupancy_Rate'].clip(0, 1).round(2)

# Reorder columns for better readability
df = df[['Date', 'Occupancy_Rate', 'Day_of_Week', 'Is_Weekend', 'Month', 'Is_Holiday_Period',
         'Nearby_Event_Attendees', 'Average_Room_Rate']]

# Save to a CSV file
df.to_csv('hotel_data_synthetic.csv', index=False)

print("Synthetic dataset `hotel_data_synthetic.csv` generated successfully.")
print("\nFirst 5 rows of the dataset:")
print(df.head())
