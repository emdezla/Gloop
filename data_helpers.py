import xml.etree.ElementTree as ET
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib
import os



def parse_data(data_type):
    data = []
    if data_type in ['glucose_level', 'finger_stick', 'basal', 'basis_heart_rate', 
                        'basis_gsr', 'basis_skin_temperature', 'basis_air_temperature', 'basis_steps']:
        for event in root.findall(f'.//{data_type}/event'):
            ts_str = event.get('ts')
            ts = datetime.strptime(ts_str, '%d-%m-%Y %H:%M:%S')
            if data_type in ['basal', 'basis_gsr', 'basis_skin_temperature', 'basis_air_temperature']:
                value = float(event.get('value'))
            else:
                value = int(event.get('value'))
            data.append([np.datetime64(ts), value])
    elif data_type == 'temp_basal':
        for event in root.findall(f'.//{data_type}/event'):
            ts_begin_str = event.get('ts_begin') or event.get('tbegin')
            ts_end_str   = event.get('ts_end') or event.get('tend')
            ts_begin = datetime.strptime(ts_begin_str, '%d-%m-%Y %H:%M:%S') if ts_begin_str else None
            ts_end   = datetime.strptime(ts_end_str, '%d-%m-%Y %H:%M:%S') if ts_end_str else None
            value = float(event.get('value'))
            data.append([np.datetime64(ts_begin), np.datetime64(ts_end), value])
    elif data_type == 'bolus':
        for event in root.findall(f'.//{data_type}/event'):
            ts_begin_str = event.get('ts_begin')
            ts_end_str   = event.get('ts_end')
            ts_begin = datetime.strptime(ts_begin_str, '%d-%m-%Y %H:%M:%S')
            ts_end   = datetime.strptime(ts_end_str, '%d-%m-%Y %H:%M:%S')
            dose = float(event.get('dose'))
            data.append([np.datetime64(ts_begin), np.datetime64(ts_end), dose])
    elif data_type == 'meal':
        for event in root.findall(f'.//{data_type}/event'):
            ts_str = event.get('ts')
            ts = datetime.strptime(ts_str, '%d-%m-%Y %H:%M:%S')
            meal_type = event.get('type')
            carbs = int(event.get('carbs'))
            data.append([np.datetime64(ts), meal_type, carbs])
    elif data_type == 'exercise':
        for event in root.findall(f'.//{data_type}/event'):
            ts_str = event.get('ts')
            ts = datetime.strptime(ts_str, '%d-%m-%Y %H:%M:%S')
            intensity = int(event.get('intensity'))
            duration = int(event.get('duration'))  # duration in minutes
            ts_end = ts + timedelta(minutes=duration)
            data.append([np.datetime64(ts), np.datetime64(ts_end), intensity])
    return np.array(data)

# Function to perform asof merge for a given measurement DataFrame.
def merge_measurement(df_perfect, df_meas, meas_col):
    # Backward merge: the most recent measurement not after the perfect timestamp.
    df_merge = pd.merge_asof(
        df_perfect,
        df_meas.rename(columns={'timestamp': 'prev_timestamp', meas_col: f'{meas_col}_value'}),
        left_on='perfect_timestamp',
        right_on='prev_timestamp',
        direction='backward'
    )
    # Forward merge: the earliest measurement not before the perfect timestamp.
    df_merge = pd.merge_asof(
        df_merge,
        df_meas.rename(columns={'timestamp': 'next_timestamp', meas_col: f'{meas_col}_value_next'}),
        left_on='perfect_timestamp',
        right_on='next_timestamp',
        direction='forward'
    )
    return df_merge

# Modified threshold in minutes
THRESHOLD = 5

# Function to choose the closest measurement timestamp and compute time difference in minutes.
def choose_nearest(row):
    pt = row['perfect_timestamp']
    prev = row.get('prev_timestamp', pd.NaT)
    nxt = row.get('next_timestamp', pd.NaT)
    
    # Both missing.
    if pd.isna(prev) and pd.isna(nxt):
        return pd.NaT, np.nan
    # Only previous exists.
    elif pd.isna(nxt):
        diff = (pt - prev).total_seconds() / 60.0
        if diff < THRESHOLD:
            return prev, diff
        else:
            return pd.NaT, np.nan
    # Only next exists.
    elif pd.isna(prev):
        diff = (nxt - pt).total_seconds() / 60.0
        if diff < THRESHOLD:
            return nxt, diff
        else:
            return pd.NaT, np.nan
    else:
        diff_prev = (pt - prev).total_seconds() / 60.0
        diff_next = (nxt - pt).total_seconds() / 60.0
        if diff_prev <= diff_next and diff_prev < THRESHOLD:
            return prev, diff_prev
        elif diff_next < THRESHOLD:
            return nxt, diff_next
        else:
            return pd.NaT, np.nan

# Function to choose the measurement value based on the nearest valid timestamp.
def choose_value(row, meas_col):
    pt = row['perfect_timestamp']
    prev = row.get('prev_timestamp', pd.NaT)
    nxt = row.get('next_timestamp', pd.NaT)
    
    # Both missing.
    if pd.isna(prev) and pd.isna(nxt):
        return np.nan
    # Only previous exists.
    elif pd.isna(nxt):
        diff = (pt - prev).total_seconds() / 60.0
        if diff < THRESHOLD:
            return row[f'{meas_col}_value']
        else:
            return np.nan
    # Only next exists.
    elif pd.isna(prev):
        diff = (nxt - pt).total_seconds() / 60.0
        if diff < THRESHOLD:
            return row[f'{meas_col}_value_next']
        else:
            return np.nan
    else:
        diff_prev = (pt - prev).total_seconds() / 60.0
        diff_next = (nxt - pt).total_seconds() / 60.0
        if diff_prev <= diff_next and diff_prev < THRESHOLD:
            return row[f'{meas_col}_value']
        elif diff_next < THRESHOLD:
            return row[f'{meas_col}_value_next']
        else:
            return np.nan

def compute_iob(df, DOA_hours=5, interval_min=5):
    """
    Compute the Insulin On Board (IOB) for each row of the DataFrame.
    
    Parameters:
        df (pd.DataFrame): DataFrame with columns 'real_basal' (in U/hr) and 'bolus' (in U).
        DOA_hours (float): Duration of insulin action in hours.
        interval_min (float): Time difference between consecutive rows in minutes.
    
    Returns:
        pd.DataFrame: A new DataFrame with an added 'IOB' column.
    """
    # Convert time step to hours
    interval_hr = interval_min / 60.0
    
    # Determine the number of time steps for the duration of insulin action (DOA)
    decay_steps = int(DOA_hours / interval_hr)
    
    # Define a quadratic decay curve (from 1 at time=0 to 0 at time=DOA)
    t = np.linspace(0, 1, decay_steps)
    decay = 1 - (t**2) * (3 - 2*t)  # smooth quadratic decay
    
    # Convert the basal rate (U/hr) to basal amount (U) delivered in each interval
    basal_amt = df['real_basal'] * interval_hr
    
    # Use convolution to compute the cumulative effect of past boluses and basal doses
    # The convolution automatically sums the contributions from previous time steps weighted by decay.
    bolus_contrib = np.convolve(df['bolus'], decay, mode='full')[:len(df)]
    basal_contrib = np.convolve(basal_amt, decay, mode='full')[:len(df)]
    
    # Create a copy of the original DataFrame and add the IOB column
    df_out = df.copy()
    df_out['IOB'] = bolus_contrib + basal_contrib
    
    return df_out