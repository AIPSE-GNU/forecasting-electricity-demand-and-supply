import pandas as pd
from datetime import datetime

def set_date(start_year, start_month, start_day, end_year, end_month, end_day):
    start = datetime(start_year, start_month, start_day)
    end = datetime(end_year, end_month, end_day)


    # Configure datetime in the format required for crawling
    index = pd.period_range(start=start, end=end, freq='W')
    df = pd.DataFrame(index=index)
    df.reset_index(inplace=True)
    df = df.astype('str')
    df = df.values

    date = []
    for i in range(len(df)):
        day = df[i][0][11:].replace('-', '')
        date.append(day)

    # setting index of dataframe
    day_index_start = df[0][0][:10]
    day_index_end = df[-1][0][11:]

    day_index_start = pd.to_datetime(day_index_start)
    day_index_end = pd.to_datetime(day_index_end)

    index = pd.period_range(start = day_index_start, end = day_index_end, freq='d')

    return date, index