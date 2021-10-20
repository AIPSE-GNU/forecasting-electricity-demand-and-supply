import pandas as pd
import numpy as np

def data_reshape(power_demand, period):


    df = np.array(power_demand)
    df = df.reshape(-1,8)
    df = df.transpose()
    df = df[1:, :]
    df = pd.DataFrame(df)


    # Change the data shape so that the column is hour and index is day to fit the VAE input shape

    a = 0
    b = 23
    power_demand = []
    try :
        for i in range(len(period)):
            data = df.loc[:,a:b]
            data.columns = [x for x in range(1, 25)]
            power_demand.append(data)
            a += 24
            b += 24
    except:
        pass

    power_demand = pd.concat(power_demand, axis=0)
    return power_demand

