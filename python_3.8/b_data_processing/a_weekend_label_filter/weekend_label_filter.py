import pandas as pd
import os

filtered_data_save_folder_path = './labeled_data/'

df = pd.read_csv( filtered_data_save_folder_path + 'before_weekend_filter.csv', index_col=0)


# Weekend label filtering is conducted to increase the accuracy of the data
df = df[df['generated_weekend_label']==df['real_weekend_label']]

# Remove the used weekend label
columns = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13',
       '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24']

df = df[columns]
print(df)

if not os.path.exists(filtered_data_save_folder_path):
   os.makedirs(filtered_data_save_folder_path)

# df.to_csv(filtered_data_save_folder_path + 'after_weekend_filter.csv')
