from date_range.date_setting import set_date
from data_reshape.data_reshape import data_reshape
from urllib.request import urlopen
from bs4 import BeautifulSoup
import os

# setting "start date" and "end date"
start_year = 2020
start_month = 1     # Entering 01 cause the error
start_day = 1

end_year = 2020
end_month = 12
end_day = 31


# Configure datetime in the format required for crawling
date, df_index = set_date(start_year,start_month,start_day, end_year,end_month,end_day)


land_power_demand = []


url_prefix = 'http://www.kpx.or.kr/www/contents.do?key=223&status=land&issueDate='
url_suffix = '&__encrypted=M0x5Tkcr56VXs5Vrg0Td%2BPfmTOXKR3vojylqtBYvwmiWCM1iGvXybDUoUhIs2cBnx5hZKRoPP76ZY8Tr0uCWlEfX8SUjfGg%2FkDJDx%2BU04bw%3D'

# Data crawling
for day in date:
    index = url_prefix + str(day) + url_suffix
    source = urlopen(index).read()
    source = BeautifulSoup(source, 'html')
    data = source.find_all('td')
    for power in range(len(data)):
        land_power_demand.append(data[power].text)

# Reshaping the data to fit the VAE input shape
land_power_demand = data_reshape(land_power_demand, df_index)
land_power_demand.index = df_index
land_power_demand = land_power_demand[str(start_year)+'-'+str(start_month)+'-'+str(start_day) : str(end_year)+'-'+str(end_month)+'-'+str(end_day)]

# Make save path
crwaling_data_save_folder_path = './power_demand/'
if not os.path.exists(crwaling_data_save_folder_path):
   os.makedirs(crwaling_data_save_folder_path)

land_power_demand.to_csv(crwaling_data_save_folder_path + 'land_power_demand.csv')