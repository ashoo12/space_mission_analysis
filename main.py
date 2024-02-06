import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

from iso3166 import countries
from datetime import datetime, timedelta

pd.options.display.float_format = '{:,.2f}'.format
df_data = pd.read_csv('mission_launches.csv')

num_of_rows=df_data.shape[0]
num_of_columns=df_data.shape[1]
print(f"number of rows={num_of_rows}")
print(f"number of columns={num_of_columns}")

names_of_columns=df_data.columns
print(f"column_names{names_of_columns}")


nan_values=df_data.isna()
print(f"total no of nan values:{nan_values.sum().sum()}")

duplicated_values=df_data.duplicated().sum()
print(f"duplicated values={duplicated_values}")

# removing junk files
clean_df_data=df_data.dropna()
print(clean_df_data)

# Number of Launches per Company is shown in chart
launches=clean_df_data['Organisation'].value_counts()
# plot a bar chart
plt.figure(figsize=(10,6))
launches.plot(kind='bar',color='blue')
plt.title('Number of Space Mission Launches by Organization')
plt.xlabel('Organization')
plt.ylabel('Number of Launches')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.tight_layout()
plt.show()

# Number of Active versus Retired Rockets
rocket_status_counts=clean_df_data['Rocket_Status'].value_counts()
# print(f"rocket_status_counts={rocket_status_counts}")

active_rockets=rocket_status_counts.get('StatusActive')
print(f"active_rockets={active_rockets}")

retired_rockets=rocket_status_counts.get('StatusRetired')
print(f"retired_rockets={retired_rockets}")

# plot a bar chart
plt.figure(figsize=(10,6))
rocket_status_counts.plot(kind='bar',color=['green','red'])
plt.title('Number of Active versus Retired Rockets')
plt.xlabel('Rockets status')
plt.ylabel('No of rockets')
plt.xticks(rotation=0, ha='right')
plt.show()


# Distribution of Mission Status
mission_status_counts=clean_df_data['Mission_Status'].value_counts()
print(f"mission_status_counts={mission_status_counts}")

success=mission_status_counts.get('Success')
print(f"successful_missions={success}")

failure=mission_status_counts.get(['Failure','Partial Failure','Prelaunch Failure'])
print(f"failed_missions={failure}")

# plot a bar chart
plt.figure(figsize=(10,6))
mission_status_counts.plot(kind='bar',color=['green','red','yellow','orange'])
plt.title('Distribution of Mission Status')
plt.xlabel('Mission status')
plt.ylabel('No of missions')
plt.xticks(rotation=0, ha='right')
plt.show()

# How Expensive are the Launches?
price_of_launches=clean_df_data['Price']

# plot a histogram
bin_edges = [0, 20, 40, 60, 80, 100]
plt.figure(figsize=(10,6))
sns.histplot(price_of_launches,color="blue",bins=bin_edges,kde=True)
plt.title("Expense of launces")
plt.xlabel("Price")
plt.ylabel("Density")
# plt.grid(axis='y', alpha=0.75)
plt.xticks(rotation=90)
plt.show()


location=clean_df_data['Location']
# extracted country names from the location column
country_names=[]
for name in location:
    country_name=name.split()[-1]
    # changing some country names
    if country_name=='Russia' or country_name=='Sea':
        country_names.append('Russian Federation')
    elif country_name=='Mexico' or country_name=='Facility'or country_name=='Canaria':
         country_names.append('USA')
    elif country_name=='Site':
        country_names.append('Iran')
    elif country_name=='Zealand':
        country_names.append('New Zealand')
    else:
        country_names.append(country_name)

# alpha codes of country names by iso library of python
alpha_codes=[]
for alpha in country_names:

   alpha_code=countries.get(alpha)[2]
   alpha_codes.append(alpha_code)
clean_df_data['Location']=alpha_codes

#
#Use a Choropleth Map to Show the Number of Failures by Country

failure=clean_df_data[clean_df_data['Mission_Status']=='Failure']
print(failure)
# Group by country and count the number of failures
failure__per_year = failure['Location'].value_counts().reset_index()
failure__per_year.columns = ['Location', 'Failure_Count',]
print(failure__per_year)
# Choropleth graph
fig = px.choropleth(
    failure__per_year,
    locations='Location',
    color='Failure_Count',
    color_continuous_scale=px.colors.sequential.matter,
    title='Number of Failures by Country',
)

fig.update_geos(projection_type="natural earth")
fig.show()

# Create a Plotly Sunburst Chart of the countries, organisations, and mission status.
fig=px.sunburst(clean_df_data,path=['Location','Organisation'],color='Mission_Status',color_continuous_scale='Viridis')
fig.show()

# Money Spent by Organisation per Launch
total_money=clean_df_data.groupby('Organisation')['Price'].sum().reset_index(name='Total_Price')
print(total_money)

# Analyse the Amount of Money Spent by Organisation per Launch
money_per_launch=clean_df_data.groupby('Organisation')['Price'].value_counts()
print(money_per_launch)

#  Number of Launches per Year
clean_df_data['Date']=pd.to_datetime(clean_df_data['Date'], infer_datetime_format=True, errors='coerce')
print(clean_df_data['Date'])
clean_df_data['Year']=clean_df_data['Date'].dt.year
yearly_launches=clean_df_data.groupby('Organisation')['Year'].value_counts()
print(yearly_launches)

# Number of Launches Month-on-Month until the Present
clean_df_data['Month']=clean_df_data['Date'].dt.month
monthly_launches=clean_df_data['Month'].value_counts().reset_index(name='Launch_Count')
print(f"{monthly_launches}")
# highest number of launches in all time
max_launch=monthly_launches['Launch_Count'].max()
print(f"highest number of launches in all time:{max_launch}")


# Create a figure
fig, ax = plt.subplots(figsize=(12, 6))

#Plot the line chart
sns.lineplot(x=clean_df_data['Month'].index, y=clean_df_data['Month'], label='Monthly Launches', ax=ax)

# Set labels and title
ax.set_xlabel('Index')
ax.set_ylabel('Monthly Launches')
ax.set_title('Monthly Launches Over Time')

# Show legend
ax.legend()

# Show the chart
plt.show()
# rolling average on the month on month time series chart
monthly_launches_for_rol_avg=clean_df_data.groupby(['Organisation','Month']).size().reset_index(name='Launch_Count')
print(monthly_launches_for_rol_avg)
rolling_average=monthly_launches_for_rol_avg.groupby('Organisation')['Launch_Count'].rolling(window=12,min_periods=1).mean().reset_index(name='Rolling_Average')
print(rolling_average)
sns.lineplot(x=monthly_launches_for_rol_avg['Month'], y=rolling_average['Rolling_Average'], label="Rolling Average", linestyle='--')

plt.title('Month-to-Month Time Series with Rolling Average')
plt.xlabel('Month')
plt.ylabel('Launch Count')
plt.legend()
plt.show()

# Which months are most popular and least popular for launches?
max_lauch_id=monthly_launches['Launch_Count'].idxmax()
print(f'most popular month:{monthly_launches.loc[max_lauch_id]}')
min_lauch_id=monthly_launches['Launch_Count'].idxmin()
print(f'least popular month:{monthly_launches.loc[min_lauch_id]}')

#  Launch Price varied Over Time
launch_price_over_time=clean_df_data.groupby(['Date','Price','Organisation']).size().reset_index(name='Launch_Count')
print(launch_price_over_time)

#Plot the line chart

sns.barplot(data=launch_price_over_time,x='Price',y='Date', label='Launch Price varied Over Time')
plt.xticks(rotation=90)
# Set labels and title
plt.xlabel('Price')
plt.ylabel('Date')
plt.title('Launch Price varied Over Time')

# Show legend
plt.legend()
plt.show()

# Number of Launches over Time by the Top 10 Organisations
top_10_organisations=launch_price_over_time.groupby('Organisation')['Launch_Count'].sum().nlargest(10).index.to_list()
# print(top_10_organisations)

for org in top_10_organisations:
    data_top_10=clean_df_data.loc[clean_df_data['Organisation']==org,['Organisation', 'Date']]
    print(data_top_10)
    sns.barplot(data=data_top_10,x='Organisation',y='Date')
plt.ylabel('Date')
plt.xlabel('Oraganisation')
plt.xticks(rotation=45)

plt.show()

# Cold War Space Race: USA vs USSR
# Create a Plotly Pie Chart comparing the total number of launches of the USSR and the USA
# Hint: Remember to include former Soviet Republics like Kazakhstan when analysing the total number of launches.
ussr=clean_df_data[clean_df_data['Location'].isin(['RUS', 'KAZ'])]
print(ussr)
ussr_launch_count=len(ussr)
print(ussr_launch_count)
usa=clean_df_data[clean_df_data['Location']=='USA']
usa_launch_count=len(usa)
print(usa_launch_count)
fig=px.pie(values=[ussr_launch_count,usa_launch_count],names=['USSR','USA'],title='Total number of launches of the USSR and the USA')
fig.show()

#  Total Number of Launches Year-On-Year by the Two Superpowers
ussr_launch_year=ussr['Year'].value_counts()
print(f'ussr launh year:{ussr_launch_year}')
usa_launch_year=usa['Year'].value_counts()
print(f'usa launh year:{usa_launch_year}')
sns.lineplot(x=usa_launch_year,y=usa_launch_year.index,label='USA')
sns.lineplot(x=ussr_launch_year,y=ussr_launch_year.index,label='USSR')
plt.title('Total number of launches of the USSR and the USA')
plt.xlabel('Number of launches')
plt.ylabel('Year')
plt.legend()
plt.show()

# Total Number of Mission Failures Year on Year
# print(ussr['Mission_Status'].value_counts())
failed_ussr=ussr[ussr['Mission_Status']=='Failure']
failed_ussr_year=failed_ussr['Year'].value_counts().sort_index()
failed_usa=usa[usa['Mission_Status']=='Failure']
failed_usa_year=failed_usa['Year'].value_counts().sort_index()
print(failed_ussr_year)
print(failed_usa_year)
sns.lineplot(x=failed_ussr_year.index,y=failed_ussr_year,label='USSR')
sns.lineplot(x=failed_usa_year.index,y=failed_usa_year,label='USA')
plt.title('Total Number of Mission Failures Year on Year')
plt.xlabel('Number of Failed Missions')
plt.ylabel('Year')
plt.legend()
plt.show()

# Chart the Percentage of Failures over Time

failure_per_year=clean_df_data[clean_df_data['Mission_Status'] == 'Failure'].groupby('Year').size()
# print(failure_per_year)
launches_per_year = clean_df_data.groupby('Year').size()
# print(launches_per_year)
failure_percentage=(failure_per_year/launches_per_year)*100
# Create a line plot for the percentage of failures over time
plt.figure(figsize=(10, 6))
sns.lineplot(x=failure_percentage.index, y=failure_percentage, marker='o')

# Set labels and title
plt.title('Percentage of Failures Over Time')
plt.xlabel('Year')
plt.ylabel('Percentage of Failures')

# Show the plot
plt.show()

# For Every Year Show which Country was in the Lead in terms of Total Number of Launches up to and including including 2020)
total_lauches_per_country=clean_df_data.groupby('Location').size().reset_index(name='Launch_Count')
print(total_lauches_per_country.sort_values(by='Launch_Count', ascending=True))
launch_count_country=clean_df_data.groupby(['Location','Mission_Status']).size().reset_index(name='Launch_Count')
print(launch_count_country)
successful_launch_count_country=launch_count_country.loc[launch_count_country['Mission_Status']=='Success']
print(successful_launch_count_country.sort_values(by='Launch_Count', ascending=True))

# Create a Year-on-Year Chart Showing the Organisation Doing the Most Number of Launches
yearly_lauch_count=clean_df_data.groupby(['Year','Organisation']).size().reset_index(name='Launch_Count')
print(yearly_lauch_count)
sns.barplot(data=yearly_lauch_count,x='Year',y='Launch_Count',hue='Organisation')
plt.xlabel('Year')
plt.ylabel('Launch_Count')
plt.xticks(rotation=45)
plt.title('Year-on-Year Chart Showing the Organisation Doing the Most Number of Launches')
plt.show()