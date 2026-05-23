# Cognifyz Data Engineering Internship
# Name: Pranav Thakare
# College: G H Raisoni University, Amravati
# Ref: CTI/A1/C357821
# Date:23 May 2026

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── Task 1.1: Load and Inspect ──────────────────
df = pd.read_csv('Railway_info.csv')
# Clean dirty 'days' column - remove trailing 'd'
df['days'] = df['days'].str.strip()
df['days'] = df['days'].apply(lambda x: x[:-1] if str(x).endswith('d') and x not in ['Wednesday'] else x)
print("First 10 rows:")
print(df.head(10))

print("\nShape:", df.shape)
print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

# ── Task 1.2: Basic Statistics ──────────────────
print("\nTotal trains:", df.shape[0])
print("Unique source stations:", df['Source_Station_Name'].nunique())
print("Unique destination stations:", df['Destination_Station_Name'].nunique())

print("\nMost common source station:")
print(df['Source_Station_Name'].value_counts().head(3))

print("\nMost common destination station:")
print(df['Destination_Station_Name'].value_counts().head(3))

# ── Task 1.3: Data Cleaning ─────────────────────
# Fill missing values
df.fillna('UNKNOWN', inplace=True)

# Standardize station names to uppercase
df['Source_Station_Name'] = df['Source_Station_Name'].str.upper().str.strip()
df['Destination_Station_Name'] = df['Destination_Station_Name'].str.upper().str.strip()
df['Train_Name'] = df['Train_Name'].str.upper().str.strip()

print("\nAfter cleaning - sample:")
print(df.head(3))

# ── Task 2.1: Data Filtering ────────────────────
saturday_trains = df[df['days'] == 'Saturday']
print("Trains on Saturday:", len(saturday_trains))
print(saturday_trains[['Train_Name','Source_Station_Name','Destination_Station_Name']])

# Filter trains from Nagpur (my city!)
nagpur_trains = df[df['Source_Station_Name'].str.contains('NAGPUR', na=False)]
print("\nTrains from Nagpur:", len(nagpur_trains))
print(nagpur_trains[['Train_Name','Source_Station_Name','Destination_Station_Name']])

# ── Task 2.2: Grouping & Aggregation ───────────
trains_per_source = df.groupby('Source_Station_Name').size().reset_index(name='Train_Count')
trains_per_source = trains_per_source.sort_values('Train_Count', ascending=False)
print("\nTop 10 source stations:")
print(trains_per_source.head(10))

trains_per_day = df.groupby('days').size().reset_index(name='Train_Count')
print("\nTrains per day:")
print(trains_per_day)

# ── Task 2.3: Data Enrichment ───────────────────
weekdays = ['Monday','Tuesday','Wednesday','Thursday','Friday']
weekends = ['Saturday','Sunday']

def categorize_day(day):
    day = str(day).strip()
    if day in weekends:
        return 'Weekend'
    elif day in weekdays:
        return 'Weekday'
    elif day.lower() == 'daily':
        return 'Daily'
    elif 'sat' in day.lower() or 'sun' in day.lower():
        return 'Weekend'
    elif any(d in day for d in ['Mon','Tue','Wed','Thu','Fri']):
        return 'Weekday'
    else:
        return 'Multi-Day'

df['Day_Category'] = df['days'].apply(categorize_day)
print("Unknown days:", df[df['Day_Category']=='Multi-Day']['days'].unique())
print("\nDay category counts:")
print(df['Day_Category'].value_counts())

import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create outputs folder
os.makedirs('outputs', exist_ok=True)

# Chart 1 — Trains per day
plt.figure(figsize=(10,5))
day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
day_counts = df['days'].value_counts().reindex(day_order, fill_value=0)
sns.barplot(x=day_counts.index, y=day_counts.values,hue=day_counts.index, palette='Blues_d',legend=False)
plt.title('Number of Trains per Day of Week')
plt.xlabel('Day')
plt.ylabel('Number of Trains')
plt.tight_layout()
plt.savefig('outputs/trains_per_day.png')
plt.show()
print("✅ Chart 1 saved")

# Chart 2 — Top 10 source stations
plt.figure(figsize=(12,5))
top_sources = df['Source_Station_Name'].value_counts().head(10)
sns.barplot(x=top_sources.values, y=top_sources.index,hue=top_sources.index, palette='Oranges_r',legend=False)
plt.title('Top 10 Source Stations')
plt.xlabel('Number of Trains')
plt.tight_layout()
plt.savefig('outputs/top_source_stations.png')
plt.show()
print("✅ Chart 2 saved")

# Chart 3 — updated
plt.figure(figsize=(6,6))
df['Day_Category'].value_counts().plot.pie(
    autopct='%1.1f%%',
    colors=['#4CAF50','#FF9800','#2196F3','#9C27B0']
)
plt.title('Train Day Category Distribution')
plt.ylabel('')
plt.savefig('outputs/day_category_pie.png')
plt.show()
print("✅ Chart 3 saved")


# ── Task 3.1: Pattern Analysis ──────────────
print("\n=== PHASE 3: PATTERN ANALYSIS ===")

# Train distribution by day
day_order = ['Monday','Tuesday','Wednesday',
             'Thursday','Friday','Saturday','Sunday']
day_counts = df['days'].value_counts().reindex(
    day_order, fill_value=0)
print("\nTrain count by day:")
print(day_counts)

# Histogram of train frequency per station
plt.figure(figsize=(10,5))
trains_per_station = df.groupby('Source_Station_Name').size()
plt.hist(trains_per_station.values, bins=50,
         color='steelblue', edgecolor='white')
plt.title('Distribution of Train Frequency per Station')
plt.xlabel('Number of Trains')
plt.ylabel('Number of Stations')
plt.tight_layout()
plt.savefig('outputs/histogram_frequency.png')
plt.show()
print("✅ Histogram saved")

# Average trains per day per station
avg_trains = df.groupby(
    ['Source_Station_Name','days']).size().reset_index(name='Count')
avg_per_station = avg_trains.groupby(
    'Source_Station_Name')['Count'].mean().reset_index()
avg_per_station.columns = ['Source_Station_Name','Avg_Trains_Per_Day']
avg_per_station = avg_per_station.sort_values(
    'Avg_Trains_Per_Day', ascending=False)
print("\nTop 10 stations by avg trains per day:")
print(avg_per_station.head(10))

# ── Task 3.2: Correlation & Insights ────────
print("\n=== CORRELATION ANALYSIS ===")

day_num_map = {
    'Monday':1,'Tuesday':2,'Wednesday':3,
    'Thursday':4,'Friday':5,'Saturday':6,'Sunday':7
}
df['day_num'] = df['days'].map(day_num_map)
day_train_counts = df.groupby('day_num').size().reset_index(
    name='train_count')
correlation = day_train_counts['day_num'].corr(
    day_train_counts['train_count'])
print(f"Correlation between day number and train count: {correlation:.3f}")

if abs(correlation) < 0.3:
    print("→ Weak correlation: trains run consistently across all days")
elif abs(correlation) < 0.7:
    print("→ Moderate correlation found")
else:
    print("→ Strong correlation found")


# ── Task 4.1: Heatmap ───────────────────────
print("\n=== PHASE 4: VISUALIZATIONS ===")

# Heatmap — top 10 stations vs days
top10_stations = trains_per_source.head(10)[
    'Source_Station_Name'].tolist()
heatmap_data = df[df['Source_Station_Name'].isin(top10_stations)]
heatmap_pivot = heatmap_data.groupby(
    ['Source_Station_Name','days']).size().unstack(fill_value=0)
heatmap_pivot = heatmap_pivot.reindex(
    columns=[d for d in day_order if d in heatmap_pivot.columns])
heatmap_pivot.columns.name = None  # ← move to here BEFORE plot
plt.figure(figsize=(12,6))
sns.heatmap(heatmap_pivot, annot=True, fmt='d', cmap='Blues')
plt.title('Train Count Heatmap — Top 10 Stations by Day')
plt.xlabel('Days')
plt.ylabel('Station Name')
plt.tight_layout()
plt.savefig('outputs/heatmap_stations_days.png')
plt.show()

# Line chart — train trend across week
plt.figure(figsize=(10,5))
plt.plot(day_counts.index, day_counts.values,
         marker='o', color='#1677ff', linewidth=2.5,
         markersize=8)
plt.fill_between(day_counts.index, day_counts.values,
                 alpha=0.1, color='#1677ff')
plt.title('Train Volume Trend Across the Week')
plt.xlabel('Day of Week')
plt.ylabel('Number of Trains')
plt.tight_layout()
plt.savefig('outputs/line_trend_week.png')
plt.show()

print("\n🎉 ALL PHASES COMPLETE!")
print("outputs/ folder contains:")
print("  - trains_per_day.png")
print("  - top_source_stations.png")
print("  - day_category_pie.png")
print("  - histogram_frequency.png")
print("  - heatmap_stations_days.png")
print("  - line_trend_week.png")

print("\n" + "="*50)
print("   KEY INSIGHTS & RECOMMENDATIONS")
print("="*50)
print("📊 INSIGHTS:")
print("  1. CST-Mumbai is the busiest station with 513 trains")
print("  2. Friday has highest trains (1,649)")
print("  3. Thursday has lowest trains (1,526)")
print("  4. 71.2% trains run on weekdays vs 28.8% on weekends")
print("  5. Nagpur connects to 22 major cities")
print("  6. CST-Mumbai averages 73.28 trains per day")
print("  7. Moderate correlation (0.381) between day & train count")
print("  8. Most stations handle less than 50 trains")

print("\n💡 RECOMMENDATIONS:")
print("  1. Increase weekend trains to reduce 71% vs 29% gap")
print("  2. CST-Mumbai needs infrastructure upgrade")
print("  3. Thursday is best day for maintenance scheduling")
print("  4. Nagpur being central should have more connections")
print("  5. Smaller stations need better connectivity")
print("="*50)
print("✅ Analysis Complete — Pranav Thakare | Cognifyz")
print("="*50)

print("\n🎉 All charts saved in outputs/ folder!")