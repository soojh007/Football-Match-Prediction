
import pandas as pd
import glob

path = 'EPL\Dataset\*.csv' 

csv_files = glob.glob(path)

print(f"Found {len(csv_files)} CSV files: {csv_files}")  

dataframes = []

for file in csv_files:
    print(f"Processing file: {file}")
    try:
        df = pd.read_csv(file, encoding='utf-8')
    except UnicodeDecodeError:
        print(f"UnicodeDecodeError encountered. Trying 'latin1' encoding for file: {file}")
        df = pd.read_csv(file, encoding='latin1')
    df = df.iloc[:, :7]
    dataframes.append(df)

if not dataframes:
    print("No dataframes to concatenate. Exiting.")  
else:
    merged_df = pd.concat(dataframes, ignore_index=True)
    merged_df.to_csv('merged_file.csv', index=False)

# Find unique values of HomeTeam
unique_home_teams = merged_df['HomeTeam'].unique()

# Display the team names
print(', '.join(map(str, unique_home_teams)))
