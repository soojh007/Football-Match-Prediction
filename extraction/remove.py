import pandas as pd

# Read the CSV file
df = pd.read_csv('merged_file.csv')

# Drop rows where all columns are NaN (empty rows)
df.dropna(how='all', inplace=True)

# Save the modified DataFrame back to CSV
df.to_csv('merged_file.csv', index=False)