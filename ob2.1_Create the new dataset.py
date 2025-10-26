import pandas as pd

### DATASET 1


# Load the uploaded file to check its contents
file_path = 'cleaned_retractions.csv'
data = pd.read_csv(file_path)

# Display the first few rows and the columns to understand its structure
data.head(), data.columns
from datetime import datetime

# Function to calculate the number of days from publishing to retracting
def calculate_days(original_date, retraction_date):
    try:
        original_date = datetime.strptime(original_date, "%Y-%m-%d")
        retraction_date = datetime.strptime(retraction_date, "%Y-%m-%d")
        return (retraction_date - original_date).days
    except:
        return None


# Filter by subject field and country
filtered_data = data[
    data['Subject'].str.contains('Biology|Genetics|Molecular Biology', case=False) &
    data['Country'].str.contains('China', case=False)
]

# Select and rename columns
final_data = filtered_data[[
    'Title', 'OriginalPaperDate', 'Author', 'Journal', 'Publisher', 'CitationCount', 'RetractionDate', 'Paywalled'
]].copy()

# Rename columns
final_data.columns = ['title', 'year', 'author', 'journal', 'publisher', 'citation', 'retraction_date', 'open_access']

# Calculate retraction time in days
final_data['retraction_time_days'] = final_data.apply(lambda x: calculate_days(x['year'], x['retraction_date']), axis=1)

# Convert 'Paywalled' to binary for open access
final_data['open_access'] = final_data['open_access'].apply(lambda x: 1 if x.lower() == 'no' else 0)

# Add constant column for retraction
final_data['retraction'] = 1

# Delete rows with negative retraction time days
final_data = final_data[final_data['retraction_time_days'] >= 0]

# Reorder columns to match the user's specification and fill missing data
final_data = final_data[[
    'title', 'year', 'author', 'journal', 'publisher', 'citation', 'retraction', 'retraction_time_days', 'open_access'
]]
final_data.head()

# Save the final formatted data to a new CSV file
output_file_path = 'dataset1.csv'
final_data.to_csv(output_file_path, index=False)


### DATASET 2

# Load the newly uploaded file to check its contents
new_file_path = 'scopus.csv'
new_data = pd.read_csv(new_file_path)

# Display the first few rows and the columns to understand its structure
new_data.head(), new_data.columns

from datetime import date

# Current year for calculating the retraction time
current_year = date.today().year

# Select necessary columns and rename them
reformatted_data = new_data[[
    'Title', 'Year', 'Authors', 'Source title', 'Publisher', 'Cited by', 'Open Access'
]].copy()

# Rename columns
reformatted_data.columns = ['title', 'year', 'author', 'journal', 'publisher', 'citation', 'open_access']

# Calculate retraction time in years
reformatted_data['retraction_time_days'] = (current_year - reformatted_data['year']) * 365

# Set retraction to 0 for all rows
reformatted_data['retraction'] = 0

# Convert 'Open Access' to binary for open access, assuming 'All Open Access' indicates free access
reformatted_data['open_access'] = reformatted_data['open_access'].apply(lambda x: 1 if 'All Open Access' in str(x) else 0)

# Reorder columns to match the user's specification
reformatted_data = reformatted_data[[
    'title', 'year', 'author', 'journal', 'publisher', 'citation', 'retraction', 'retraction_time_days', 'open_access'
]]

reformatted_data.head()

# Save the reformatted data to a new CSV file
reformatted_output_file_path = 'scopus1.csv'
reformatted_data.to_csv(reformatted_output_file_path, index=False)


# Check DUPLICATION between dataset1 and scopus1 then form dataset2

scopus1_path = 'scopus1.csv'
dataset1_path = 'dataset1.csv'

scopus1_data = pd.read_csv(scopus1_path)
dataset1_data = pd.read_csv(dataset1_path)
# Convert titles to lowercase for case-insensitive comparison
scopus1_data['title'] = scopus1_data['title'].str.lower()
dataset1_data['title'] = dataset1_data['title'].str.lower()

# Find titles in Scopus1 that are also in Dataset1
dataset1_titles = set(dataset1_data['title'])
scopus1_data_filtered = scopus1_data[~scopus1_data['title'].isin(dataset1_titles)]

# Show how many entries were removed
entries_before_scopus1 = scopus1_data.shape[0]
entries_after_scopus1 = scopus1_data_filtered.shape[0]
entries_removed_scopus1 = entries_before_scopus1 - entries_after_scopus1

entries_before_scopus1, entries_after_scopus1, entries_removed_scopus1
# Save the cleaned Scopus1 data to a new CSV file named dataset2.csv
dataset2_file_path = 'dataset2.csv'
scopus1_data_filtered.to_csv(dataset2_file_path, index=False)

## Merge dataset 1 and dataset 2

# Load the files for merging
dataset1 = 'dataset1.csv'
dataset2 = 'dataset2.csv'

dataset1_merge = pd.read_csv(dataset1)
dataset2_merge = pd.read_csv(dataset2)

# Normalize the name of author, publisher, journal to lower case
for column in ['author', 'publisher', 'journal']:
    if column in dataset1_merge.columns:
        dataset1_merge[column] = dataset1_merge[column].str.lower()
    if column in dataset2_merge.columns:
        dataset2_merge[column] = dataset2_merge[column].str.lower()

# Merge dataset1 and dataset2
merged_dataset = pd.concat([dataset1_merge, dataset2_merge], ignore_index=True)

# Save the merged dataset to a new CSV file
merged_dataset_file_path = 'merged_dataset.csv'
merged_dataset.to_csv(merged_dataset_file_path, index=False)



# Create CSV files: Author – Retraction:A list of all authors and the count of their retractions; Publisher – Retraction: A list of all publishers and the count of their retractions; Journal – Retraction: A list of all journal names and the count of their retractions.
file_path = 'cleaned_retractions.csv'
data = pd.read_csv(file_path)
# Normalize the names to lowercase
data['Author'] = data['Author'].str.lower()
data['Publisher'] = data['Publisher'].str.lower()
data['Journal'] = data['Journal'].str.lower()

# Some authors might be listed together in a single entry; we need to split them first
# Split authors by ';' and then explode the dataframe to have one author per row
data['Author'] = data['Author'].str.split(';')
author_retraction_counts = data.explode('Author')['Author'].value_counts().reset_index()
author_retraction_counts.columns = ['Author', 'Retraction Count']

# Now for publisher and journal retraction counts
publisher_retraction_counts = data['Publisher'].value_counts().reset_index()
publisher_retraction_counts.columns = ['Publisher', 'Retraction Count']

journal_retraction_counts = data['Journal'].value_counts().reset_index()
journal_retraction_counts.columns = ['Journal', 'Retraction Count']

author_retraction_counts.head(), publisher_retraction_counts.head(), journal_retraction_counts.head()

# Save the retraction counts into separate CSV files
author_retraction_counts_file = 'author_retraction_counts.csv'
publisher_retraction_counts_file = 'publisher_retraction_counts.csv'
journal_retraction_counts_file = 'journal_retraction_counts.csv'

author_retraction_counts.to_csv(author_retraction_counts_file, index=False)
publisher_retraction_counts.to_csv(publisher_retraction_counts_file, index=False)
journal_retraction_counts.to_csv(journal_retraction_counts_file, index=False)

(author_retraction_counts_file, publisher_retraction_counts_file, journal_retraction_counts_file)


## CREATE NEW COLUMN IN MERGE DATASET

# Load the necessary files
merged_dataset_path = 'merged_dataset.csv'
author_counts_path = 'author_retraction_counts.csv'
publisher_counts_path = 'publisher_retraction_counts.csv'
journal_counts_path = 'journal_retraction_counts.csv'

merged_dataset = pd.read_csv(merged_dataset_path)
author_counts = pd.read_csv(author_counts_path)
publisher_counts = pd.read_csv(publisher_counts_path)
journal_counts = pd.read_csv(journal_counts_path)

# Preview the loaded data
merged_dataset.head(), author_counts.head(), publisher_counts.head(), journal_counts.head()


# Split authors and count them
merged_dataset['Number of Authors'] = merged_dataset['author'].str.split(';').apply(len)

# Explode the authors for mapping with retraction counts
merged_dataset['individual_authors'] = merged_dataset['author'].str.split(';')
exploded_authors = merged_dataset.explode('individual_authors')


# Now merge the publisher and journal retraction counts
publisher_retraction_map = publisher_counts.set_index('Publisher')['Retraction Count'].to_dict()
journal_retraction_map = journal_counts.set_index('Journal')['Retraction Count'].to_dict()

merged_dataset['Retractions by Publisher'] = merged_dataset['publisher'].map(publisher_retraction_map)
merged_dataset['Retractions by Journal'] = merged_dataset['journal'].map(journal_retraction_map)

# Preview the updated dataset
merged_dataset[['title', 'author', 'Number of Authors', 'publisher', 'Retractions by Publisher', 'journal', 'Retractions by Journal']].head()

# Save the updated merged dataset
updated_merged_dataset_path = 'merged_dataset_1.csv'
merged_dataset.to_csv(updated_merged_dataset_path, index=False)

updated_merged_dataset_path


# Create column:​ SJR


scimagojr_path = 'scimagojr 2023.csv'
# Attempt to load the Scimago Journal Rankings (SJR) data with a semi-colon delimiter
try:
    scimagojr_data = pd.read_csv(scimagojr_path, delimiter=';', error_bad_lines=False)
except Exception as e:
    # If semi-colon doesn't work, attempt to read the file with more general assumptions
    scimagojr_data = pd.read_csv(scimagojr_path, delimiter=';', error_bad_lines=False, quoting=3)

# Display the first few rows and the columns to understand its structure
scimagojr_data.head(), scimagojr_data.columns

# Load the datasets
merged_dataset = pd.read_csv('merged_dataset_1.csv')

# Prepare the SJR data by stripping any potential leading/trailing spaces in the Title and converting to lowercase for a case-insensitive merge
scimagojr_data['Title'] = scimagojr_data['Title'].str.strip().str.lower()
merged_dataset['journal'] = merged_dataset['journal'].str.strip().str.lower()

# Merge the SJR values into the merged dataset based on the journal title
merged_dataset = merged_dataset.merge(scimagojr_data[['Title', 'SJR']], left_on='journal', right_on='Title', how='left')


# Show the first few rows of the updated dataset to verify the merge and new column
merged_dataset[['journal', 'SJR']].head()

# Fill NaN values in the SJR column with '0'
merged_dataset['SJR'] = merged_dataset['SJR'].fillna('0')

# Export the updated dataset to a new CSV file
output_file_path = 'merged_dataset_2.csv'
merged_dataset.to_csv(output_file_path, index=False)



## Create  Journal Discontinued (value 1 for the journal that is discontinued)

# Load the CSV file to check its contents and structure
file_path = 'scimagojr 2023.csv'
# Attempting to load the CSV file with a different delimiter or error handling to avoid parsing issues
data = pd.read_csv(file_path, on_bad_lines='skip')
data.head()
# Attempt to properly parse the CSV file by specifying the correct delimiter and reloading the data
data = pd.read_csv(file_path, delimiter=';', on_bad_lines='skip')
data.head()

# Filter rows where the title contains "(discontinued)"
discontinued_journals = data[data['Title'].str.contains("\(discontinued\)")]

# Remove the "(discontinued)" part from the titles
discontinued_journals['Title'] = discontinued_journals['Title'].str.replace(" \(discontinued\)", "", regex=True)

# Normalize all names to lowercase and remove content inside parentheses
discontinued_journals ['Title'] = discontinued_journals ['Title'].str.lower().str.replace(r"\(.*?\)", "", regex=True).str.strip()


# Extract the cleaned titles to a new CSV file
output_file_path = 'discontinued_journals.csv'
discontinued_journals[['Title']].to_csv(output_file_path, index=False, header=False)

# Show the first few rows of the cleaned titles to verify
discontinued_journals[['Title']].head()


merged_data = pd.read_csv('merged_dataset_2.csv')
discontinued_journals = pd.read_csv('discontinued_journals.csv')
# Renaming the column in discontinued_journals to match 'journal' for clarity
discontinued_journals.columns = ['journal']

# Checking for any journal names in the 'merged_data' that match or contain names in the 'discontinued_journals' list
merged_data['Discontinue journal'] = merged_data['journal'].apply(lambda x: 1 if any(discontinued_journals['journal'].str.lower().str.contains(x.lower())) else 0)

# Save the updated dataframe to a new CSV file
updated_file_path = 'merged_dataset_3.csv'
merged_data.to_csv(updated_file_path, index=False)


## Handle the problem of Chinese author's name


### Check Author retraction count for non retracted author 
from collections import Counter

# Load the dataset
file_path = 'merged_dataset_3.csv'
data = pd.read_csv(file_path)

# Adjusting the function to handle missing values and ensure all entries are treated as strings
def process_data(data_subset):
    # Extract relevant details
    details = data_subset[['author', 'publisher', 'journal']].fillna('Unknown')
    
    # Creating a dictionary to hold the results
    processed_data = {}
    
    # Loop through each row in the subset
    for _, row in details.iterrows():
        authors_list = row['author'].split(';')
        for author in authors_list:
            if author not in processed_data:
                # Initialize dictionary for new author
                processed_data[author] = {
                    'publisher': set(),
                    'journal': set(),
                    'collaborators': []
                }
            
            # Add publisher and journal
            processed_data[author]['publisher'].add(str(row['publisher']))
            processed_data[author]['journal'].add(str(row['journal']))
            # Add all other authors as collaborators
            processed_data[author]['collaborators'].extend([a for a in authors_list if a != author])
    
    # Convert sets to single string and find the most frequent collaborator
    for author, info in processed_data.items():
        info['publisher'] = ', '.join(info['publisher'])
        info['journal'] = ', '.join(info['journal'])
        if info['collaborators']:
            info['most_frequent_collaborator'] = Counter(info['collaborators']).most_common(1)[0][0]
        else:
            info['most_frequent_collaborator'] = 'None'
        
        # Cleanup the structure
        del info['collaborators']
    
    return processed_data
# Splitting the dataset into retracted and non-retracted
retracted_data = data[data['retraction'] == 1]
non_retracted_data = data[data['retraction'] == 0]

# Process each subset
retracted_authors_details = process_data(retracted_data)
non_retracted_authors_details = process_data(non_retracted_data)


# Convert processed data into DataFrame for saving to CSV
df_retracted = pd.DataFrame.from_dict(retracted_authors_details, orient='index').reset_index().rename(columns={'index': 'author'})
df_non_retracted = pd.DataFrame.from_dict(non_retracted_authors_details, orient='index').reset_index().rename(columns={'index': 'author'})

df_retracted.head(), df_non_retracted.head()

# File paths for the output CSV files
retracted_file_path = 'retracted_authors_details.csv'
non_retracted_file_path = 'non_retracted_authors_details.csv'

# Save to CSV
df_retracted.to_csv(retracted_file_path, index=False)
df_non_retracted.to_csv(non_retracted_file_path, index=False)



# Load the recently uploaded dataset of retracted authors details
retracted_authors_path = 'retracted_authors_details.csv'
retracted_authors_data = pd.read_csv(retracted_authors_path)

# Display the first few rows of the author names to inspect the formatting
retracted_authors_data['author'].head(10)


def normalize_author_name(name):
    parts = name.split()
    if any('.' in part or len(part) == 1 for part in parts):
        # Move initials to the end if they are not
        parts = [p for p in parts if '.' not in p and len(p) > 1] + [p for p in parts if '.' in p or len(p) == 1]
    if len(parts) == 1:
        return parts[0].lower()  # Handle single-part names conservatively
    # Convert to last name + initials format
    last_name = parts[-1]
    initials = ''.join(part[0] + '.' for part in parts[:-1])
    normalized_name = f"{last_name} {initials}".strip()
    return normalized_name.lower()

# Apply normalization
retracted_authors_data['normalized_author_name'] = retracted_authors_data['author'].apply(normalize_author_name)
retracted_authors_data['most_frequent_collaborator']=retracted_authors_data['most_frequent_collaborator'].apply(normalize_author_name)
# Save the updated dataset with the normalized author names to a new CSV file
normalized_authors_file_path = 'normalized_retracted_authors_details.csv'
retracted_authors_data.to_csv(normalized_authors_file_path, index=False)

# Show the first few entries to verify the normalization
retracted_authors_data[['author', 'normalized_author_name']].head(10), normalized_authors_file_path

## Add the author retraction count to the file 
# Load the provided CSV files
normalized_authors = pd.read_csv('normalized_retracted_authors_details.csv')
retraction_counts = pd.read_csv('author_retraction_counts.csv')


# Merge the two DataFrames based on the author names
merged_data = normalized_authors.merge(retraction_counts, left_on="author", right_on="Author", how="left")

# Rename the 'Retraction Count' column to 'author_retraction_count' and drop the redundant 'Author' column
merged_data = merged_data.rename(columns={"Retraction Count": "author_retraction_count"}).drop(columns=["Author"])

# Replace NaN values in 'author_retraction_count' with 0 (assuming no match means no retractions)
merged_data['author_retraction_count'] = merged_data['author_retraction_count'].fillna(0)

# Display the updated DataFrame
merged_data.head()

# Save the updated DataFrame to a new CSV file
output_file_path = 'normalized_retracted_authors_details_1.csv'
merged_data.to_csv(output_file_path, index=False)

# Load the CSV file
file_path = 'non_retracted_authors_details.csv'
authors_df = pd.read_csv(file_path)

import re

# Function to standardize author names
def standardize_author_name(name):
    # Remove unnecessary spaces
    name = name.strip()
    # Check if the name is already in the correct format (last name first, followed by initials)
    if re.match(r'^[a-z]+\s+[a-z]\.?[a-z]?\.?$', name):
        return name
    # If not, attempt to reformat the name
    parts = name.split()
    if len(parts) > 1:
        last_name = parts[0]
        initials = ''.join([part[0].lower() + '.' for part in parts[1:]])
        return f'{last_name} {initials}'
    return name

# Apply the standardization function to the 'author' column
authors_df['author'] = authors_df['author'].apply(standardize_author_name)
authors_df['most_frequent_collaborator'] = authors_df['most_frequent_collaborator'].str.lstrip().str.replace('-', '')
# Save the cleaned DataFrame to a new CSV file
cleaned_file_path = 'non_retracted_authors_details_1.csv'
authors_df.to_csv(cleaned_file_path, index=False)

# Add author retraction count to non retracted authors file
# Load the datasets
non_retracted_df = pd.read_csv('non_retracted_authors_details_1.csv')
normalized_retracted_df = pd.read_csv('normalized_retracted_authors_details_1.csv')

# Define a function to find the retraction count based on multiple criteria
def find_retraction_count(row, normalized_df):
    # Find matches based on the author name
    matches = normalized_df[normalized_df['normalized_author_name'] == row['author']]
    
    if len(matches) == 1:
        return matches['author_retraction_count'].values[0]
    
    if len(matches) > 1:
        # If multiple matches, filter by journal
        journal_matches = matches[matches['journal'].apply(lambda x: row['journal'] in x)]
        if len(journal_matches) == 1:
            return journal_matches['author_retraction_count'].values[0]
        
        if len(journal_matches) > 1:
            # If still multiple matches, filter by most_frequent_collaborator
            collaborator_matches = journal_matches[journal_matches['most_frequent_collaborator'] == row['most_frequent_collaborator']]
            if len(collaborator_matches) == 1:
                return collaborator_matches['author_retraction_count'].values[0]
            
            if len(collaborator_matches) > 1:
                return collaborator_matches['author_retraction_count'].mode()[0] # Return the most frequent count if still ambiguous

    return 0  # Return 0 if no match found

# Apply the function to each row in non_retracted_df
non_retracted_df['author_retraction_count'] = non_retracted_df.apply(find_retraction_count, axis=1, normalized_df=normalized_retracted_df)


# Save the updated dataframe
non_retracted_df.to_csv('non_retracted_authors_2.csv', index=False)

## Create the Total retraction by author
# Load the datasets
merged_df = pd.read_csv('merged_dataset_3.csv')
author_retraction_counts_df = pd.read_csv('author_retraction_counts.csv')
non_retracted_authors_df = pd.read_csv('non_retracted_authors_2.csv')

# Merge the datasets on the author field
merged_df['individual_authors'] = merged_df['individual_authors'].apply(lambda x: eval(x))  # Convert string representation of list to actual list
author_retraction_dict = dict(zip(author_retraction_counts_df['Author'], author_retraction_counts_df['Retraction Count']))

# Create a function to count retractions by individual authors
def count_retractions(authors, retraction_dict):
    return sum(retraction_dict.get(author.strip(), 0) for author in authors)

# Prepare the non_retracted_authors_df for merging
non_retracted_authors_df.rename(columns={'author': 'individual_authors'}, inplace=True)

# Update the non_retracted_authors_df to match the list format of individual_authors in merged_df
non_retracted_authors_df['individual_authors'] = non_retracted_authors_df['individual_authors'].apply(lambda x: [x])

# Create a combined retraction dictionary from both sources
combined_retraction_dict = author_retraction_dict.copy()
combined_retraction_dict.update(dict(zip(non_retracted_authors_df['individual_authors'].apply(lambda x: x[0]), 
                                         non_retracted_authors_df['author_retraction_count'])))

# Apply the function to the merged dataframe with the combined retraction dictionary
merged_df['Total Retractions by Authors'] = merged_df['individual_authors'].apply(lambda authors: count_retractions(authors, combined_retraction_dict))

# Save the updated dataframe to a new CSV file
output_file_path = 'merged_dataset_4.csv'
merged_df.to_csv(output_file_path, index=False)



## Clean and reformat file
# Load the newly uploaded dataset for further modifications
data = 'merged_dataset_4.csv'
data_to_modify = pd.read_csv(data, delimiter=',', on_bad_lines='skip')

# Delete the specified columns
columns_to_delete = ['author', 'journal', 'publisher', 'individual_authors', 'Title']
data_to_modify.drop(columns=columns_to_delete, inplace=True, errors='ignore')

# Reformat 'year' column to just show the year as a number
data_to_modify['year'] = pd.to_datetime(data_to_modify['year'], errors='coerce').dt.year

# Convert 'SJR' column to numeric, replacing non-numeric characters and handling errors
data_to_modify['SJR'] = pd.to_numeric(data_to_modify['SJR'].str.replace(',', '.', regex=False), errors='coerce')

# Standardize column names
data_to_modify.columns = ['Title', 'Year', 'Citation', 'Retraction', 'Retraction Time (days)','Open Access','Number of Authors','Number of Retractions by Publishers',  'Number of Retractions by Journal', 'SJR', 'Journal Discontinue','Total Retractions by Authors',]

# Fill blank values with 0
data_to_modify.fillna(0, inplace=True)

# Export the cleaned and reformatted dataframe to a new CSV file
final_output_path = 'final.csv'
data_to_modify.to_csv(final_output_path, index=False) 

