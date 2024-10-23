"""
Module Name: 
    Merge Participants Behavioral

Description:
    Take all the behavioral processed data files and merge them into one bigger csv file.

Author:
    Elyès Tabbane

Date:
    {V.0: 08.07.2024}

Version:
    V0
"""
from modules.parameters import *
from modules.functions_behavioral_analysis import *

# Open the JSON file with path to processed data and raw data
with open('modules/file_names.json', 'r') as file:
    json_file_names=json.load(file)
  
# Get the Processed path for all subjects 
processed_file_names=json_file_names['processed']

# Turn the dictionnary of paths into a list
list_processed_file_names=[path for path in processed_file_names.values()]

# Define the output path
merge_csv_files(list_processed_file_names,path_save_merged_behavioral_file)
print('\nMERGING DONE')

