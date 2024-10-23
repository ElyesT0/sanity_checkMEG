from modules.parameters import *
from modules.functions_behavioral_analysis import *


nb_line_metadata=172
df=pd.read_csv(path_behavioral, skiprows=nb_line_metadata)

# Drop the "Performance" column
df.drop(labels="Performance",axis=1, inplace=True)

# Convert strings to arrays
clean_raw=str2int_dataset(df)


# -- Compute all the distance of Damerau-Levenshtein and add them to our dataframe
for i, row in clean_raw.iterrows():
    clean_raw.at[i, "distance_dl"] = dl_distance(row["PresentedSequence"], row["ProducedSequence"])

# -- Subject ID should be equal to the participant_nb
clean_raw['subject_id']=sub_nb

# ---------------------------------------
# ************* Token Error *************
# ---------------------------------------
# explanation: a response is a token error if it misses or adds to the base set of positions used to build 
# the stimuli

for i in range(len(clean_raw)):
    test=is_token_err(clean_raw.at[i,"PresentedSequence"], clean_raw.at[i,"ProducedSequence"])
    clean_raw.at[i,"TokenErr"]=test
    clean_raw.at[i,"TokenErr_forg"]=False
    clean_raw.at[i,"TokenErr_add"]=False
    if test:   
    # -- Adding (not in precedent experiment): token_forg and token_add
    # if True: token_forg => case of TokenErr where at least one of the tokens is in missing (has been forgotten)
    # if False: token_add => case of TokenErr where at least one of the tokens is missing
        test_forg=is_token_forg(clean_raw.at[i,"PresentedSequence"], clean_raw.at[i,"ProducedSequence"])
        clean_raw.at[i,"TokenErr_forg"]=test_forg
        clean_raw.at[i,"TokenErr_add"]=not test_forg


# Create a column in clean_raw that allows comparing different answers with the same temporal structure
column_holder=[]
for i in range(len(clean_raw)):
    try:
        column_holder.append(compare_tokens(clean_raw.iloc[i]["PresentedSequence"],clean_raw.iloc[i]["ProducedSequence"]))
    except TypeError:
        print(f'TypeError: ')
        print(f'clean_raw.iloc[{i}]["PresentedSequence"] = ',clean_raw.iloc[i]["PresentedSequence"])
        print(f'clean_raw.iloc[{i}]["ProducedSequence"] = ',clean_raw.iloc[i]["ProducedSequence"])
        
        
        
clean_raw["comparable_temp"]=column_holder

# Add a Performance column
column_holder=[]
for index,row in clean_raw.iterrows():
    if np.array_equal(row['PresentedSequence'],row['ProducedSequence']):
        performance='success'
    else:
         performance='fail'
         
    column_holder.append(performance)
clean_raw['performance']=column_holder

# Save data
clean_raw.to_csv(path_processed_behavioral_file)

# write the new path to the file_names.json file
xpd_fileName_dict['experiment']['processed'][f'{sub_nb}']=path_processed_behavioral_file
# -- then Save the modified JSON object back to the file
with open('modules/file_names.json', 'w') as file:
    json.dump(xpd_fileName_dict, file, indent=4)
