from modules.parameters import *


def merge_csv_files(file_paths, output_path):
    """
    Merge multiple CSV files into a single DataFrame and save it to a new CSV file.
    
    Parameters:
        file_paths (list of str): List of paths to the CSV files to be merged.
        output_path (str): Path to the output CSV file.
    """
    # Initialize an empty list to store the DataFrames
    dataframes = []

    # Iterate over the file paths and read each CSV file into a DataFrame
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        dataframes.append(df)
        print(f'Added {file_path}')
    
    # Concatenate all DataFrames
    merged_df = pd.concat(dataframes, ignore_index=True)

    # Save the merged DataFrame to a new CSV file
    merged_df.to_csv(output_path, index=False)
    print(f'Merged file saved to {output_path}')
    
# ----------------------------------------------
def dl_distance(s1, s2):
    """
    Computes the Damerau-Levenshtein distance between two strings.

    The Damerau-Levenshtein distance is a metric for measuring the edit distance
    between two sequences. It is the minimum number of operations needed to transform
    one string into the other, where an operation is defined as an insertion, deletion,
    substitution, or transposition of two adjacent characters.

    Parameters:
    s1 (str): The first string.
    s2 (str): The second string.

    Returns:
    int: The Damerau-Levenshtein distance between the two strings.
    
    Example:
    >>> dl_distance("kitten", "sitting")
    3
    """
    
    d = {}
    lenstr1 = len(s1)
    lenstr2 = len(s2)
    for i in range(-1,lenstr1+1):
        d[(i,-1)] = i+1
    for j in range(-1,lenstr2+1):
        d[(-1,j)] = j+1

    for i in range(lenstr1):
        for j in range(lenstr2):
            if s1[i] == s2[j]:
                cost = 0
            else:
                cost = 1
            d[(i,j)] = min(
                           d[(i-1,j)] + 1, # deletion
                           d[(i,j-1)] + 1, # insertion
                           d[(i-1,j-1)] + cost, # substitution
                          )
            if i and j and s1[i]==s2[j-1] and s1[i-1] == s2[j]:
                d[(i,j)] = min (d[(i,j)], d[i-2,j-2] + cost) # transposition

    return int(d[lenstr1-1,lenstr2-1])


def str2int_dataset(df):
    clean_df = df.copy()
    
    clean_df['PresentedSequence'] = clean_df['PresentedSequence'].apply(eval)
    clean_df['ProducedSequence'] = clean_df['ProducedSequence'].apply(eval)
    clean_df['RTs'] = clean_df['RTs'].apply(eval)
    
    return clean_df

        
def is_token_err(origin, recall):
    #Test if there is a token error: a token not presented have been reproduced or a token presented was forgotten
    return set(origin)!=set(recall)

def is_token_forg(origin,recall):
    # Adding (not in precedent experiment):
    # if True: token_forg => case of TokenErr where at least one of the tokens is in missing (has been forgotten)
    # if False: token_add => case of TokenErr where at least one of the tokens is missing
    return set(recall).issubset(set(origin))

def compare_tokens(origin, recall):
    """a function that will compare two sequences, and return an absolut mapping of the 
    reproduction (1 for token 1, 2 for token 2, 3 for token 3, 4 for token 4, 
    -1 for wrong token)
    
     Exemple :

    compare_tokens([1,2,3,1,2,3],[5,3,2,5,3,2]) => out: [-1, 3, 2, -1, 3, 2]
    compare_tokens([0,2,5,0,2,5],[0,2,5,0,2,5]) => out: [1, 2, 3, 1, 2, 3]



    Args:
        origin (arr): sequence shown to the participant
        recall (arr): sequence recalled by the participant

    Returns:
        _type_: _description_
    """
    # Establish a mapping from token to original ordinal position
    try:
        origin=eval(origin)
        recall=eval(recall)
    except TypeError:
        pass
    
    # Establish a mapping from token to original ordinal position
    original_order=pd.unique(np.array(origin))
    mapping=dict(zip(original_order,range(1,len(original_order)+1)))
    # Return new mapping (comparable temp)
    return [mapping.get(x, -1) for x in recall]


#--------------------------------------------------------------------------------------------

def confidence_interval95(arr):
    """
    Calculate the 95% confidence interval for a given array of values.

    Args:
        arr (numpy array or list): Array or list of values.

    Returns:
        tuple: Lower and upper bounds of the 95% confidence interval.
    """
    # Calculate standard deviation of the array of values
    sd = np.std(arr, ddof=1)  # Use ddof=1 to get the sample standard deviation

    # Calculate standard error of the mean: SEM = sd / sqrt(n)
    sem = sd / np.sqrt(len(arr))

    # Using a t-table, find the t-score for the given degrees of freedom and the chosen confidence interval
    confidence_level = 0.95  # 95% confidence level
    df = len(arr) - 1  # degrees of freedom
    t_score = stats.t.ppf((1 + confidence_level) / 2, df)

    # Calculate the Margin of Error: MOE = t_score * SEM
    moe = t_score * sem

    # Calculate the confidence interval: CI(95) = Mean ± MOE
    ci = (np.mean(arr) - moe, np.mean(arr) + moe)
    
    return ci

#--------------------------------------------------------------------------------------------


# ******************************************************************
# PLOTTING FUNCTIONS
# ******************************************************************

def plot_mean_dl(data):
    available_seq_name_list=[]
    for name in seq_name_list:
        if name in data['sequenceName'].unique():
            available_seq_name_list.append(name)
            
            
    # Participants IDs
    IDs=[data["subject_id"][0]]
    for i in range(len(data)-1):
        if data["subject_id"][i] not in IDs:
            IDs.append(data["subject_id"][i])
    
    # Calculate the mean distance_DL for each sequence per participant
    temp_distDL_perParticipant = []

    for name in available_seq_name_list:
        new_arr = []
        for participant in IDs:
            subset = data[(data["subject_id"] == participant) & (data["sequenceName"] == name)]
            mean_distance_dl = np.nanmean(subset["distance_dl"])  # Use np.nanmean to handle NaN values
            new_arr.append(mean_distance_dl)
        temp_distDL_perParticipant.append(new_arr)

    # Convert the list of lists into a 2D NumPy array
    distDL_perParticipant = np.array(temp_distDL_perParticipant)

    # Calculate confidence intervals
    CI_meanDL = [confidence_interval95(dist) for dist in distDL_perParticipant]
    all_sem = [stats.sem(dist, nan_policy='omit') for dist in distDL_perParticipant]

    # Extract the lower and upper bounds of the confidence interval
    lower_bound = np.array([item[0] for item in CI_meanDL])
    upper_bound = np.array([item[1] for item in CI_meanDL])

    # Plotting
    plt.rcParams["figure.facecolor"] = "white"

    fig, ax = plt.subplots(figsize=plot_figsize)
    ax.barh(np.arange(len(available_seq_name_list)), np.nanmean(distDL_perParticipant, axis=1),
            xerr=all_sem, capsize=5, align="center", color=plot_colors)
    
    ax.set_yticks(np.arange(len(available_seq_name_list)))
    ax.set_yticklabels(available_seq_name_list, fontsize=14)
    ax.invert_yaxis()


    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    plt.title("Mean Distance DL - All Sequences", fontsize=title_size, pad=padding_size)
    plt.show()


def plot_error_rate(data):
    available_seq_name_list=[]
    for name in seq_name_list:
        if name in data['sequenceName'].unique():
            available_seq_name_list.append(name)
            
    # == step1 == Create y axis: a list with rate of success per sequence
    success_rate=[]
    plt.rcParams['figure.facecolor'] = 'white'

    # New name_list with count of included sequences for each seq type in data_main (all main)

    error_rates_all=[]


    for i in range(len(available_seq_name_list)):
        nb_success=len(data[(data["sequenceName"]==available_seq_name_list[i])&(data["performance"]=="success")])
        nb_total=len(data[data["sequenceName"]==available_seq_name_list[i]])
        success_rate.append(100*nb_success/nb_total)
        error_rates_all.append(100-success_rate[i])

    colors = plot_colors

    plt.rcParams['figure.facecolor'] = 'white'
    fig,ax=plt.subplots(figsize=plot_figsize)
    ax.barh(np.arange(len(available_seq_name_list)),error_rates_all, align="center", color=colors)
    ax.set_yticks(range(len(available_seq_name_list)))
    ax.set_yticklabels(available_seq_name_list)
    ax.invert_yaxis()
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

    ax.set_xlim(0,100)
    ax.set_xlabel("Error Rate (%)", fontsize=14, labelpad=14)
    ax.set_title("Total Error Rate", fontsize=title_size, pad=padding_size)
    plt.show()



#--------------------------------------------------
def plot_token_error(data):
    available_seq_name_list=[]
    for name in seq_name_list:
        if name in data['sequenceName'].unique():
            available_seq_name_list.append(name)
            
    all_token_err=[]
    for name in available_seq_name_list:
        all_token_err.append(np.sum(data[data["sequenceName"]==name]["TokenErr"]))

    fig, ax=plt.subplots(figsize=plot_figsize)
    ax.set_yticks(range(len(available_seq_name_list)))
    ax.set_yticklabels(available_seq_name_list)
    ax.set_xticks(range(100))
    ax.invert_yaxis()
    ax.barh(range(len(available_seq_name_list)),all_token_err)
    plt.title('Token Error (absolute number)', fontsize=title_size,pad=padding_size)
    plt.show()



# ******************************************************************
# PIPELINE FUNCTIONS
# ******************************************************************


def run_preliminary_behavioral_pipeline(data_path=path_processed_behavioral_file):
    data=pd.read_csv(data_path)
    # plot DL per sequence
    plot_mean_dl(data)
    # plot Error rate per sequence
    plot_error_rate(data)
    # Plot token error per sequence
    plot_token_error(data)