"""
Module Name: 
    Functions sanity check

Description:
    Provide a set of functions to test the coherence of MEG data.

Author:
    Elyès Tabbane

Date:
    {V.0: 02.07.2024}

Version:
    V0

"""


from modules.parameters import *


from IPython.display import display_markdown
# ******************************************************************
# SANITY CHECK RELATED FUNCTIONS
# ******************************************************************

def get_filenames_with_run(directory):
    # List to store filenames containing "run" and ending with ".fif"
    filenames_with_run = []
    
    # Iterate over all files in the specified directory
    for filename in os.listdir(directory):
        # Check if "run" is in the filename and it ends with ".fif"
        if "run" in filename and filename.endswith(".fif"):
            filenames_with_run.append(filename)
    
    return filenames_with_run

def partition_array(events_presentation):
    filtered_events = [evt for evt in events_presentation if evt[2] not in [4, 5, 60, 61]]
    
    # Initialize variables
    partitioned_arrays = []
    current_array = []

    # Iterate through filtered_events and partition into groups of 12
    for i, item in enumerate(filtered_events):
        current_array.append(item)
        if len(current_array) == 12:
            partitioned_arrays.append(current_array)
            current_array = []

    # Append the last partition if it contains any events
    if current_array:
        partitioned_arrays.append(current_array)
    
    return partitioned_arrays





def get_event_presentation(path_raw, path_behavior, expected_nb_trials=15):
    """
    Processes MEG data and behavioral data to extract and compare event sequences.

    This function performs the following tasks:
    1. Loads and processes raw MEG data files to identify event sequences.
    2. Reads and processes behavioral data files.
    3. Compares the event sequences identified in MEG data with those in behavioral data.
    4. Excludes specific events (e.g., fixation cross, fixation blue) from the analysis.
    5. Outputs a summary of whether the event sequences in MEG data match those in the behavioral data.

    Parameters:
    - path_raw (str): Path to the directory containing raw MEG data files.
    - path_behavior (str): Path to the behavioral data file.
    - expected_nb_trials (int, optional): Expected number of trials per block. Default is 15.

    Returns:
    - None: This function prints the results directly and does not return any value.

    Note:
    - The function assumes specific metadata structure and event codes.
    - The function modifies logging levels and suppresses specific warnings during execution.
    - Requires `mne`, `pandas`, `numpy`, and other standard Python libraries.

    """
    logging.getLogger('mne').setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message="This file contains raw Internal Active Shielding data", category=RuntimeWarning)

    # Prints the number of presentation events (exclude fixation cross and fixation blue)
    # Prints the number of items in sequences in one run in behavioral file
    
    ###################### Behavioral FILE
    # -- To read an XPD file you can read it as CSV but need to skip all the metadata lines
    # Load behavioral file
    nb_line_metadata=172
    df=pd.read_csv(path_behavior, skiprows=nb_line_metadata)

    
    ###################### Create EVENT File
    # Return a big numpy array of shape (x,y,z) with
    # x = number of runs in the experiment
    # y = number of events in one run
    # z = 3 (time of event, port code, event code)
    
    # List raw filenames
    raw_filenames=get_filenames_with_run(path_raw)
    
    # Holder for all the arrays of all runs' events
    all_events_presentation=np.array([])
    
    # Holder for booleans resulting of the test between MEG identified presented sequence and Behavioral identified presented sequence
    same=[]
    
    # Holder for booleans resulting of the test for right number of trials per block (expected_nb_trials)
    full_block=[]
    
    for i in range(len(raw_filenames)):
        
        # Load MEG data
        raw=mne.io.read_raw_fif(os.path.join(path_raw,raw_filenames[i]), preload=True, allow_maxshield=True, verbose=False)
        
        # Find events
        events_presentation=mne.find_events(raw,mask_type = "not_and",mask = 2**6+2**7+2**8+2**9+2**10+2**11+2**12+2**13+2**14+2**15, verbose=False, min_duration=0.1)
        
        # Exclude events with codes 4 and 5 (fixation cross and fixation blue as well as win or loss)
        filtered_events = [evt for evt in events_presentation[:,2] if evt not in [4, 5, 60, 61]]
        all_events_presentation=np.append(all_events_presentation, filtered_events)
        
        # Free up memory space
        del(raw)
        
        # Unique list of events' codes
        unique_events=np.unique(filtered_events)
        
        # Number of Trials per block
        partitioned_arrays=partition_array(events_presentation)
        nb_trials=len(partitioned_arrays)
                
        # Unique list of events' names
        translated_unique=[reverse_event_dict[num] for num in unique_events]
        
        # Behavioral data sequence name
        seqName=df[df['block']==(i)]['sequenceName'].unique()[0]
        
        # Test if MEG and Behavioral point to the same presented sequence
        same.append(all(item.startswith(seqName + '-') for item in translated_unique))
        
        # Test if there's the right number of trials per block
        full_block.append(nb_trials==expected_nb_trials)
        
        if nb_trials==expected_nb_trials:
            display_markdown(f'**{raw_filenames[i]}**. Number of trials: OK', raw=True)
            # display_markdown(f'Events found for *{raw_filenames[i]}*: **{unique_events}**', raw=True)
            # display_markdown(f' Translates to **{translated_unique}**', raw=True)
            # display_markdown(f" Behavioral file states that it correspond to : **{seqName}**", raw=True)
            # display_markdown(f'there are **{nb_trials}** trials in this block', raw=True)
            # display_markdown('-----------------\n', raw=True)
        else:
            display_markdown(f'<span style="color:red"> Please check **{raw_filenames[i]}** </span>', raw=True)
            display_markdown(f'there are **{nb_trials}** trials in this block. Expected {expected_nb_trials}', raw=True)
            check_wrong_nb_trial(i+1)
        
        if same[-1]:
            display_markdown(f'**{raw_filenames[i]}**. Matching Raw Events and Behavioral file: OK', raw=True)
        else:
            display_markdown(f'<span style="color:red"> Please check **{raw_filenames[i]}**. Events in Raw and in Behavioral do NOT match </span>', raw=True)
        
        display_markdown('-----------------\n', raw=True)
        
        
            
    
    all_same=all(same)
    all_full_block=all(full_block)
    summary=[]
    
    if all_same:
        summary.append('All presented sequence from MEG data correspond to presented sequence in the Behavioral file.')
    else:
        summary.append('**ERROR**: At least one of the sequence presented was different in the MEG data and the Behavioral file.')
        
    if all_full_block:
        summary.append(f'\nAll presented blocks have the expected number of trials ({expected_nb_trials})')
    else:
        summary.append(f'\n**ERROR**: At least one block does not have the right number of trials.')
        
            
    
    display_markdown(f"Summary: \n **{' '.join(summary)}**", raw=True)
    
    logging.getLogger('mne').setLevel(logging.NOTSET)
    warnings.resetwarnings()
    
#-------------------

def check_wrong_nb_trial(run_nb):

    # 1. Check the events in the raw file
    raw=mne.io.read_raw_fif(os.path.join(path_raw,f'run{run_nb:02}_raw.fif'),preload=True, allow_maxshield=True)
    events_presentation = mne.find_events(raw,mask_type = "not_and",mask = 2**6+2**7+2**8+2**9+2**10+2**11+2**12+2**13+2**14+2**15, verbose=False, min_duration=0.1)
    print(f'Number of presented events during block {run_nb:02}: ',len(events_presentation))
    del raw

    print('*** Checking the behavioral file***')
    df=pd.read_csv(path_processed_behavioral_file)
    print(f"There are {len(df[df['block']==run_nb])} trials in block {run_nb:02}")


#-------------------


def plot_presentation_SOA(path_raw,path_behavior):
    logging.getLogger('mne').setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message="This file contains raw Internal Active Shielding data", category=RuntimeWarning)
    
    ###################### Behavioral FILE
    # -- To read an XPD file you can read it as CSV but need to skip all the metadata lines
    # Load behavioral file
    nb_line_metadata=172
    df=pd.read_csv(path_behavior, skiprows=nb_line_metadata)

    
    ###################### Create EVENT File
    # Return a big numpy array of shape (x,y,z) with
    # x = number of runs in the experiment
    # y = number of events in one run
    # z = 3 (time of event, port code, event code)
    
    # List raw filenames
    raw_filenames=get_filenames_with_run(path_raw)
    
    # Holder for all the arrays of all runs' events
    all_events_presentation_timings=np.array([])
    
    all_run_soa=np.array([])

    
    for i in range(len(raw_filenames)):
        run_soa=[]
        
        # Load MEG data
        raw=mne.io.read_raw_fif(os.path.join(path_raw,raw_filenames[i]), preload=True, allow_maxshield=True, verbose=False)
        
        # Find events
        events_presentation=mne.find_events(raw,mask_type = "not_and",mask = 2**6+2**7+2**8+2**9+2**10+2**11+2**12+2**13+2**14+2**15, verbose=False, min_duration=0.1)
        
        # Exclude events with codes 4 and 5 (fixation cross and fixation blue as well as win or loss)
        filtered_events = [evt for evt in events_presentation if evt[2] not in [4, 5, 60, 61]]
        all_events_presentation_timings=np.append(all_events_presentation_timings, filtered_events)

        # Get all timings
        partitioned_events=partition_array(events_presentation)
        
        
        for arr in partitioned_events:
            for i in range(len(arr)-1):
                run_soa.append(arr[i+1][0]-arr[i][0])
    
        all_run_soa=np.append(all_run_soa,run_soa)
        del(raw)
        
    all_run_soa=all_run_soa.flatten()
    #Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(all_run_soa, edgecolor='black')
    plt.xlabel('SOA (ms)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Stimulus Onset Asynchrony (SOA) -- All Runs')
    plt.grid(True)
    plt.tight_layout()
    #plt.show()
    logging.getLogger('mne').setLevel(logging.NOTSET)
    warnings.resetwarnings()


def get_time_per_run(path_raw):
    logging.getLogger('mne').setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message="This file contains raw Internal Active Shielding data", category=RuntimeWarning)
     # List raw filenames
    raw_filenames=get_filenames_with_run(path_raw)

    all_durations=[]
    
    for i in range(len(raw_filenames)):
        
        # Load MEG data
        raw=mne.io.read_raw_fif(os.path.join(path_raw,raw_filenames[i]), preload=True, allow_maxshield=True, verbose=False)
        
        # Find events
        events_presentation=mne.find_events(raw,mask_type = "not_and",mask = 2**6+2**7+2**8+2**9+2**10+2**11+2**12+2**13+2**14+2**15, verbose=False, min_duration=0.1)
        
        # Exclude events with codes 4 and 5 (fixation cross and fixation blue as well as win or loss)
        filtered_events = [evt for evt in events_presentation if evt[2] not in [4, 5, 60, 61]]
        
        # Compute duration
        duration_run=(filtered_events[-1][0]-filtered_events[0][0])/1000
        
        display_markdown(f'Run {i+1} took: {duration_run/60:.2f} minutes', raw=True)
        
        all_durations.append(duration_run/60)
        
        del(raw)
        
        
    display_markdown(f' **Mean duration**: {np.mean(all_durations):.2f} minutes', raw=True)
    logging.getLogger('mne').setLevel(logging.NOTSET)
    warnings.resetwarnings()


# -------------------------
# Plotting MEG data
# -------------------------
def plot_evoked_gfp(epo_path,save_path,sub_nb, epo_type):
    """Plot evoked response based on provided Epochs. Save the resulting figure in a save_path.

    Args:
        epo_path (str): directory to read epochs.
        save_path (str): directory to save figure as png.
        sub_nb (int): subject number (ex: 1, 2, 3...)
        epo_type (str): epoch type (e.g. Item, sequence, etc). Used in the title and name of png.
    """
    # -- modify epo_type string
    adjusted_epo_type = epo_type.replace(' ', '_')
    # -- load epochs
    epo=mne.read_epochs(epo_path,preload=True)
    epo.apply_baseline((-0.2,0))

    # -- get evoked
    evoked=epo.average()

    # -- Plot evoked
    # Count the number of MAG and GRAD sensors
    n_mag = len(mne.pick_types(evoked.info, meg='mag'))
    n_grad = len(mne.pick_types(evoked.info, meg='grad'))

    # Plot evoked with GFP and title
    fig = evoked.plot(gfp=True, titles='Evoked Response with GFP')

    # Customize the plot to add titles to each subfigure
    sensor_titles = [f'MAG ({n_mag})', f'GRAD ({n_grad})']
    for ax, title in zip(fig.axes, sensor_titles):
        ax.set_title(title)
    # Add a main title
    fig.suptitle(f'Evoked Response with GFP - {epo_type} - sub_{sub_nb:02}', fontsize=16)

    # -- Save figure: 
    fig.savefig(os.path.join(save_path,f'sub{sub_nb:02}_{adjusted_epo_type}_evoked_response_gfp.png'))
    del(epo)
    del(evoked)
    
    
# ******************************************************************
# BIDS RELATED FUNCTIONS
# ******************************************************************

def extract_events_and_event_IDs(raw):
    """
    Extracts specific events and their IDs from raw MEG/EEG data using MNE.
    """
    events = mne.find_events(raw, min_duration=0.01)

    # Filter out feedback score events (trigger codes that are multiples of 10)
    events_sequence_presentation = events[(events[:, 2] > 10) & (events[:, 2] < 97)]
    events_fixation = events[events[:, 2] == 9]
    events_fixation_blue = events[events[:, 2] == 99]
    events_resting_phase = events[events[:, 2] == 128]

    events_of_interest = np.vstack([events_fixation, events_fixation_blue, events_sequence_presentation, events_resting_phase])

    dict_fixation = {'Fixation': 9}
    dict_reproduction = {'Reproduction': 99}
    dict_sequences = event_dict
    dict_resting_state = {'Resting_state': 128}

    event_ids_dict = {**dict_fixation, **dict_reproduction, **dict_sequences, **dict_resting_state}

    return events_of_interest, event_ids_dict

def prepare_data_for_mne_bids_pipeline(subject='02', base_path="/Users/fosca/Documents/Fosca/INSERM/Projets/ReplaySeq/ICM/Data_ICM/",
                                       triux=True, task_name='reproduction', cal_filename='sss_cal_3101_160108.dat', ct_filename='ct_sparse.fif',
                                       run_names=[f"{i:02}" for i in range(1, 19)]):
    """
    Prepare and convert MEG data to BIDS format for MNE-BIDS pipeline processing.
    """
    original_data_path = base_path + "/raw/"
    root = base_path + '/BIDS/'

    for run in run_names:
        print(f"--- saving in bids format run {run} ---")
        data_path = original_data_path + subject + f'/run{run}_raw.fif'
        raw = mne.io.read_raw_fif(data_path, allow_maxshield=True, preload=True)

        if triux:
            # Example of renaming BIO channels for Triux system (assuming BIO channels need renaming)
            mapping = {ch: ch.replace('BIO', 'EEG') for ch in raw.ch_names if 'BIO' in ch}
            raw.rename_channels(mapping)

        events, event_ids = extract_events_and_event_IDs(raw)

        # Create the BIDS path
        bids_path = BIDSPath(subject=subject, task=task_name, run=run, datatype='meg', root=root)

        # Write the raw data
        write_raw_bids(raw, bids_path=bids_path, allow_preload=True, format='FIF',events=events,
                       event_id=event_ids, overwrite=True)

        # Write events to a file
       # events_fname = bids_path.copy().update(suffix='eve', extension='.fif')
        #mne.write_events(events_fname, events, overwrite=True)

        # Write event_ids to a TSV file (optional)
        #event_id_path = bids_path.copy().update(suffix='events', extension='.tsv')
        #mne_bids.tsv_to_str(event_ids, event_id_path.fpath, overwrite=True)

        # Write MEG calibration and crosstalk files
        cal_fname = root + f'/system_calibration_files/{cal_filename}'
        ct_fname = root + f'/system_calibration_files/{ct_filename}'
        write_meg_calibration(calibration=cal_fname, bids_path=bids_path)
        write_meg_crosstalk(fname=ct_fname, bids_path=bids_path)
