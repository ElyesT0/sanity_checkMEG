from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import glob
import matplotlib.pyplot as plt
import numpy as np
import mne
import pandas as pd
from sklearn.decomposition import PCA
from mne.decoding import UnsupervisedSpatialFilter
from mne.decoding import SlidingEstimator, GeneralizingEstimator, cross_val_multiscore

# ______________________________________________________________________________________________________________________
def SVM_decoder():
    """
    This is the basic SVM decoder that generalizes across time in a one versus all manner
    :return:
    """
    clf = make_pipeline(StandardScaler(), SVC(C=1, kernel='linear', probability=True))
    time_gen = GeneralizingEstimator(clf, scoring=None, n_jobs=-1, verbose=True)
    return time_gen

# ______________________________________________________________________________________________________________________
def plot_gat(score, times, chance=0.5, title ='SVM classifier, scores', vmin = 0, vmax = 0):
    if vmin == 0:
        vmin = chance - 5 * np.std(score)
    if vmax == 0:
        vmax = chance + 5 * np.std(score)
    fig, ax = plt.subplots(1)
    im = ax.matshow(score, vmin=vmin, vmax=vmax, cmap='RdBu_r', origin='lower',
                    extent=times[[0, -1, 0, -1]])
    ax.axhline(0., color='k')
    ax.axvline(0., color='k')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xlabel('Testing Time (s)')
    ax.set_ylabel('Training Time (s)')
    ax.set_title(title)
    plt.colorbar(im, ax=ax)

    return fig

#_______________________________________________________________________________________________________________________
def extract_first_occurrences(lst):
    first_occurrences = {}
    seen = set()

    for index, value in enumerate(lst):
        if value not in seen:
            seen.add(value)
            first_occurrences[index] = value

    return first_occurrences

# _______________________________________________________________________________________________________________________
def from_seq_to_seqID(sequence):

    seq = sequence.replace('[','')
    seq = seq.replace(']','')
    seq = seq.split(',')
    seq = [int(i) for i in seq]
    new_seq = np.asarray([0]*12)
    l = 0
    A = seq[l]
    inds_A = np.where(np.asarray(seq) == A)[0]
    new_seq[inds_A] = l
    pres_pos = [A]
    for ii in range(1,12):
        if seq[ii] not in pres_pos:
            l += 1
            A = seq[ii]
            inds_A = np.where(np.asarray(seq) == A)[0]
            new_seq[inds_A] = l
            pres_pos.append(A)

    SEQS = {'[0 1 0 1 0 1 0 1 0 1 0 1]':'Rep2','[0 1 1 1 1 0 0 1 0 0 0 1]':'CRep2',
    '[0 1 2 0 1 2 0 1 2 0 1 2]':'Rep3', '[0 1 2 0 2 1 1 2 0 1 0 2]':'CRep3',
    '[0 1 2 3 0 1 2 3 0 1 2 3]':'Rep4','[0 1 2 3 2 1 3 0 0 3 1 2]':'CRep4',
    '[0 0 1 1 2 2 0 0 1 1 2 2]':'RepEmbed', '[0 0 1 1 2 2 0 0 2 2 1 1]':'C1RepEmbed',
    '[0 1 2 0 2 1 0 1 2 0 2 1]':'C2RepEmbed'}

    return (SEQS[str(new_seq)])

# _______________________________________________________________________________________________________________________
def load_behavioral_file(subject, start_keyword='subject_id'):
    """
    Extracts and processes a pandas DataFrame from a CSV file by dynamically finding the start of relevant data,
    and converts specific string representations of lists into actual lists.

    Parameters:
    - file_path: str, the path to the CSV file.
    - start_keyword: str, the keyword that indicates the start of relevant data (default is 'subject_id').

    Returns:
    - df: pandas DataFrame, containing the extracted and processed data.
    """
    start_line = None
    if subject == '01' or subject == '02':
        # todo for Elyès
        file_path = glob(config.behavior_raw_path + "/sub-" + subject + "/*.csv")
        start_line = 0
    else:
        # todo for Elyès
        file_path = glob(config.behavior_raw_path + "/sub-" + subject + "/*.xpd")

    file_path = file_path[0]
    # Read the file into a list of lines
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Find the line number where the relevant data starts
    if start_line !=0:
        for i, line in enumerate(lines):
            if start_keyword in line:
                start_line = i
                break

    # Load the CSV starting from the identified line
    df = pd.read_csv(file_path, skiprows=start_line, index_col=False)

    if subject == '01' or subject == '02':
        df.drop(columns=['participant_number'])
        df['subject_id'] = int(subject)
        df['sequenceName'] = [from_seq_to_seqID(seq) for seq in df['PresentedSequence'].values]
    else:
        columns_of_interest = [
            'subject_id', 'block', 'sequenceName', 'trial',
            'PresentedSequence', 'ProducedSequence', 'RTs', 'Performance'
        ]
        df = df[columns_of_interest]

    # Select only the columns of interest

    # Convert string representations of lists to actual lists for relevant columns
    list_columns = ['PresentedSequence', 'ProducedSequence', 'RTs']
    for col in list_columns:
        df[col] = df[col].apply(eval)

    return df

# ____________________________________________________________________________________________________
def expand_dataframe_with_position(df):
    """
    Expands the dataframe by duplicating each row 12 times, corresponding to each item in the PresentedSequence,
    and adds a column 'PresentedPosition' that contains each of the 12 items.

    Parameters:
    - df: pandas DataFrame, the original dataframe.

    Returns:
    - expanded_df: pandas DataFrame, the expanded dataframe with 'PresentedPosition' column added.
    """
    # Initialize an empty list to store the expanded rows
    expanded_rows = []

    # Iterate through each row in the dataframe
    for _, row in df.iterrows():
        # Get the PresentedSequence for the current row
        presented_sequence = row['PresentedSequence']

        # Check if the length of PresentedSequence is 12 (as expected)
        if len(presented_sequence) != 12:
            raise ValueError("PresentedSequence does not have exactly 12 items.")

        # Create 12 new rows, one for each item in PresentedSequence
        for i, item in enumerate(presented_sequence):
            # Copy the original row and add the PresentedPosition
            new_row = row.copy()
            new_row['PresentedPosition'] = item
            expanded_rows.append(new_row)

    # Convert the list of expanded rows into a new DataFrame
    expanded_df = pd.DataFrame(expanded_rows)

    return expanded_df

#_______________________________________________________________________________________________________________________
def extract_epochs_first_presentation(subject):
    """
    This function extracts from epochs items the epochs for which the position could not be anticipated
    :param subject:
    :return: epochs of spatial items that could not be anticipated
    """
    import warnings

    # todo for Elyès
    path = config.derivatives_path + '/items/sub-'+subject+'/meg/sub-'+subject+'_task-reproduction_epo.fif'

    warnings.warn("Careful, this function only works for epochs before rejection", UserWarning)
    epochs = mne.read_epochs(path, preload=True)
    if subject =='01' or subject =='02':
        epochs=epochs[epochs.events[:,2]!=1]

    metadata = load_behavioral_file(subject)
    epochs.metadata = expand_dataframe_with_position(metadata)
    presented_sequences = metadata["PresentedSequence"].values
    indices = []
    for k, seq in enumerate(presented_sequences):
        first_occurrences = extract_first_occurrences(seq)
        indices.append([i+12*k for i in list(first_occurrences.keys())])
    indices = np.concatenate(indices)
    return epochs[indices]


def run_decoding_sanity_check(sub_nb):

    epochs = extract_epochs_first_presentation(sub_nb)
    epochs.apply_baseline(baseline=(-0.1,0.))
    epochs.pick_types(meg=True)
    # on décime les données pour que ça soit plus léger
    epochs.decimate(4)
    PCA_model = UnsupervisedSpatialFilter(PCA(70), average=False)
    # on fait une PCA qui correspond aux données après maxfilter pour qu'il n'y ait que des dimensions informatives dans le décodeur
    PCA_data = PCA_model.fit_transform(epochs.get_data())
    epochs._data = PCA_data
    model_gat = SVM_decoder()

    One_score_simple_gat = cross_val_multiscore(model_gat, X=epochs.get_data(), y=epochs.events[:, 2])
    fig = plot_gat(np.mean(One_score_simple_gat, axis=0), times=epochs.times,
                     chance=1 / len(np.unique(epochs.events[:, 2])))
    plt.shows()

    return fig


