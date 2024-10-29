import os
import mne
from mne_bids import BIDSPath, write_raw_bids, write_meg_calibration, write_meg_crosstalk
from mne.preprocessing import ICA, corrmap, create_ecg_epochs, create_eog_epochs, maxwell_filter
import mne_bids
import numpy as np
import pandas as pd
import os
import logging
import warnings
import matplotlib.pyplot as plt
from scipy import stats
import gc
from tqdm import tqdm
from copy import deepcopy
import pickle
import json
from pybv import write_brainvision


# ******************************************************************
# PATHS
# ******************************************************************
post_processing=True
# Subject number
sub_nb=8

# Get file name in XPD from the file_names.json file
with open('modules/file_names.json', 'r') as file:
    xpd_fileName_dict=json.load(file)
raw_xpd_fileName=xpd_fileName_dict['experiment']['raw']
xpd_fileName=raw_xpd_fileName[f'{sub_nb}']


path_root="/Volumes/T5_EVO/1-experiments/REPLAYSEQ/2-Data"

# Path to MEG data
path_raw=os.path.join(path_root,f"raw/Data_neurospin/sub-{sub_nb:02}")

try:
    # Path Empty room
    path_empty_room=os.path.join(path_raw,"empty_room.fif")
except FileNotFoundError:
    print('No empty room file Found')

# Path to Behavioral data
path_behavioral=os.path.join(path_root,f'behavior/raw/sub-{sub_nb:02}/{xpd_fileName}')

# Path to calibration files
path_system_calibration=os.path.join(path_root,"1-main_MEG/BIDS/system_calibration_files")
path_calibration=os.path.join(path_system_calibration,"sss_cal_3176_20240123_2.dat")
path_cross_talk=os.path.join(path_system_calibration,"ct_sparse.fif")

# Names of runs fif files
list_raw = os.listdir(path_raw)
list_raw_fif = [name for name in list_raw if 'run' in name and name.endswith('.fif')]
all_run_fif=[os.path.join(path_raw,name) for name in list_raw_fif]

# Post processing Paths
if post_processing:
    path_derivatives_fullSequence=os.path.join(path_root,f"derivatives/sequence/sub-{sub_nb:02}/meg")
    path_derivatives_items=os.path.join(path_root,f"derivatives/items/sub-{sub_nb:02}/meg")
    list_derivatives_items=os.listdir(path_derivatives_items)
    list_raw_sss = [name for name in list_derivatives_items if 'proc-sss' in name and name.endswith('.fif')]
    all_run_sss=[os.path.join(path_derivatives_items,name) for name in list_raw_sss] # Prendre tout là dedans: /Volumes/T5_EVO/1-experiments/REPLAYSEQ/2-Data/derivatives/items/sub-08/meg
    epo_item_path=os.path.join(path_derivatives_items,f'sub-{sub_nb:02}_task-reproduction_epo.fif')
    epo_sequence_path=os.path.join(path_derivatives_fullSequence,f'sub-{sub_nb:02}_task-reproduction_epo.fif')
    
# Save path
# -- Behavioral
path_save_processed_behavioral=os.path.join(path_root,f'behavior/processed/sub-{sub_nb:02}')
path_processed_behavioral_file=os.path.join(path_save_processed_behavioral,f'sub-{sub_nb:02}-processed_meg_behavioral.csv')

# -- ICA
path_save_ICA=os.path.join(path_root,f'/1-main_MEG/objects/ica-saved/sub-{sub_nb:02}')

# -- Merged Behavioral
path_save_merged_behavioral=os.path.join(path_root,'behavior/merged')
path_save_merged_behavioral_file= os.path.join(path_save_merged_behavioral,'merged_behavioral_megSeq.csv')

# -- EEG only
path_save_eeg=os.path.join(path_root,f'1-main_MEG/eeg/sub-{sub_nb:02}')


# ******************************************************************
# USEFUL DICTIONARIES
# ******************************************************************
seq_name_list=['Rep2','CRep2','Rep3','CRep3','Rep4','CRep4', 'RepEmbed', 'C1RepEmbed', 'C2RepEmbed']
# Sequence names ordered in increasing level of difficulty 
ordered_seq_name_list=['RepEmbed','Rep2','Rep3','Rep4','C1RepEmbed','C2RepEmbed','CRep2','CRep3','CRep4']

event_dict={'fixation_blue': 4,
'fixation': 5,
'Rep2-1': 6,
'Rep2-2': 7,
'Rep2-3': 8,
'Rep2-4': 9,
'Rep2-5': 10,
'Rep2-6': 11,
'CRep2-1': 12,
'CRep2-2': 13,
'CRep2-3': 14,
'CRep2-4': 15,
'CRep2-5': 16,
'CRep2-6': 17,
'Rep3-1': 18,
'Rep3-2': 19,
'Rep3-3': 20,
'Rep3-4': 21,
'Rep3-5': 22,
'Rep3-6': 23,
'CRep3-1': 24,
'CRep3-2': 25,
'CRep3-3': 26,
'CRep3-4': 27,
'CRep3-5': 28,
'CRep3-6': 29,
'Rep4-1': 30,
'Rep4-2': 31,
'Rep4-3': 32,
'Rep4-4': 33,
'Rep4-5': 34,
'Rep4-6': 35,
'CRep4-1': 36,
'CRep4-2': 37,
'CRep4-3': 38,
'CRep4-4': 39,
'CRep4-5': 40,
'CRep4-6': 41,
'RepEmbed-1': 42,
'RepEmbed-2': 43,
'RepEmbed-3': 44,
'RepEmbed-4': 45,
'RepEmbed-5': 46,
'RepEmbed-6': 47,
'C1RepEmbed-1': 48,
'C1RepEmbed-2': 49,
'C1RepEmbed-3': 50,
'C1RepEmbed-4': 51,
'C1RepEmbed-5': 52,
'C1RepEmbed-6': 53,
'C2RepEmbed-1': 54,
'C2RepEmbed-2': 55,
'C2RepEmbed-3': 56,
'C2RepEmbed-4': 57,
'C2RepEmbed-5': 58,
'C2RepEmbed-6': 59,
'win': 60,
'loss': 61}

event_itemsOnly_dict={
    'Rep2-1': 6,
'Rep2-2': 7,
'Rep2-3': 8,
'Rep2-4': 9,
'Rep2-5': 10,
'Rep2-6': 11,
'CRep2-1': 12,
'CRep2-2': 13,
'CRep2-3': 14,
'CRep2-4': 15,
'CRep2-5': 16,
'CRep2-6': 17,
'Rep3-1': 18,
'Rep3-2': 19,
'Rep3-3': 20,
'Rep3-4': 21,
'Rep3-5': 22,
'Rep3-6': 23,
'CRep3-1': 24,
'CRep3-2': 25,
'CRep3-3': 26,
'CRep3-4': 27,
'CRep3-5': 28,
'CRep3-6': 29,
'Rep4-1': 30,
'Rep4-2': 31,
'Rep4-3': 32,
'Rep4-4': 33,
'Rep4-5': 34,
'Rep4-6': 35,
'CRep4-1': 36,
'CRep4-2': 37,
'CRep4-3': 38,
'CRep4-4': 39,
'CRep4-5': 40,
'CRep4-6': 41,
'RepEmbed-1': 42,
'RepEmbed-2': 43,
'RepEmbed-3': 44,
'RepEmbed-4': 45,
'RepEmbed-5': 46,
'RepEmbed-6': 47,
'C1RepEmbed-1': 48,
'C1RepEmbed-2': 49,
'C1RepEmbed-3': 50,
'C1RepEmbed-4': 51,
'C1RepEmbed-5': 52,
'C1RepEmbed-6': 53,
'C2RepEmbed-1': 54,
'C2RepEmbed-2': 55,
'C2RepEmbed-3': 56,
'C2RepEmbed-4': 57,
'C2RepEmbed-5': 58,
'C2RepEmbed-6': 59,
}


reverse_event_dict={value: key for key, value in event_dict.items()}

# ******************************************************************
# VISUALIZATION PARAMETERS
# ******************************************************************


plot_figsize=(8,5)
plot_colors=['#03045E', '#03045E', '#0077B6', '#0077B6', '#00B4D8', '#00B4D8', '#ADE8F4', '#ADE8F4', 
         '#03045E', '#03045E', '#0077B6', '#0077B6', '#00B4D8', '#00B4D8', '#ADE8F4', '#ADE8F4', 
         '#03045E', '#03045E', '#0077B6', '#0077B6', '#00B4D8', '#00B4D8', '#ADE8F4', '#ADE8F4', '#ADE8F4']
title_size=15
padding_size=10