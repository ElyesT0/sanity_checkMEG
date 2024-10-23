import mne

study_name = "REPLAYSEQ"

bids_root = "/Volumes/T5_EVO/1-experiments/REPLAYSEQ/7-data_neurospin/1-main_MEG/BIDS"
deriv_root = "/Volumes/T5_EVO/1-experiments/REPLAYSEQ/7-data_neurospin/1-main_MEG/epochs/2-epochs_items/mne-bids-pipeline"

task = "reproduction"

runs = 'all'

find_flat_channels_meg = True
find_noisy_channels_meg = True
use_maxwell_filter = True
mf_ctc_fname = bids_root + "/system_calibration_files/ct_sparse.fif"
mf_cal_fname = bids_root + "/system_calibration_files/sss_cal_3176_20240123_2.dat"

# ch_types = ["meg","eeg"] # No numerization points for EEG so it outputs an error.
ch_types = ["meg"]
data_type='meg'

# Trying to give a standard positionning of eeg channels
#eeg_template_montage = mne.channels.make_standard_montage("standard_1005")
# eeg_template_montage = "easycap-M10"

l_freq =1.0
h_freq = 40.0

# # SSP and peak-to-peak rejection
# spatial_filter = "ssp"
# n_proj_eog = dict(n_mag=0, n_grad=0)
# n_proj_ecg = dict(n_mag=2, n_grad=2)
# ssp_reject_ecg = {"grad": 2000e-13, "mag": 5000e-15}
# reject = None

# SSP and peak-to-peak rejection
ica_l_freq = 1.0
spatial_filter = "ica" #NOTE DIFF
n_proj_eog = dict(n_mag=0, n_grad=0)
n_proj_ecg = dict(n_mag=2, n_grad=2)
ica_reject = {"grad": 4000e-13, "mag": 4e-12} #NOTE DIFF
ica_max_iterations = 3000
#ica_decim = None

# Epochs
epochs_tmin = -0.2
epochs_tmax = 0.6
epochs_decim = 4
baseline = (None,0)

# Conditions / events to consider when epoching
conditions = ['SequenceID-Rep2/Position-1', 'SequenceID-Rep2/Position-2', 'SequenceID-Rep2/Position-3', 'SequenceID-Rep2/Position-4', 'SequenceID-Rep2/Position-5', 'SequenceID-Rep2/Position-6', 'SequenceID-CRep2/Position-1', 'SequenceID-CRep2/Position-2', 'SequenceID-CRep2/Position-3', 'SequenceID-CRep2/Position-4', 'SequenceID-CRep2/Position-5', 'SequenceID-CRep2/Position-6', 'SequenceID-Rep3/Position-1', 'SequenceID-Rep3/Position-2', 'SequenceID-Rep3/Position-3', 'SequenceID-Rep3/Position-4', 'SequenceID-Rep3/Position-5', 'SequenceID-Rep3/Position-6', 'SequenceID-CRep3/Position-1', 'SequenceID-CRep3/Position-2', 'SequenceID-CRep3/Position-3', 'SequenceID-CRep3/Position-4', 'SequenceID-CRep3/Position-5', 'SequenceID-CRep3/Position-6', 'SequenceID-Rep4/Position-1', 'SequenceID-Rep4/Position-2', 'SequenceID-Rep4/Position-3', 'SequenceID-Rep4/Position-4', 'SequenceID-Rep4/Position-5', 'SequenceID-Rep4/Position-6', 'SequenceID-CRep4/Position-1', 'SequenceID-CRep4/Position-2', 'SequenceID-CRep4/Position-3', 'SequenceID-CRep4/Position-4', 'SequenceID-CRep4/Position-5', 'SequenceID-CRep4/Position-6', 'SequenceID-RepEmbed/Position-1', 'SequenceID-RepEmbed/Position-2', 'SequenceID-RepEmbed/Position-3', 'SequenceID-RepEmbed/Position-4', 'SequenceID-RepEmbed/Position-5', 'SequenceID-RepEmbed/Position-6', 'SequenceID-C1RepEmbed/Position-1', 'SequenceID-C1RepEmbed/Position-2', 'SequenceID-C1RepEmbed/Position-3', 'SequenceID-C1RepEmbed/Position-4', 'SequenceID-C1RepEmbed/Position-5', 'SequenceID-C1RepEmbed/Position-6', 'SequenceID-C2RepEmbed/Position-1', 'SequenceID-C2RepEmbed/Position-2', 'SequenceID-C2RepEmbed/Position-3', 'SequenceID-C2RepEmbed/Position-4', 'SequenceID-C2RepEmbed/Position-5', 'SequenceID-C2RepEmbed/Position-6']

# This is often helpful when doing multiple subjects.  If 1 subject fails processing stops
#on_error = 'continue'

# Decoding
decode = False

# Noise estimation
process_empty_room = True

# noise_cov == None
ssp_meg = "combined"

# Adding the ICA component setting
ica_n_components = 15

# Configuration to apply ICA and the new reject parameter
# This block would typically be run by the MNE-BIDS pipeline

# Channel re-tagging dictionary
# channel_retagging = {
#     'BIO002': 'eog',
#     'BIO003': 'ecg'
# }

# def preprocess_and_apply_ica(raw):
#     # Apply channel re-tagging
#     raw.set_channel_types(channel_retagging)
    
#     # Apply ICA
#     ica = mne.preprocessing.ICA(n_components=ica_n_components, reject=ica_reject)
#     ica.fit(raw)
    
#     # Optionally save the ICA solution if needed
#     # ica.save(f"{deriv_root}/ica_solution.fif")
    
#     return raw, ica