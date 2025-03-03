from modules.functions_sanity_check import *

# ******************************************************************
# PATHS
# ******************************************************************
base_path='/Volumes/T5_EVO/1-experiments/REPLAYSEQ/7-data_neurospin/1-main_MEG/'
subject=1
cal_filename='sss_cal_3176_20240123_2.dat'
ct_filename='ct_sparse.fif'



# ******************************************************************
# RUN FUNCTIONS
# ******************************************************************
prepare_data_for_mne_bids_pipeline(subject=f'0{subject}', base_path=base_path,
                                       triux=True, task_name='reproduction',cal_filename=cal_filename,ct_filename=ct_filename,
                                       run_names = [f"{i:02}" for i in range(1, 14)])