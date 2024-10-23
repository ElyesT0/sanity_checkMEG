import mne 
from modules.parameters import *

def convert_EEG(path, name, save_path):
    # Load the raw MEG-EEG file
    raw = mne.io.read_raw_fif(os.path.join(path, name), allow_maxshield=True, preload=True)
    
    # Pick only EEG channels
    eeg_picks = mne.pick_types(raw.info, meg=False, eeg=True)
    
    # Create a new Raw object with only EEG data
    raw_eeg = raw.copy().pick(eeg_picks)
    
    # Define the path for the temporary EEG data in FIF format
    fif_path = os.path.join(save_path, f'eeg-{name}')
    
    # Save the EEG data to a new file in FIF format
    raw_eeg.save(fif_path, overwrite=True)
    
    # Convert the Raw object to numpy arrays and write using pybv
    data, times = raw_eeg.get_data(return_times=True)
    sfreq = raw_eeg.info['sfreq']
    ch_names = raw_eeg.ch_names

    # Define the base filename for BrainVision files
    base_filename = os.path.join(save_path, f'eeg_only_data-{name.split(".")[0]}')
    
    # Write the data to BrainVision format
    write_brainvision(data=data, sfreq=sfreq, ch_names=ch_names, 
                      fname_base=base_filename, 
                      folder_out=save_path)

    # Remove the temporary FIF file
    os.remove(fif_path)

    print(f'{name} extracted and saved successfully in BrainVision format. FIF file deleted.')

    # Clean up to free memory
    del raw, eeg_picks, raw_eeg, data, times, sfreq, ch_names

for name in list_raw_fif:
    convert_EEG(path_raw,name,path_save_eeg)