-- About the file_names.json.
It contains a set of nested dictionnaries. 
- ExPyriment: Contains path to data collected in xpd format using the python library expyriment.
    -- Raw: xpd file path. This is the data output by the expyriment instance : run_megSeq.py
    -- Processed: csv file path. This is the data that has been preprocessed with preprocessing_behavioral.py

- MNE: Contains path to data collected with the MEG machine.
    -- Raw: path to fif file for each participants and each run.
        structure: 
            -- sub-x: (ex: sub-01) contains the paths to fif file of runs for subject number x.
                -- run-x: (ex: run-01) contains the path to fif file for run number x.
    
    -- behavioral: same structure but for csv files of behavioral data collected.