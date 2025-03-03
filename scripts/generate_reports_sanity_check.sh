#!/bin/bash
# ./run_notebooks.sh
#
# This script automates the execution of Jupyter notebooks for multiple subjects
# by modifying a parameter in a Python script, running the corresponding notebook,
# and converting the executed notebook to an HTML report. 
# It generates the SANITY checks reports and upload them to my website.
#
# STEPS:
# 1. Define paths for:
#    - The Python script containing the `sub_nb` parameter
#    - The directory of notebooks
#    - The model notebook used as a template
#    - The reports directory for storing HTML outputs
#
# 2. Ensure the reports directory exists (`mkdir -p` ensures it is created if missing).
#
# 3. Loop through subject numbers from 4 to 18:
#    - Format the number with leading zeros (e.g., "04", "05", ..., "18").
#    - Modify the `sub_nb` value in `parameters.py` using `sed` (create a backup `.bak` file).
#    - Define paths for the subject-specific notebook and its HTML output.
#    - If the notebook does not exist, copy the model notebook to create it.
#    - Execute the notebook using `jupyter nbconvert --execute`.
#    - If execution succeeds:
#      - Convert the executed notebook to an HTML report, excluding input cells.
#      - Move the generated HTML file to the final reports directory using `rsync`,
#        ensuring only modified files are updated and removing the temporary copy.
#
# 4. Print status messages indicating progress and errors where applicable.
#
# This ensures all subject notebooks are executed, updated, and properly stored
# as HTML reports for further review.

#!/bin/bash
# ./run_notebooks.sh

# Define paths
PYTHON_SCRIPT="/Users/elyestabbane/Documents/UNICOG/8-MEG/sanity_checkMEG/scripts/modules/parameters.py"
NOTEBOOK_DIR="/Users/elyestabbane/Documents/UNICOG/8-MEG/sanity_checkMEG/scripts/reports_sanity_checks-preprocessing"
NOTEBOOK_PREFIX="sanityCheck_preprocessing_sub_"
MODEL_NOTEBOOK="${NOTEBOOK_DIR}/sanityCheck_preprocessing_model.ipynb"
REPORTS_DIR="/Users/elyestabbane/Documents/UNICOG/8-MEG/sanity_checkMEG/reports/pre-processing"

# Ensure the reports directory exists
mkdir -p "$REPORTS_DIR"

# Loop through sub_nb values from 4 to 18
for i in {4..18}
do
    # Format the number with leading zero if necessary
    PADDED_NUM=$(printf "%02d" $i)

    echo "Running with sub_nb=$PADDED_NUM"

    # Modify the sub_nb value in parameters.py using `sed`
    sed -i '.bak' "s/sub_nb=[0-9][0-9]*/sub_nb=$i/" "$PYTHON_SCRIPT"

    # Define the full notebook and HTML paths
    NOTEBOOK_PATH="${NOTEBOOK_DIR}/${NOTEBOOK_PREFIX}${PADDED_NUM}.ipynb"
    TEMP_HTML_PATH="${NOTEBOOK_DIR}/${NOTEBOOK_PREFIX}${PADDED_NUM}.html"
    FINAL_HTML_PATH="${REPORTS_DIR}/${NOTEBOOK_PREFIX}${PADDED_NUM}.html"

    # If notebook does not exist, copy the model notebook
    if [[ ! -f "$NOTEBOOK_PATH" ]]; then
        echo "Notebook not found. Creating from model: $NOTEBOOK_PATH"
        cp "$MODEL_NOTEBOOK" "$NOTEBOOK_PATH"
    fi

    # Execute the notebook
    echo "Executing Jupyter Notebook: $NOTEBOOK_PATH"
    jupyter nbconvert --execute --to notebook --inplace "$NOTEBOOK_PATH"

    # Check if execution was successful
    if [[ $? -eq 0 ]]; then
        echo "✅ Successfully executed $NOTEBOOK_PATH"

        # Convert the executed notebook to HTML without input cells
        echo "Converting $NOTEBOOK_PATH to HTML..."
        jupyter nbconvert --to html --no-input "$NOTEBOOK_PATH" --output "$TEMP_HTML_PATH"

        # Verify HTML file creation
        if [[ -f "$TEMP_HTML_PATH" ]]; then
            echo "✅ HTML file created: $TEMP_HTML_PATH"

            # Use rsync to move and replace only if changed
            rsync -av --remove-source-files "$TEMP_HTML_PATH" "$FINAL_HTML_PATH"

            echo "✅ Moved HTML to: $FINAL_HTML_PATH"
        else
            echo "❌ Failed to create HTML for $NOTEBOOK_PATH"
        fi
    else
        echo "❌ Error executing $NOTEBOOK_PATH. Skipping conversion."
    fi
done

echo "All notebooks processed successfully!"
