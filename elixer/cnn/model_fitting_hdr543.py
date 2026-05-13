from model_fitting_config import *

#---------------------------------------------------
# Model Fitting

# Input directory containing 2D spectra stored as ['im_sum'] in f'{detectid}.npy' files
input_dir = '/scratch/10359/shiromukae/2D_Spectra/lae/' 
# Subdirectories for each HDR data release
sub_dirs = {
    'hdr3': 'hdr3/',
    'hdr4': 'hdr4/',
    'hdr5': 'hdr5/',
}

# Process each HDR directory
for hdr in ['hdr5','hdr4','hdr3']:
    # Construct the full path to the HDR directory
    hdr_dir = os.path.join(input_dir, sub_dirs[hdr])
    print(f"Processing {hdr_dir}...")

    # Apply the trained CNN model to all spectra in the directory
    # label=-1 indicates that these data do not have known labels
    result_table = process_directory(hdr_dir, label=-1)

    # Save the output table if results are available
    if result_table:
        output_table = f"cnn_{training_id}_{hdr}.txt"
        ascii.write(result_table, os.path.join(dir_child, output_table), overwrite=True)
        print(f"Saved: {output_table}")


#---------------------------------------------------
# Catalog Output

# Output file names for each HDR catalog
out_tab_hdr5 = f'cnn_{training_id}_hdr5.txt'
out_tab_hdr4 = f'cnn_{training_id}_hdr4.txt'
out_tab_hdr3 = f'cnn_{training_id}_hdr3.txt'

# Read CNN score tables for each HDR catalog
tab_hdr5 = ascii.read(dir_child+out_tab_hdr5)
tab_hdr4 = ascii.read(dir_child+out_tab_hdr4)
tab_hdr3 = ascii.read(dir_child+out_tab_hdr3)

# Stack the HDR5, HDR4, and HDR3 tables into a single table
tab_hdr5_hdr4_hdr3 = vstack([tab_hdr5, tab_hdr4, tab_hdr3])
# Sort the combined table by detectid
tab_hdr5_hdr4_hdr3.sort('detectid')
# Keep only detectid and the averaged CNN score
tab_hdr5_hdr4_hdr3 = tab_hdr5_hdr4_hdr3['detectid','CNN_Score_2D_Spectra']

output_table = f'cnn_{training_id}_hdr5.0.1_lae.txt'
output_table = os.path.join(dir_child, output_table)
ascii.write(tab_hdr5_hdr4_hdr3, output_table, overwrite=True)
print(f"Saved: {output_table}")

