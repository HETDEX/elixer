#---------------------------------------------------
# Python modules
import os
import sys
#sys.path.append('./model/')
import numpy as np
from astropy.table import Table, vstack, join
from astropy.io import ascii
import torch
from torch.utils.data import Dataset, DataLoader
#from elixer import global_config as G

def mkdir_output(output_dir):
    # Create the output directory if it does not exist
    if os.path.isdir(output_dir): None
    else: os.mkdir(output_dir)


#---------------------------------------------------
# Loading Trained Model
training_id = '14.0.1_d0.2' # Training ID
model_name = 'TDSA_LeakyGAP' # Model architecture
# Path to the trained model parameters from 3-fold cross-validation
ELIXER_CODE_PATH = os.path.dirname(os.path.realpath(__file__))
best_model_path = os.path.join(ELIXER_CODE_PATH, './model/')

# Input size: channels, height, width
# The cutout is applied in the data-loading step below
input_size = (1, 9, 40)  

num_classes = 1  # Binary classification: 0 or 1
dropout_rate = 0.2  # Required for loading the model architecture, but not used during fitting

# Create output directories
dir_parent = './output/'
dir_child = dir_parent + training_id + '/' 
mkdir_output(dir_parent)
mkdir_output(dir_child)


#---------------------------------------------------
# Model architecture selection
def create_model(model_name, input_channels, num_classes, dropout_rate):
    # Select and initialize the model architecture
    if model_name == "TDSA_LeakyGAP":
        #import cnn.model_fitting_config as ML_CNN
        from cnn.model.TDSA_LeakyGAP_logit import TDSA_LeakyGAP_logit
        return TDSA_LeakyGAP_logit(input_channels=input_channels, num_classes=num_classes, dropout_rate=dropout_rate)    

# Use GPU if available; otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Create the model architecture
model = create_model(model_name, input_size[0], num_classes, dropout_rate)


#---------------------------------------------------
# Dataset class


# HERE DD ... take the array (or list) of 2d Arrays rather than file_paths and lable in the constructor
class SpectraDataset(Dataset): #DD vesion
    def __init__(self, spec_cutouts, detectids, label=-1):
        self.spec_cutouts = spec_cutouts
        self.label = label #-1 == do not yet have label
        self.detectids = detectids

    def __len__(self):
        # Return the number of input spectra (always 1)
        return len(self.detectids)

    def __getitem__(self, idx):
        # Load one 2D spectrum from a .npy file
        #file_path = self.file_paths[idx]
        data = self.spec_cutouts[idx] #np.load(file_path).astype(np.float32)
        # Normalize the spectrum using its mean and standard deviation
        data = (data - np.mean(data)) / (np.std(data) + 1e-8)
        # Apply a spatial cutout: height x width = 9 x 40
        data = data[:, 30:70]
        # Add the channel dimension: (channels, height, width) = (1, 9, 40)
        data = np.expand_dims(data, axis=0)
        # Extract detectid from the file name
        detectid = self.detectids[idx]  #int(os.path.basename(file_path).split('.')[0])

        # Convert the spectrum and label to PyTorch tensors
        x = torch.from_numpy(data)
        y = torch.tensor(float(self.label), dtype=torch.float32)
        return x, y, detectid

# class SpectraDataset(Dataset):
#     def __init__(self, file_paths, label):
#         self.file_paths = file_paths
#         self.label = label
#
#     def __len__(self):
#         # Return the number of input spectra
#         return len(self.file_paths)
#
#     def __getitem__(self, idx):
#         # Load one 2D spectrum from a .npy file
#         file_path = self.file_paths[idx]
#         data = np.load(file_path).astype(np.float32)
#         # Normalize the spectrum using its mean and standard deviation
#         data = (data - np.mean(data)) / (np.std(data) + 1e-8)
#         # Apply a spatial cutout: height x width = 9 x 40
#         data = data[:, 30:70]
#         # Add the channel dimension: (channels, height, width) = (1, 9, 40)
#         data = np.expand_dims(data, axis=0)
#         # Extract detectid from the file name
#         detectid = int(os.path.basename(file_path).split('.')[0])
#
#         # Convert the spectrum and label to PyTorch tensors
#         x = torch.from_numpy(data)
#         y = torch.tensor(float(self.label), dtype=torch.float32)
#         return x, y, detectid


#---------------------------------------------------
# Fitting Process

def process_detections(spec_cutouts, detectids, label=-1, folds=3, batch_size=10000, dl_batch_size=256, num_workers=4):
   """

   :param spec_cutouts: [] array of 2D spec cutouts (the single top4 summations)
   :param detectids:  [] array of detectids that corresponds to the 2D cutouts
   :param label:
   :param folds:
   :param batch_size:
   :param dl_batch_size:
   :param num_workers:
   :return:
   """

   # Model fitting for each cross-validation fold
   data_tables = []
   if len(spec_cutouts) < num_workers:
        num_workers = 1

    # Clear memory before loading a new model
   for fold_num in range(folds):
       if device.type == "cuda":
           torch.cuda.empty_cache()

       print(f"\n===== Fold {fold_num + 1} Fitting =====")

       # Load the trained model parameters for this fold
       best_model_name = best_model_path + f"best_model_{training_id}_fold{fold_num + 1}.pth"
       state = torch.load(best_model_name, map_location="cpu")

       # Apply the trained parameters to the model architecture
       model.load_state_dict(state)
       model.to(device)
       model.eval()

       # Process files in large batches to avoid memory issues
       output_tables = []

       #
       # here ... remove the loop

       #
       for i in range(0, len(spec_cutouts), batch_size):
           batch_specs = spec_cutouts[i:i + batch_size]
           batch_dets = detectids[i:i + batch_size]

           # dd here, alter SpectraDataset

           # Create a dataset and data loader for this batch
           dataset = SpectraDataset(batch_specs, batch_dets, label)
           loader = DataLoader(
               dataset,
               batch_size=dl_batch_size,
               shuffle=False,
               num_workers=num_workers,
               pin_memory=(device.type == "cuda"),
               persistent_workers=(num_workers > 0),
           )

           detect_ids, cnn_scores = [], []

           # Disable gradient calculation during inference
           with torch.no_grad():
               for x, _, did in loader:
                   # Input shape: (batch_size, channels, height, width)
                   x = x.to(device, non_blocking=True)

                   # Compute model outputs as logits
                   logits = model(x)
                   logits = logits.squeeze(1)

                   # Convert logits to probabilities using sigmoid
                   probs = torch.sigmoid(logits).detach().cpu().numpy()

                   # Convert detectid tensor to a Python list if needed
                   if torch.is_tensor(did):
                       did = did.cpu().tolist()

                   # Store detectids and CNN scores
                   detect_ids.extend(did)
                   cnn_scores.extend(probs.tolist())

           # Sort the table by detectid
           table = Table({'detectid': detect_ids, f'CNN_Score_2D_Spectra_Fold_{fold_num + 1}': cnn_scores})
           table.sort('detectid')
           output_tables.append(table)

        # Stack all batch-level tables for this fold
       data_table = vstack(output_tables)
       data_tables.append(data_table)

   # ---------------------------------------------------
   # Merge results from all folds
   merged_table = data_tables[0]
   for table in data_tables[1:]:
       merged_table = join(merged_table, table, keys='detectid', join_type='left')

   # Compute the average CNN score across all folds
   fold_columns = [f'CNN_Score_2D_Spectra_Fold_{i}' for i in range(1, folds + 1)]
   merged_table[f'CNN_Score_2D_Spectra'] = np.mean([merged_table[col] for col in fold_columns], axis=0)
   return merged_table

def process_directory(directory, label, folds=3, batch_size=10000, dl_batch_size=256, num_workers=4):
    # Collect all .npy files in the input directory
    file_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npy')]
    file_paths.sort()

    # Model fitting for each cross-validation fold
    data_tables = []
    # Clear memory before loading a new model
    for fold_num in range(folds):
        if device.type == "cuda":
            torch.cuda.empty_cache()
            
        print(f"\n===== Fold {fold_num+1} Fitting =====")
    
        # Load the trained model parameters for this fold
        best_model_name = best_model_path+f"best_model_{training_id}_fold{fold_num+1}.pth"
        state = torch.load(best_model_name, map_location="cpu")

        # Apply the trained parameters to the model architecture
        model.load_state_dict(state)
        model.to(device)
        model.eval()

        # Process files in large batches to avoid memory issues        
        output_tables = []

        #
        # here ... remove the loop

        #
        for i in range(0, len(file_paths), batch_size):
            batch_paths = file_paths[i:i + batch_size]

            # dd here, alter SpectraDataset

            # Create a dataset and data loader for this batch
            dataset = SpectraDataset(batch_paths, label)
            loader = DataLoader(
                dataset,
                batch_size=dl_batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=(device.type == "cuda"),
                persistent_workers=(num_workers > 0),
            )

            detect_ids, cnn_scores = [], []

            # Disable gradient calculation during inference
            with torch.no_grad():
                for x, _, did in loader:
                    # Input shape: (batch_size, channels, height, width)
                    x = x.to(device, non_blocking=True)
            
                    # Compute model outputs as logits
                    logits = model(x)    
                    logits = logits.squeeze(1)       

                    # Convert logits to probabilities using sigmoid
                    probs = torch.sigmoid(logits).detach().cpu().numpy()
                    
                    # Convert detectid tensor to a Python list if needed
                    if torch.is_tensor(did):
                        did = did.cpu().tolist()
            
                    # Store detectids and CNN scores
                    detect_ids.extend(did)
                    cnn_scores.extend(probs.tolist())
                                
            # Sort the table by detectid                                
            table = Table({'detectid': detect_ids, f'CNN_Score_2D_Spectra_Fold_{fold_num+1}': cnn_scores})
            table.sort('detectid')
            output_tables.append(table)    
        
        # Stack all batch-level tables for this fold
        data_table = vstack(output_tables)
        data_tables.append(data_table)
        
    
    #---------------------------------------------------
    # Merge results from all folds
    merged_table = data_tables[0]
    for table in data_tables[1:]:
        merged_table = join(merged_table, table, keys='detectid', join_type='left')
    
    # Compute the average CNN score across all folds
    fold_columns = [f'CNN_Score_2D_Spectra_Fold_{i}' for i in range(1, folds+1)]
    merged_table[f'CNN_Score_2D_Spectra'] = np.mean([merged_table[col] for col in fold_columns], axis=0)
    return merged_table

