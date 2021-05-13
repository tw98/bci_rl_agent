import os
import numpy as np
import nibabel as nib
import pickle
import phate
# import scprep
import scipy
import sklearn.manifold
import sklearn.decomposition
import itertools
import matplotlib.pyplot as plt
from nilearn.input_data import NiftiMasker
from nilearn.image import math_img
from scipy.stats import zscore
import re


class DataLoader():
    def __init__(self, 
                voxel_nii_files: [str], 
                roi_nii_path: str):
        self.roi_nii = nib.load(roi_nii_path)
        self.voxel_nii_files = voxel_nii_files
        self.files_loaded = 0

    def mask_ROI_cluster(self, voxel_nii, cluster_id, do_zscore=True):
        if cluster_id is None:
            mask_nii = math_img("img > 0", img = self.roi_nii)
        else:
            mask_nii = math_img("img == {}".format(cluster_id), img = self.roi_nii)
            
        masker = NiftiMasker(mask_img=mask_nii)
        masked_voxels = masker.fit_transform(voxel_nii)
        
        if do_zscore:
            masked_voxels = zscore(masked_voxels, axis=0, nan_policy='raise')

        return masked_voxels


    def get_cluster_labels(self):
        R = self.roi_nii.get_fdata()
        values = np.unique(R[R > 0])
        
        return values.astype(int)
    

    def get_state_pairs(self, voxels, n_prev_states):
        prev_states = []
        cur_states = []
        n_timepoints = voxels.shape[0]
        
        for i in range(n_prev_states, n_timepoints):
            current_state = voxels[i, :]
            current_state = np.expand_dims(current_state, axis=0)
            cur_states.append(current_state)
            
            prevs = voxels[i-n_prev_states:i, :]
            prev_states.append(prevs)
                    
        return np.array(prev_states), np.array(cur_states)


    def get_state_slices(self, voxels, n_prev_states):
        slices = []
        n_timepoints = voxels.shape[0]
        
        for i in range(n_prev_states, n_timepoints):
            state = voxels[i-n_prev_states:i+1, :]
            slices.append(state)

        return np.array(slices)

    def get_run_id(self, nii_file):
        run_str = re.findall('run-\d+', nii_file)
        
        if run_str:
            run_id = int(run_str[0].split('-')[1])
        else:
            run_id = self.files_loaded
        
        self.files_loaded += 1
        return run_id

    # creates a dataset for the predicition task
    # In this version, the runs are loaded and preprocessed separatly
    # For each run, the respective data samples are extracted
    # These data samples are concatenated together in the end
    def get_state_dataset(self, cluster_id, n_prev_states, run_ids=False, 
                            do_zscore=True, do_pca=False, n_comps=50):
        states_list = []

        # load in, preprocess, and extract data samples for each run 
        for nii_file in self.voxel_nii_files:
            print(f'loading data from {nii_file}...')
            voxel_nii = nib.load(nii_file) 
        
            voxels = self.mask_ROI_cluster(voxel_nii, cluster_id, do_zscore)

            # preproces the data

            # if do_phate:
            #     print('PHATE Preprocessing...')
            #     phate_operator = phate.PHATE(n_components=n_comps)
            #     voxels = phate_operator.fit_transform(voxels)

            if do_pca:
                print('PCA Preprocessing...')
                pca_operator = sklearn.decomposition.PCA(n_components=n_comps)
                voxels = pca_operator.fit_transform(voxels)

            # extract data samples
            states = self.get_state_slices(voxels, n_prev_states)

            # add run_id to data samples
            if run_ids:
                run_id = self.get_run_id(nii_file)
                id_row = np.full(states.shape[-1], run_id)
                # create new state sample array with run_id row added as first row
                # for sample in states:
                #     print(sample.shape)
                #     print(id_row.shape)
                states = np.array([np.vstack((id_row, sample)) for sample in states])
                

            states_list.append(states)
        
        if len(states_list) > 0:
            dataset = np.concatenate(tuple(states_list), axis=0)
        else:
            dataset = states_list[0]

        return dataset

    # creates a dataset for the predicition task
    # In this version, the runs are preprocessed together (embedded in the subspace)
    # Then, the respective data samples for each run are extracted 
    def get_state_dataset2(self, cluster_id, n_prev_states, run_ids=False, 
                            do_zscore=True, do_pca=False, n_comps=50):

        states_list = []

        run_voxels = []
        idx_runs = [0]
        total_timesteps = 0
        run_ids_list = []

        # load in data and combine into one large np array  
        for nii_file in self.voxel_nii_files:
            print(f'loading data from {nii_file}...')
            voxel_nii = nib.load(nii_file) 
        
            voxels = self.mask_ROI_cluster(voxel_nii, cluster_id, do_zscore)
            run_voxels.append(voxels)
            total_timesteps += voxels.shape[0]
            idx_runs.append(total_timesteps)
            
            run_ids_list.append(self.get_run_id(nii_file))
            
            print(f'Run {run_ids_list[-1]} - size of data: {voxels.shape}')
            # print(idx_runs)
            # print(run_ids_list)
            # print(len(run_voxels))
            # print(total_timesteps)
            
        all_voxels = np.concatenate(tuple(run_voxels), axis=0)
        print(f'size of data when all voxels combined {all_voxels.shape}')

        # preproces the data
        if do_pca:
            print('PCA Preprocessing...')
            pca_operator = sklearn.decomposition.PCA(n_components=n_comps)
            all_voxels = pca_operator.fit_transform(all_voxels)

        assert(len(run_ids_list) == (len(idx_runs) - 1))
        assert(total_timesteps == all_voxels.shape[0])
        
        # separate runs again and extract data samples 
        for i in range(len(run_ids_list)):
            start, end = idx_runs[i], idx_runs[i+1]
            states = self.get_state_slices(all_voxels[start:end, :], n_prev_states)

            if run_ids:
                id_row = np.full(states.shape[-1], run_ids_list[i])
                states = np.array([np.vstack((id_row, sample)) for sample in states])

            states_list.append(states)
        
        if len(states_list) > 0:
            dataset = np.concatenate(tuple(states_list), axis=0)
        else:
            dataset = states_list[0]
            
        return dataset
        
    def get_pandas_df(self, cluster_id, n_prev_states, run_ids=True):
        return