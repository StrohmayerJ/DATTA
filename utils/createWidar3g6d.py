import numpy as np
import os
import csiread
import pickle
import argparse
from tqdm import tqdm

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Widar3-g6 dataset structure:
# 20181130 (room 1) Users: 5,10,11,12,13,14,15,16,17
# 20181204 (room 2) Users: 1
# 20181209 (room 2) Users: 2,6
# 20181211 (room 3) Users: 3,7,8,9

# Train-Test domain split:
# TEST:
# 0: room 1 user 5
# 1: room 1 user 10
# 2: room 1 user 11
# 3: room 1 user 12
# 4: room 1 user 13
# 5: room 1 user 14
# 6: room 1 user 15
# 7: room 1 user 16
# 8: room 1 user 17

# TRAIN:
# 9: room 2 user 1
# 10: room 2 user 2
# 11: room 2 user 6
# 12: room 3 user 3
# 13: room 3 user 7
# 14: room 3 user 8
# 15: room 3 user 9

# Function to zero-pad the CSI data
def zero_padding(data, T_MAX):
    data_pad = []
    for i in range(len(data)):
        current_shape = np.array(data[i]).shape
        pad_width = ((0, 0), (0, T_MAX - current_shape[1]))
        data_pad.append(np.pad(data[i], pad_width, 'constant', constant_values=0).tolist())
    res = np.array(data_pad)
    return res

def createSplit(opt):

    # Check if opt.mode is TRAIN or TEST
    if opt.mode not in ['TRAIN', 'TEST']:
        print('Invalid mode. Please select TRAIN or TEST.')
        return  # Exit the function if mode is invalid

    T_MAX = 0  # Max length of samples
    SELECTED_LABELS = [1, 2, 3, 4, 5, 6]
    SELECTED_RX = [1, 2, 3, 4, 5, 6]
    data_cache = os.path.join(opt.data,'widar3-g6_csi_domain_train_cache.pkl' if opt.mode == 'TRAIN' else 'widar3-g6_csi_domain_test_cache.pkl')

    # Read CSI and labels
    csiComplex = []
    activity_labels = []
    user_labels = []
    environment_labels = []
    domain_labels = []
    sample_lengths = []

    # Gather all files first to set up the progress bar
    all_files = []
    for data_root, data_dirs, data_files in os.walk(opt.data):
        for data_file_name in data_files:
            file_path = os.path.join(data_root, data_file_name)
            if 'cache' in file_path:
                continue
            all_files.append((data_root, data_file_name))

    total_files = len(all_files)
    print(f"Total files to process: {total_files}")

    for data_root, data_file_name in tqdm(all_files, desc="Processing CSI files", unit="file"):
        file_path = os.path.join(data_root, data_file_name)

        # Determine environment label
        if '20181130' in file_path:
            if opt.mode == 'TRAIN':
                continue
            environment_label = 1
        elif '20181204' in file_path or '20181209' in file_path:
            if opt.mode == 'TEST':
                continue
            environment_label = 2
        else:
            if opt.mode == 'TEST':
                continue
            environment_label = 3

        # Read CSI data
        csidata = csiread.Intel(file_path, nrxnum=3, ntxnum=1, pl_size=10, if_report=False)
        csidata.read()
        csi = csidata.get_scaled_csi()[:, :, 0, 0]  # Select first antenna
        csi = np.transpose(csi, (1, 0))
        csi = csi[:, 0::10]  # Downsample from 1000Hz to 100Hz

        # Check if length is zero
        if csi.shape[1] == 0:
            continue

        # Normalize CSI data
        max_abs = np.max(np.abs(csi))
        if max_abs == 0:
            continue  # Avoid division by zero
        csi = csi / max_abs

        csi = csi.astype(np.complex64)  # Reduce from complex128 to complex64

        # Extract labels
        try:
            activity_label = int(data_file_name.split('-')[1])
            receiver_label = int(data_file_name.split('-')[5].split('.')[0][1:])
            user_label = int(data_file_name.split('-')[0][4:])
        except (IndexError, ValueError):
            continue  # Skip files with unexpected naming

        # Determine domain label
        if environment_label == 1:  # room 1
            if user_label == 5:
                domain_label = 0
            elif user_label == 10:
                domain_label = 1
            elif user_label == 11:
                domain_label = 2
            elif user_label == 12:
                domain_label = 3
            elif user_label == 13:
                domain_label = 4
            elif user_label == 14:
                domain_label = 5
            elif user_label == 15:
                domain_label = 6
            elif user_label == 16:
                domain_label = 7
            else:
                domain_label = 8
        elif environment_label == 2:  # room 2
            if user_label == 1:
                domain_label = 9
            elif user_label == 2:
                domain_label = 10
            else:
                domain_label = 11
        else:  # room 3
            if user_label == 3:
                domain_label = 12
            elif user_label == 7:
                domain_label = 13
            elif user_label == 8:
                domain_label = 14
            else:
                domain_label = 15

        # Select gesture
        if (activity_label not in SELECTED_LABELS):
            continue

        # Select RX
        if (receiver_label not in SELECTED_RX):
            continue

        # Update T_MAX
        length = csi.shape[1]

        if length < 120 or length > 220:
            continue

        if T_MAX < length:
            T_MAX = length

        # Save List
        csiComplex.append(csi)
        activity_labels.append(activity_label)
        environment_labels.append(environment_label)
        user_labels.append(user_label)
        domain_labels.append(domain_label)
        sample_lengths.append(length)

    # Zero-padding
    csiComplex = zero_padding(csiComplex, T_MAX)

    # labels
    a = np.array(activity_labels, dtype=np.int8)
    e = np.array(environment_labels, dtype=np.int8)
    u = np.array(user_labels, dtype=np.int8)
    d = np.array(domain_labels, dtype=np.int8)  

    # Save processed data
    data_to_save = {
        'csiComplex': csiComplex,
        'activities': a,
        'environments': e,
        'users': u,
        'domains': d,
        'T_MAX': T_MAX
    }

    unique_activities = np.unique(a)
    unique_environments = np.unique(e)
    unique_users = np.unique(u)
    unique_domains = np.unique(d)

    # Compute and print statistics
    for activity in unique_activities:
        print('Activity:', activity, 'Number of samples:', len(np.where(a == activity)[0]))
    for environment in unique_environments:
        print('Environment:', environment, 'Number of samples:', len(np.where(e == environment)[0]))
    for user in unique_users:
        print('User:', user, 'Number of samples:', len(np.where(u == user)[0]))
    for domain in unique_domains:
        print('Domain:', domain, 'Number of samples:', len(np.where(d == domain)[0]))

    print(f"{opt.mode} split number of samples: {len(csiComplex)}, with a max length of {T_MAX}")
    print('Unique activities:', unique_activities, 'Number of activities:', len(unique_activities))
    print('Unique environments:', unique_environments, 'Number of environments:', len(unique_environments))
    print('Unique users:', unique_users, 'Number of users:', len(unique_users))
    print('Unique domains:', unique_domains, 'Number of domains:', len(unique_domains))
    print('Mean sample length:', np.mean(sample_lengths))
    print('Std sample length:', np.std(sample_lengths))
    print('Median sample length:', np.median(sample_lengths))
    print('Min sample length:', np.min(sample_lengths))
    print('Max sample length:', np.max(sample_lengths))
    activity_distribution = []
    for activity in unique_activities:
        activity_distribution.append(len(np.where(a == activity)[0]) / len(a))
    print('Activity distribution:', activity_distribution)

    # Save the data using the highest pickle protocol for efficiency
    with open(data_cache, 'wb') as f:
        pickle.dump(data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/widar3g6d', help='Location of Widar3.0 data')
    parser.add_argument('--mode', default='TRAIN', help='Split type: TRAIN or TEST')
    opt = parser.parse_args()
    createSplit(opt)


'''
Activity: 1 Number of samples: 5199
Activity: 2 Number of samples: 3599
Activity: 3 Number of samples: 2245
Activity: 4 Number of samples: 4478
Activity: 5 Number of samples: 4467
Activity: 6 Number of samples: 4494
Environment: 2 Number of samples: 8564
Environment: 3 Number of samples: 15918
User: 1 Number of samples: 3974
User: 2 Number of samples: 747
User: 3 Number of samples: 3909
User: 6 Number of samples: 3843
User: 7 Number of samples: 3915
User: 8 Number of samples: 3859
User: 9 Number of samples: 4235
Domain: 9 Number of samples: 3974
Domain: 10 Number of samples: 747
Domain: 11 Number of samples: 3843
Domain: 12 Number of samples: 3909
Domain: 13 Number of samples: 3915
Domain: 14 Number of samples: 3859
Domain: 15 Number of samples: 4235
TRAIN split number of samples: 24482 with a max length of 220
Unique activities: [1 2 3 4 5 6] Number of activities: 6
Unique environments: [2 3] Number of environments: 2
Unique users: [1 2 3 6 7 8 9] Number of users: 7
Unique domains: [ 9 10 11 12 13 14 15] Number of domains: 7
Mean sample length: 170.7379299076873
Std sample length: 23.264258846075588
Median sample length: 171.0
Min sample length: 120
Max sample length: 220
Activity distribution: [0.2123601012989135, 0.1470059635650682, 0.09170002450780165, 0.18290989298259946, 0.18246058328567927, 0.1835634343599379]

Activity: 1 Number of samples: 6693
Activity: 2 Number of samples: 4867
Activity: 3 Number of samples: 4037
Activity: 4 Number of samples: 6441
Activity: 5 Number of samples: 5993
Activity: 6 Number of samples: 6135
Environment: 1 Number of samples: 34166
User: 5 Number of samples: 3965
User: 10 Number of samples: 3857
User: 11 Number of samples: 4334
User: 12 Number of samples: 3865
User: 13 Number of samples: 4023
User: 14 Number of samples: 3806
User: 15 Number of samples: 4078
User: 16 Number of samples: 3806
User: 17 Number of samples: 2432
Domain: 0 Number of samples: 3965
Domain: 1 Number of samples: 3857
Domain: 2 Number of samples: 4334
Domain: 3 Number of samples: 3865
Domain: 4 Number of samples: 4023
Domain: 5 Number of samples: 3806
Domain: 6 Number of samples: 4078
Domain: 7 Number of samples: 3806
Domain: 8 Number of samples: 2432
TEST split number of samples: 34166 with a max length of 220
Unique activities: [1 2 3 4 5 6] Number of activities: 6
Unique environments: [1] Number of environments: 1
Unique users: [ 5 10 11 12 13 14 15 16 17] Number of users: 9
Unique domains: [0 1 2 3 4 5 6 7 8] Number of domains: 9
Mean sample length: 174.31027922496048
Std sample length: 23.93955180094525
Median sample length: 176.0
Min sample length: 120
Max sample length: 220
Activity distribution: [0.19589650529766434, 0.14245156003043963, 0.11815840309079202, 0.18852075162442195, 0.17540830064976878, 0.1795644793069133]
'''