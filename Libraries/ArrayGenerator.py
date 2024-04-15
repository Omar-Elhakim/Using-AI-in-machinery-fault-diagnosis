import numpy as np

dataset_categories = {
    1 : "motor",        # Classifying motor faults                          (binary)
    2 : "bearing",      # Classifying bearing faults                        (binary)
    3 : "flow",         # Classifying flow faults                           (binary)
    4 : "lxc",          # Claasifying LXC faults                            (binary)
    5 : "mixed",        # Classifying LXC, Unbalance, and Base problems     (multiclass multilabel)
    6 : "coupling",     # Classifying coupling related faults               (binary)
    7 : "all",          # Classifying all of the above                      (multiclass multilabel)
    8 : "status"        # Classifying status                                (multiclass)
}

dataset_descriptions = {
    1 : {"Faults classified" : "AirGap, Electrical and SoftFoot problems", "Channels taken" : "Motor channels only", "Labels array" : "Binary"},
    2 : {"Faults classified" : "BearingFault, BearingWear, BearingLubrication", "Channels taken" : "According to label", "Labels array" : "Binary"},
    3 : {"Faults classified" : "Flow problem", "Channels taken" : "Machine channels only", "Labels array" : "Binary"},
    4 : {"Faults classified" : "LXC, XC, Losseness", "Channels taken" : "All", "Labels array" : "Binary"},
    5 : {"Faults classified" : "LXC, Unbalance, BaseProblems", "Channels taken" : "All", "Labels array" : "multiclass multilabel"},
    6 : {"Faults classified" : "Coupling problems", "Channels taken" : "Drive end channels only", "Labels array" : "Binary"},
    7 : {"Faults classified" : "All", "Channels taken" : "All", "Labels array" : "multiclass multilabel"},
    8 : {"Faults classified" : "Status", "Channels taken" : "All", "Labels array" : "multiclass"}

}

def separate_data(ts, labels, split_method):
    """
    A function that separates the ts arrays by channel, bearing, machine or none.
    Args:
        ts: 3D numpy array of time series data,
        labels: Associated labels
        by: split method (Options: channel, bearing, machine, none)
    Returns:
        A tuple of new split 3D array and their associated labels.
    """
    _, num_channels, num_points = ts.shape

    if split_method == "channel":
        new_ts = ts.reshape(-1, 1, num_points)
        new_labels = np.repeat(labels, repeats=num_channels, axis=0)
    
    elif split_method == "bearing":
        new_ts = ts.reshape(-1, 3, num_points)
        new_labels = np.repeat(labels, repeats=num_channels // 3, axis=0)

    elif split_method == "machine":
        new_ts = ts.reshape(-1, 6, num_points)
        new_labels = np.repeat(labels, repeats=num_channels // 6, axis=0)

    elif split_method == None:
        new_ts = ts
        new_labels = np.array(labels)
    
    else:
        raise Exception ("Invalid split type: choose from [channel, bearing, machine, None]")
    
    return new_ts, new_labels



def generate_dataset_array(dataset_category, ts, labels, split_method):
    """
    A function that generates the 3D time series arrays and their associated labels for the dataset categories described above.
    Args:
        dataset_category: int (from 1 to 7)
        ts: 3D time series array
        labels: the associated labels
        split_method: the method to split the data by (channel, bearing, machine, or None)
    
    Returns:
        A tuple of new 3D time series array and their associated labels
    """
    if dataset_category == 1: 
        faults = ["AirGap", "SoftFoot", "ElectricalProblem"]
        new_labels = [1 if any(motor_fault in fault for fault in faults_list for motor_fault in faults) else 0 for faults_list in labels]
        new_ts = ts[:, :6, :]

    elif dataset_category == 2:
        faults = ["BearingFault", "BearingWear", "BearingLubrication"]
        new_labels = np.zeros((len(labels), 2))

        for major_index, faults_list in enumerate(labels):
            for fault in faults_list:
                fault_name, fault_location = fault.split(',')
                if any(bearing_fault == fault_name for bearing_fault in faults):
                    if fault_location == "motor":
                        new_labels[major_index][0] = 1
                    else:
                        new_labels[major_index][1] = 1
        new_ts_1 = ts[:, :6, :]
        new_labels_1 = new_labels[:, 0]

        new_ts_2 = ts[:, 6:, :]
        new_labels_2 = new_labels[:, 1]
        
        new_labels = np.concatenate([new_labels_1, new_labels_2], axis=0)
        new_ts = np.concatenate([new_ts_1, new_ts_2], axis=0)

    elif dataset_category == 3:
        faults = ["Flow"]
        new_labels = [1 if any(flow_fault in fault for fault in faults_list for flow_fault in faults) else 0 for faults_list in labels]
        new_ts = ts[:, 6:, :]

    elif dataset_category == 4:
        faults = ["LXC", "Looseness", "XC"]
        new_labels = [1 if any(lxc_fault in fault for fault in faults_list for lxc_fault in faults) else 0 for faults_list in labels]
        new_ts = ts

    elif dataset_category == 5:
        faults_and_indices = {
            0 : ["LXC", "Looseness", "XC"],
            1 : ["BaseProblems"],
            2 : ["Unbalance"]
        }

        new_labels = np.zeros((len(labels), 3))
        for major_index, faults_list in enumerate(labels):
            for fault in faults_list:
                fault_name, fault_location = fault.split(',')
                for index, fault_names in faults_and_indices.items():
                    if fault_name in fault_names:
                        new_labels[major_index][index] = 1
        new_ts = ts


    elif dataset_category == 6:
        faults = ["Coupling", "BeltProblem"]
        new_labels = [1 if any(coupling_fault in fault for fault in faults_list for coupling_fault in faults) else 0 for faults_list in labels]
        new_ts = ts[:, 3:9, :]

    elif dataset_category == 7:
        faults_and_indices = {
            0 : ["AirGap", "SoftFoot", "ElectricalProblem"],
            1 : ["BearingFault", "BearingWear", "BearingLubrication"],
            2 : ["flow"],
            3 : ["LXC", "Looseness", "XC"],
            4 : ["BaseProblems"],
            5 : ["Unbalance"],
            6 : ["Coupling", "BeltProblem"]
        } 

        new_labels = np.zeros((len(labels), 7))
        for major_index, faults_list in enumerate(labels):
            for fault in faults_list:
                fault_name, fault_location = fault.split(',')
                for index, fault_cluster in faults_and_indices.items():
                    if fault_name in fault_cluster:
                        new_labels[major_index][index] = 1

        new_ts = ts

    elif dataset_category == 8:
        if type(labels[0]) == list:
            raise Exception ("Invalid labels for cateogry 8")
        new_ts = ts
        new_labels = labels

    else:
        raise Exception ("Invalid dataset cateogry, choose from [1, 2, 3, 4, 5, 6, 7]")

    print (dataset_descriptions[dataset_category])
    return separate_data(ts=new_ts, labels=new_labels, split_method=split_method)




def calculate_fft(signal):
    """Calculates the FFT for a single channeled time series.
        Used with apply_along_axis when applied to a two dimensional series.
    Args:
        signal: a numpy array time series signal (single channel)
    Returns:
        A numpy array containing the fft with length equal to half the length
        of the input time series array."""
    signal = signal * np.hanning(len(signal)) # Apply hanning window
    fft = np.fft.fft(signal)
    fft = np.abs(fft) # Take absolute value
    fft = fft[: len(fft)] # Take the first half only
    return fft




def apply_sliding_window(ts_3d, labels, window_size=1024, overlap_pct=0.1, with_fft=False):
    """Applies the sliding window on the ts 3d array and its associated labels.
    
    Args:
        ts_3d: 3d time series array
        labels: associated labels
        window_size: int, window size
        overlap_pct: float, overlap percentage between (0, 1)
        with_fft: boolean, applies the fft function
    Returns:
        a tuple of new windowed 3d times series array and its associated labels"""
    
    x_new = []
    y_new = []
    
    for ts, label in zip(ts_3d, labels):
        ts = ts[:, ~np.any(np.isnan(ts), axis=0)]
        num_channels, signal_length = ts.shape
        overlap = int(window_size * overlap_pct)
        stride = window_size - overlap
        num_windows = (signal_length - window_size) // stride + 1

        if with_fft == True:
            x_temp = np.zeros((num_windows, num_channels, window_size))
        else:
            x_temp = np.zeros((num_windows, num_channels, window_size))

        y_temp = []

        for i in range(num_windows):
            start = i * stride
            end = start + window_size
            if with_fft == True:
                x_temp[i] = np.apply_along_axis(calculate_fft, arr=ts[:, start:end], axis=1) 
            else:
                x_temp[i] = ts[:, start:end]
            y_temp.append(label)

        x_new.append(x_temp)
        y_new.append(y_temp)

    x_new, y_new = np.concatenate(x_new), np.concatenate(y_new)

    return x_new, y_new

    

    



