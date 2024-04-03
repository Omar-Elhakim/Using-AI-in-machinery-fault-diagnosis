# Create a class that takes as input the type of measurement to be used "VHR" , "VLR"
import pandas as pd
import numpy as np
import os


class PreLoader:
    """Handles files loading and datasets preparation to get passed into the DataPreparation class"""

    parent_folder = r"D:\ai-vibration-diagnostics\datasets"

    def __init__(self, type):
        self.type = type
        if self.type == "vhr":
            self.df = pd.read_csv(
                r"D:\ai-vibration-diagnostics\datasets\vhr\metadata.csv"
            )

        elif self.type == "vlr":
            self.df = pd.read_csv(
                r"D:\ai-vibration-diagnostics\datasets\vlr\metadata.csv"
            )

        elif self.type == "both":
            df_hr = pd.read_csv(
                r"D:\ai-vibration-diagnostics\datasets\vhr\metadata.csv"
            )
            df_lr = pd.read_csv(
                r"D:\ai-vibration-diagnostics\datasets\vlr\metadata.csv"
            )
            self.df = pd.concat([df_hr, df_lr]).reset_index(drop=True)

        else:
            raise Exception("undefined measurement type")

        # Eval columns
        self.df["standardized_faults"] = self.df["standardized_faults"].apply(
            lambda x: eval(x)
        )
        self.df["rpm"] = self.df["rpm"].apply(lambda x: eval(x))
        self.df["bearing_abbs"] = self.df["bearing_abbs"].apply(lambda x: eval(x))

        # Standardize status
        self.df["status"] = self.df["status"].apply(lambda x: x.strip())

        # Flatten faults
        self.df["standardized_faults"] = self.df["standardized_faults"].apply(
            lambda xss: [x for xs in xss for x in xs]
        )
        self.df["standardized_faults"] = self.df["standardized_faults"].apply(
            lambda x: [fault for fault in x if fault != ""]
        )
        self.df["standardized_faults"] = self.df["standardized_faults"].apply(
            lambda x: [fault for fault in x if fault != "NAF"]
        )
        self.df["standardized_faults"] = self.df["standardized_faults"].apply(
            lambda lst: [",".join(y.split(",")[:2]) for y in lst]
        )

        # Sort out speeds
        self.df["rpm"] = (
            self.df["rpm"]
            .explode()
            .apply(lambda x: x.replace("rpm", "").replace(",", "").strip())
        )
        self.df["rpm"] = self.df["rpm"].apply(lambda x: float(x))

    def sort_dict(ts_path_dictionary):
        return dict(sorted(ts_path_dictionary.items(), reverse=False))

    def get_ts_files_paths(self):
        """Returns a list of paths (ts, metadata)"""
        ts_files_paths = {}
        for filename in os.listdir(PreLoader.parent_folder):
            if self.type in filename:
                folder_to_loop_inside = os.path.join(PreLoader.parent_folder, filename)
                for filename in os.listdir(folder_to_loop_inside):
                    if "metadata" in filename:
                        continue
                    file_bearing_index = int(filename.split(".")[0])
                    ts_files_paths[file_bearing_index] = os.path.join(
                        folder_to_loop_inside, filename
                    )

                break

        ts_files_paths = PreLoader.sort_dict(ts_path_dictionary=ts_files_paths)
        return ts_files_paths

    def get_3d_ts_array(self):
        """Returns a 3D time series array"""
        ts_file_paths = PreLoader.get_ts_files_paths(self).values()
        ts_list = []
        bearing_all_ts = [np.array(pd.read_csv(path)) for path in ts_file_paths]
        for i in range(len(self.df)):
            sample_ts = np.array([arr[i] for arr in bearing_all_ts])
            ts_list.append(sample_ts)

        return np.array(ts_list)

    def get_labels_array(self, labels_type):
        if labels_type == "faults":
            return np.array(self.df["standardized_faults"])
        elif labels_type == "status":
            return np.array(self.df["status"])
        else:
            raise Exception("Invalid labels type, chooise between faults and status")

    def set_faults_as_one(self, faults_to_combine, target_fault):
        for index, row in self.df.iterrows():
            old_faults_list = row["standardized_faults"]
            new_temp_faults_list = []
            for old_fault in old_faults_list:
                if any(
                    fault_to_combine in old_fault
                    for fault_to_combine in faults_to_combine
                ):
                    new_fault = target_fault + "," + old_fault.split(",")[1]
                    new_temp_faults_list.append(new_fault)
                else:
                    new_temp_faults_list.append(old_fault)
            self.df.at[index, "standardized_faults"] = new_temp_faults_list
