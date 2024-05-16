import os
import pandas as pd
import argparse
import time


class CSVMerger:
    def __init__(self, input_dir, output_file):
        self.input_dir = input_dir
        self.output_file = output_file
        self.directories = ['person', 'anatomy', 'doremus',
                            'restaurant', 'SPIMBENCH_small-2019', 'SPIMBENCH_large-2016']

    def merge(self):
        # Get a list of all the .csv files in the input directory
        csv_files = []
        for directory in self.directories:
            csv_files += [os.path.join(self.input_dir+directory, f) for f in os.listdir(
                self.input_dir+directory) if f.endswith('train.csv')]
        print(csv_files)
        # Read each CSV file into a pandas DataFrame
        dataframes = [pd.read_csv(f) for f in csv_files]

        # Concatenate all the DataFrames into a single DataFrame
        merged_df = pd.concat(dataframes, ignore_index=True)

        # Write the merged DataFrame to a new CSV file
        merged_df.to_csv(self.output_file, index=False)


if __name__ == "__main__":

    def arg_manager():
        parser = argparse.ArgumentParser()
        parser.add_argument("--input_path", type=str, default="./outputs/")
        parser.add_argument("--output_path", type=str,
                            default="./outputs/merged/")
        return parser.parse_args()

    start = time.time()
    args = arg_manager()
    input_dir = args.input_path
    output_dir = args.output_path
    output_file = output_dir + 'merged_train_data.csv'

    csv_merger = CSVMerger(input_dir, output_file)
    csv_merger.merge()
    print('Running Time : ', (time.time() - start), ' seconds ')
    print('\n \n')
# already done if you want to repeat the process, update the variable and delete the old 'train.csv file in the directory './outputs/merged/'.
