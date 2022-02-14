import numpy as np
import pandas as pd
import os


# Reading the data from the given data folder 
def read_dataset(data_root_path, split):
    folder_names = ['Albatross', 'frangipani', 'Marigold', 'anthuriam', 'Red_headed_Woodpecker', 'American_Goldfinch'] 
    folder_paths = [os.path.join(data_root_path, fn + '_' + split) for fn in folder_names] 

    data_frame = []
    for id in range(len(folder_paths)):
        cn = folder_names[id] 
        folder_path = folder_paths[id]

        for file_ in os.listdir(folder_path):
            if (file_[-4:] == '.jpg'):
                fp = os.path.join(folder_path, file_)    
                elements = [fp, cn, id]
                data_frame.append(elements)

    data_frame = np.array(data_frame)
    pd_df = pd.DataFrame(data = data_frame, columns = ['fp', 'cn', 'cid'])
    
    print("data frame, pandas: ", pd_df.shape)
    print("data head: ",  pd_df.head())

    data_file_name = './data_files/' + split + '.csv'
    pd_df.to_csv(data_file_name)


# This function will traverse through all the images from the dataset and create a single csv of all the data files
def process_data():
    data_root_path = './assignment1_data/'
    read_dataset(data_root_path, 'train')
    read_dataset(data_root_path, 'test')


if __name__ == "__main__":
    process_data()
