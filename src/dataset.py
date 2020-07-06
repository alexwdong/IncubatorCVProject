import os
import glob
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image

def generate_label(image_dir,label_path, label_col = 0):
  '''
  label_path(string): Path to the csv file with labels.
  image_dir (string): Directory with all the images.
  label_col: column of image id, default to 0th column in the label.csv
  ''' 
  label_csv = pd.read_csv(label_path,index_col = label_col)
  # create dictionary mapping label to label index
  lab2idx = {lab:idx for idx,lab in enumerate(list(set(label_csv['breed'])))}
  # create dictionary mapping label index back to label
  idx2lab = {idx:lab for idx,lab in enumerate(list(set(label_csv['breed'])))}
  # list of image pathes relative to root directory. Eg. format: subdirectory/image.jpg
  path = ['/'.join(img_fullname.split('/')[-2:]) for root, dirs, files in os.walk(image_dir)
                       for dir in dirs
                       for img_fullname in glob.glob(os.path.join(root,dir,'*'))]
  label_idx = [lab2idx[label_csv.loc[im.split('/')[-1].split('.')[0]][0]] for im in path]
  
  label = pd.DataFrame(columns = ['path','label_idx'])

  label['path'] = path
  label['label_idx'] = label_idx
          
  return label, lab2idx, idx2lab


class DogDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, label_csv, root_dir, transform=None):
        """
        Args:
            csv_file (file): Modified csv file with image path and label indexes. 
                             Refer to function 'generate_label' 
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.label_csv = label_csv
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.label_csv)

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()

        img_path = os.path.join(self.root_dir,
                                self.label_csv.iloc[idx, 0])
        image = Image.open(img_path)
        label = self.label_csv.iloc[idx, 1]
       
        if self.transform:
            image = self.transform(image)
        

        return image,label
