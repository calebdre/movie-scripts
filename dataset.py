from torch.utils.data import Dataset, DataLoader
import torch
import random

class TBTTScriptsDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        return self.decouple_batch(self.batches[key])
            
    def get_loader(self, batch_size = 32):
        self.batches = self.batchify(batch_size)        
        return self
        
    def batchify(self, batch_size):
        combinations = {}
        for i, (script, score) in enumerate(zip(self.data, self.labels)):
            key = len(script)
            val = (torch.tensor(script, dtype = torch.long), score)
            
            if key in combinations.keys():
                combinations[key].append(val)
            else:
                combinations[key] = [val]
        
        batches = []
        for key in combinations.keys():
            comb = combinations[key]
            comb_len = len(comb)
            
            if comb_len >= batch_size:
                num_comb_batches = comb_len / batch_size
                comb_batches = np.array_split(comb, num_comb_batches)
                
                random.shuffle(comb_batches)
                for comb_batche in comb_batches:
                    batches.append(comb_batche)
            else:
                random.shuffle(comb)
                batches.append(comb)

        return batches
    
    def decouple_batch(self, batch):
        scripts = [datum[0] for datum in batch]
        labels = [datum[1] for datum in batch]
        
        return torch.stack(scripts) , torch.tensor(labels, dtype=torch.float)
    
class ScriptsDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        return (self.data[key], self.labels[key])
    
    def get_loader(self, batch_size = 32):
        return DataLoader(
            dataset = self,
            batch_size = batch_size,
            collate_fn = self.collate,
            shuffle = True
        )
    
    def collate(self, batch):
        data_list = []
        label_list = []
        length_list = []

        for datum in batch:
            label_list.append(datum[1])
            length_list.append(len(datum[0]))
            data_list.append(torch.tensor(datum[0]))
            
        data_list = pad_sequence(data_list, batch_first = True)
        sorted_length_list, sorted_idxs = torch.sort(torch.tensor(length_list), descending = True)
        data_list = data_list[sorted_idxs]
        label_list = torch.tensor(label_list)[sorted_idxs]
        
        return data_list, sorted_length_list, label_list