import torch.utils.data
from dataloader.dataset import UnalignedDataset
def CreateDataLoader(opt):
    dataloader = DataLoaderDataset()
    dataloader.initialize(opt)
    return dataloader

def CreateDataset(opt):
    dataset = None
    dataset = UnalignedDataset()
    print("Dataset %s creatd!" % (dataset.name()))
    dataset.initialize(opt)
    return dataset


class DataLoaderDataset():
    def initialize(self, opt):
        self.opt = opt
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads)
        )

    def load_data(self):
        return self.dataloader

