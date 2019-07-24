import os
import torchvision.transforms as transforms
import torch.utils.data as data
import PIL


from dataloader.image_folder import make_dataset

class UnalignedDataset(data.Dataset):
    def __init__(self):
        super(UnalignedDataset, self).__init__()

    def name(self):
        return 'UnalignedDataset'

    def __len__(self):
        return max(self.A_size, self.B_size)
    
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

        self.A_paths = make_dataset(self.dir_A, domain='A')
        self.B_paths = make_dataset(self.dir_B, domain='B')

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        transform_list = []
        #resize the input to 128 * 128
        osize = [opt.imgSize, opt.imgSize]
        transform_list.append(transforms.Resize(osize, PIL.Image.BICUBIC))
        transform_list += [transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5),
                                                        (0.5, 0.5, 0.5))]
        self.transform = transforms.Compose(transform_list)

        def __getitem__(self, index):
            A_path = self.A_paths[index % self.A_size]
            B_path = self.B_paths[index % self.B_size]

            A_img = PIL.Image.open(A_path).convert('RGB')
            B_img = PIL.Image.open(B_path).convert('RGB')
            
            A_img = self.transform(A_img)
            B_img = self.transform(B_img)

            return {'A': A_img, 'B': B_img,
                    'A_paths': A_path, 'B_paths': B_path}

        
            
        

