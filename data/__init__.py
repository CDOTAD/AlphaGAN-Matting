from data.input_dataset import InputDataset
import torch.utils.data


class AlphaGANDataLoader(object):
    def __init__(self, opt):
        self.dataset = InputDataset(opt.dataroot)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            shuffle=True,
            batch_size=opt.batch_size,
            num_workers=4,
            drop_last=True
        )

    def load_data(self):
        return self

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data