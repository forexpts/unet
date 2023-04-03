import torch
from torch.utils.data import random_split, DataLoader
#その他
import numpy as np
import matplotlib as plt
import random
#自作
from utils.dataset import TestDataSet
from model.unet import UNet


if __name__ == "__main__":
    device = torch.device("cuda")

    N = 3
    TEST_DATAPATH = "./data/stage_test/"
    dataset = TestDataSet(TEST_DATAPATH)
    print(dataset.__len__())
    nums = random.sample(range(0, dataset.__len__()), N)

    LOAD_PATH = "./checkpoint/bestmodel.pt"
    model = UNet(3,1)
    model.load_state_dict(torch.load(LOAD_PATH)["state_dict"])
    model.to(device)

    figure, ax = plt.pyplot.subplots(nrows=len(nums), ncols=2, figsize=(5, 8))

    ax[0, 0].set_title("Image")
    ax[0, 1].set_title("Mask")

    model.eval()
    with torch.no_grad():
        for i, img_num in enumerate(nums):
            img = dataset.__getitem__(img_num)
            img = img.to(device)
            img = img.unsqueeze(0) 
            mask = model(img)
            img = img.squeeze().cpu().numpy().transpose((1,2,0))
            mean=np.array((0.485, 0.456, 0.406))
            std=np.array((0.229, 0.224, 0.225))
            img  = std * img + mean
            img = img*255
            img = img.astype(np.uint8)
            print(mask.size())
            mask = mask.squeeze(dim=0).cpu().numpy().transpose((1,2,0))


            ax[i, 0].text(-0.2, 0.5, f"{img_num}", fontsize=12, ha='right', va='center', transform=ax[i, 0].transAxes)
            ax[i, 0].imshow(img)
            ax[i, 1].imshow(mask)
            ax[i, 0].set_xlabel(img_num)
            ax[i, 0].set_axis_off()
            ax[i, 1].set_axis_off()
    
    #plt.tight_layout()
    plt.pyplot.savefig("test.png")
