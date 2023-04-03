#pytorch
import torch
from torch.utils.data import random_split, DataLoader
#その他
import os
import wandb
from tqdm import tqdm
import numpy as np
import matplotlib as plt
#自作
from utils.dataset import LoadDataSet
from model.unet import UNet
from utils.score import DiceLoss, IoU


def save_ckp(checkpoint, is_best, checkpoint_path, best_model_path):
    #checkpointセーブする奴
    torch.save(checkpoint, checkpoint_path)
    if is_best:
        torch.save(checkpoint, best_model_path)
    

if __name__ == "__main__":
    #GPUが使えなかったら中止
    assert torch.cuda.is_available()
    device = torch.device("cuda")

    #保存先準備
    os.makedirs("./checkpoint/", exist_ok=True)

    #データセットロード
    TRAIN_PATH = "./data/stage1_train"
    train_dataset = LoadDataSet(TRAIN_PATH)

    #学習用と評価用に分割
    split_ratio = 0.25
    train_size=int(np.round(train_dataset.__len__()*(1 - split_ratio)))
    valid_size=int(np.round(train_dataset.__len__()*split_ratio))

    train_data, valid_data = random_split(train_dataset, [train_size, valid_size])
    
    #データローダー
    train_loader = DataLoader(dataset=train_data, batch_size=10, shuffle=True)
    val_loader = DataLoader(dataset=valid_data, batch_size=10)
 
    print("Length of train data: {}".format(len(train_data)))
    print("Length of validation data: {}".format(len(valid_data)))

    #モデル準備
    model = UNet(3,1).cuda() #.cuda()はGPU限定, .to(device)はdeviceにGPUやCPUを指定できる
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
    criterion = DiceLoss()
    acurracy_metric = IoU()
    num_epochs = 20
    valid_loss_min = np.Inf #無限って

    total_train_loss  = []
    total_train_score = []
    total_valid_loss  = []
    total_valid_score = []

    #experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    
    losses_value = 0

    for epoch in range(num_epochs):
        
        #訓練
        train_loss  = []
        train_score = []
        valid_loss  = []
        valid_score = []

        pbar = tqdm(train_loader, desc = "descriptionZ") #progress bar

        for x_train, y_train in pbar:
            #tensorをGPUメモリへ
            x_train = (x_train).cuda()
            y_train = (y_train).cuda()

            #損失計算
            optimizer.zero_grad()
            output = model(x_train)
            loss = criterion(output, y_train)
            losses_value = loss.item()

            #精度評価
            score = acurracy_metric(output, y_train)

            #配列に損失と精度を保存
            train_loss.append(losses_value)
            train_score.append(score.item())

            # experiment.log({
            #             "train loss": losses_value,
            #             "train_score":score.item(),
            #             "epoch": epoch
            #         })

            #バックプロパゲーション
            loss.backward()
            optimizer.step()

            pbar.set_description(f"Epoch: {epoch+1}, loss: {losses_value}, IoU: {score}")

        #評価
        with torch.no_grad():
            for img, mask in val_loader:
                #tensorをGPUメモリへ
                img = img.cuda()
                mask = mask.cuda()
                
                #損失計算
                output = model(img)
                loss = criterion(output, mask)
                losses_value = loss.item()

                #精度評価
                score = acurracy_metric(output, mask)

                #配列に損失と精度を保存
                valid_loss.append(losses_value)
                valid_score.append(score.item())

                # experiment.log({
                #         "valid loss": losses_value,
                #         "valid_score":score.item(),
                #         "epoch": epoch
                #     })

        #各配列にはこのepochでのイテレータのバッチごとのlossとscoreが入っている
        #最後尾にバッチごとのデータを平均し、今回のepochでの平均のlossとscoreを保存
        total_train_loss.append(np.mean(train_loss))
        total_train_score.append(np.mean(train_score))
        total_valid_loss.append(np.mean(valid_loss))
        total_valid_score.append(np.mean(valid_score))
        
        print(f"Train Loss: {total_train_loss[-1]}, Train IOU: {total_train_score[-1]}")
        print(f"Valid Loss: {total_valid_loss[-1]}, Valid IOU: {total_valid_score[-1]}")

        checkpoint = {
            "epoch": epoch,
            "valid_loss_min": total_valid_loss[-1],
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }

        checkpoint_path = "./checkpoint/checkpoint_"+str(epoch)+".pt"
        best_model_path = "./checkpoint/bestmodel.pt"
        save_ckp(checkpoint, False, checkpoint_path, best_model_path)
    
        # 評価データにおいて最高精度のモデルのcheckpointの保存
        if total_valid_loss[-1] <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,total_valid_loss[-1]))
            save_ckp(checkpoint, True, checkpoint_path, best_model_path)
            valid_loss_min = total_valid_loss[-1]

        # experiment.log({
        #                 "total_train_loss": total_train_loss[-1],
        #                 "total_train_score": total_train_score[-1],
        #                 "total_valid_loss":total_valid_loss[-1],
        #                 "total_valid_score":total_valid_score[-1],
        #                 "epoch": epoch
        #             })
        
        print("")

    plt.plot(range(num_epochs), total_train_loss)
    plt.plot(range(num_epochs), total_valid_loss, c='#00ff00')
    plt.xlim(0, num_epochs)
    plt.ylim(0, 2.5)
    plt.xlabel('EPOCH')
    plt.ylabel('LOSS')
    plt.legend(['train loss', 'test loss'])
    plt.title('loss')
    plt.savefig("loss_image.png")
    plt.clf()

    plt.plot(range(num_epochs), total_train_score)
    plt.plot(range(num_epochs), total_valid_score, c='#00ff00')
    plt.xlim(0, num_epochs)
    plt.ylim(0, 1)
    plt.xlabel('EPOCH')
    plt.ylabel('ACCURACY')
    plt.legend(['train acc', 'test acc'])
    plt.title('accuracy')
    plt.savefig("accuracy_image.png")

    