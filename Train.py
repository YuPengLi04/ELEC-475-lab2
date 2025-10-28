# train.py — SnoutNet.txt training only (no validation)
import argparse, random, torch, torch.nn as nn
import matplotlib.pyplot as plt
import math
from torch.utils.data import DataLoader
from torchvision import transforms
from model import SnoutNet
from oxfordDataset import OxfordPetNoses
from PretrainedModel import build_model


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_all(s=42):
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def make_loader(data_root, img_size, batch_size, augment, shuffle=True):
    tx = transforms.Compose([
        transforms.Resize((img_size, img_size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    if augment == 1:
        ds = OxfordPetNoses(root_dir=data_root,mode="train", keep_aspect=True, transform=tx, normalize=True,
                            aug_rotate=True, aug_hflip_p=0.5,
                            aug_color_jitter={"brightness": 0.2, "contrast": 0.2, "saturation": 0.2})
    else:
        ds = OxfordPetNoses(root_dir=data_root,mode="train", keep_aspect=True, transform=tx, normalize=True)

    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=False)


def train(epochs, model, optimizer, loss_fn, train_loader, scheduler, device, save_file, plot_file):
    model.train()  # stays in train mode the whole time
    loss_train = []
    for epoch in range(1, epochs + 1):
        running = 0.0
        for imgs, uv in train_loader:
            imgs = imgs.to(device).float()
            uv = uv.to(device).float()

            optimizer.zero_grad(set_to_none=True)
            preds = model(imgs)
            loss = loss_fn(preds, uv)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            running += loss.item()

        train_loss = running / len(train_loader)
        loss_train += [train_loss]

        print(f"[{epoch:02d}/{epochs}] train_loss={math.sqrt(train_loss):.4f}")
        scheduler.step(train_loss)

        # save checkpoint each epoch (or keep last only)
        if save_file:
            torch.save(model.state_dict(), save_file)

    if plot_file != None:
        plt.figure(2, figsize=(12, 7))
        plt.clf()
        plt.plot(loss_train, label='train')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(loc=1)
        print('saving ', plot_file)
        plt.savefig(plot_file)

    print("Training complete.")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--epochs", type=int, default=10)
    ap.add_argument("-b", "--batch_size", type=int, default=32)
    ap.add_argument("-p", "--plot", type=str, default="SnoutNet_plot.png")
    ap.add_argument("-s", "--save_file", type=str, default="SnoutNet.txt.pth")
    ap.add_argument("-a", "--augment", type=int, default=0)
    ap.add_argument("--data_root", type=str, default="data/oxford-iiit-pet-noses")
    ap.add_argument("--img_size", type=int, default=227)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=100)
    # for pretrained model
    ap.add_argument("-t", "--toggle_pretrain", type=int, default=0)
    ap.add_argument("--backbone", type=str, default="alexnet", choices=["alexnet", "vgg16"])
    return ap.parse_args()


def main():
    args = parse_args()
    seed_all(args.seed)
    device = get_device()
    print(f"device: {device}")

    train_loader = make_loader(args.data_root, args.img_size, args.batch_size, args.augment)

    if args.toggle_pretrain == 1:
        model = build_model(args.backbone, pretrained=True, freeze_backbone=True)
        model = model.to(device).float()
    else:
        model = SnoutNet().to(device).float()

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    train(args.epochs, model, optimizer, loss_fn, train_loader, scheduler, device, args.save_file, args.plot)


if __name__ == "__main__":
    main()
