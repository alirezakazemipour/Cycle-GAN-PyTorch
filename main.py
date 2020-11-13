import cv2
import glob
import random
from train import Train
import numpy as np
import time
import imageio
from concurrent import futures
import os
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter

A_images_dir = glob.glob(
    "horse2zebra/trainA/*.jpg")
B_images_dir = glob.glob(
    "horse2zebra/trainB/*.jpg")

device = torch.device("cuda")
lr = 2e-4
TRAIN_FLAG = False


def normalize_img(image):
    return (image.astype(np.float) / 127.5) - 1


def random_crop(image):
    max_x = image.shape[1] - 256
    max_y = image.shape[0] - 256

    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)

    crop = image[y: y + 256, x: x + 256]
    return crop


def random_jitter(a):
    a = cv2.resize(a, (286, 286), interpolation=cv2.INTER_NEAREST)
    a = random_crop(a)

    if np.random.random() > 0.5:
        a = np.fliplr(a)
    return a


def preprocess_image_train(addr):
    img = cv2.imread(addr)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = random_jitter(img)
    img = normalize_img(img)
    return img


if TRAIN_FLAG:

    with futures.ProcessPoolExecutor() as executor:
        trainA = np.stack(list(executor.map(preprocess_image_train, A_images_dir)))
        trainB = np.stack(list(executor.map(preprocess_image_train, B_images_dir)))

    trainA = np.expand_dims(trainA, axis=1)
    trainB = np.expand_dims(trainB, axis=1)
    trainA = torch.from_numpy(trainA).float().permute([0, 1, 4, 2, 3]).contiguous().to(device)
    trainB = torch.from_numpy(trainB).float().permute([0, 1, 4, 2, 3]).contiguous().to(device)

    train = Train(3, device, lr)
    ep = 0
    if os.path.exists("CycleGan.pth"):
        ep = train.load_weights("CycleGan.pth")
        print("Checkpoint loaded")

    for epoch in range(ep + 1, 200 + 1):
        for step in tqdm(range(1, 1 + min(len(trainA), len(trainB)))):
            start_time = time.time()
            idx_a = random.randint(0, len(trainA) - 1)
            idx_b = random.randint(0, len(trainB) - 1)

            A_image, B_image = trainA[idx_a], trainB[idx_b]

            fake_a, recycle_a, fake_b, recycle_b = train.forward(A_image, B_image)
            generator_loss, a_gan_loss, a_cycle_loss, loss_idt_A, b_gan_loss, b_cycle_loss, loss_idt_B = \
                train.calculate_generator_loss(A_image, fake_a, recycle_a, B_image, fake_b, recycle_b)
            train.optimize_generator(generator_loss)

            history_fake_a, history_fake_b = train.get_history(fake_a.detach(), fake_b.detach())

            a_dis_loss, b_dis_loss = train.calculate_discriminator_loss(A_image, history_fake_a, B_image,
                                                                        history_fake_b)

            train.optimize_discriminator(a_dis_loss, b_dis_loss)
            # print(f"Step:{step}| "
            #       f"Date:{time.time() - start_time:3.3f}")

            # with SummaryWriter("logs/") as writer:
            #     writer.add_scalar("A_GAN_Loss", a_gan_loss, epoch * step)
            #     writer.add_scalar("A_Recycle_Loss", a_cycle_loss, epoch * step)
            #     writer.add_scalar("A_Identity_Loss", loss_idt_A, epoch * step)
            #     writer.add_scalar("A_Dis_Loss", a_dis_loss, epoch * step)
            #     writer.add_scalar("B_GAN_Loss", b_gan_loss, epoch * step)
            #     writer.add_scalar("B_Recycle_Loss", b_cycle_loss, epoch * step)
            #     writer.add_scalar("B_Identity_Loss", loss_idt_B, epoch * step)
            #     writer.add_scalar("B_Dis_Loss", b_dis_loss, epoch * step)

        if epoch > 100:
            # train.schedule_optimizers()
            lr = max(1 - 1e-2 * (epoch - 100), 0) * lr
            for g_param_group, d_param_group in zip(train.generator_opt.param_groups,
                                                    train.discriminator_opt.param_groups):
                g_param_group['lr'] = lr
                d_param_group["lr"] = lr

        print(f"Epoch:{epoch}| "
              f"generator_loss:{generator_loss.item():.3f}| "
              f"discriminator:{0.5 * (a_dis_loss + b_dis_loss).item():.3f}| "
              f"duration:{time.time() - start_time:.3f}| "
              f"lr:{lr} ")

        train.save_weights(epoch, lr)
        if epoch % 25 == 0:
            I = fake_b[0].permute([1, 2, 0]).detach().cpu().numpy()
            image_numpy = (I + 1.0) / 2.0
            image_numpy = (image_numpy * 255).astype(np.uint8)
            imageio.imwrite(f"step_a{epoch}.png", image_numpy)
            imageio.imwrite(f"step_a{epoch}_real.png", trainA[idx_a][0].permute([1, 2, 0]).cpu().numpy())
else:
    horse = cv2.imread("horse.jpg")
    horse = cv2.cvtColor(horse, cv2.COLOR_BGR2RGB)
    horse = cv2.resize(horse, (256, 256), interpolation=cv2.INTER_NEAREST)
    horse = normalize_img(horse)
    zebra = cv2.imread("zebra.jpg")
    zebra = cv2.cvtColor(zebra, cv2.COLOR_BGR2RGB)
    zebra = cv2.resize(zebra, (256, 256), interpolation=cv2.INTER_NEAREST)
    zebra = normalize_img(zebra)

    horse = np.expand_dims(horse, axis=0)
    zebra = np.expand_dims(zebra, axis=0)
    horse = torch.from_numpy(horse).float().permute([0, 3, 1, 2]).contiguous().to(device)
    zebra = torch.from_numpy(zebra).float().permute([0, 3, 1, 2]).contiguous().to(device)

    test = Train(3, device, lr)
    test.load_weights("CycleGan.pth")
    test.set_to_eval()

    fake_zebra = test.A_Generator(horse)
    fake_horse = test.B_Generator(zebra)

    fake_zebra = fake_zebra[0].permute([1, 2, 0]).detach().cpu().numpy()
    fake_zebra = (fake_zebra + 1.0) / 2.0
    fake_zebra = (fake_zebra * 255).astype(np.uint8)
    imageio.imwrite(f"fake_zebra.png", fake_horse)

    fake_horse = fake_horse[0].permute([1, 2, 0]).detach().cpu().numpy()
    fake_horse = (fake_horse + 1.0) / 2.0
    fake_horse = (fake_horse * 255).astype(np.uint8)
    imageio.imwrite(f"fake_horse.png", fake_horse)
