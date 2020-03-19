import cv2
import glob
import random
from train import Train
import numpy as np
import time
import matplotlib.pyplot as plt

A_images_dir = glob.glob("./dataset/A/trainA/*.jpg")
B_images_dir = glob.glob("./dataset/B/trainB/*.jpg")

trainA = [cv2.imread(dir).astype(np.float) / 255.0 for dir in A_images_dir]
trainB = [cv2.imread(dir).astype(np.float) / 255.0 for dir in B_images_dir]

train = Train(3)
for epoch in range(200):
    start_time = time.time()
    idx_a = random.randint(0, len(trainA) - 1)
    idx_b = random.randint(0, len(trainB) - 1)
    fake_a, recycle_a, fake_b, recycle_b = train.forward(trainA[idx_a], trainB[idx_b])
    # train.add_to_history(fake_a.detach().cpu().numpy(), fake_b.detach().cpu().numpy())
    generator_loss = train.calculate_generator_loss(trainA[idx_a], fake_a, recycle_a, trainB[idx_b], fake_b, recycle_b)
    train.optimize_generator(generator_loss)

    history_fake_a, history_fake_b = train.get_history(fake_a.detach().cpu().numpy(), fake_b.detach().cpu().numpy())

    # if history_fake_a is not None:
    a_dis_loss, b_dis_loss = train.calculate_discriminator_loss(
        trainA[idx_a], history_fake_a,
        trainB[idx_b], history_fake_b)

    train.optimize_discriminator(a_dis_loss, b_dis_loss)

    print(f"Epoch:{epoch}| "
          f"generator_loss:{generator_loss.item():3.3f}| "
          f"discriminator:{0.5 * (a_dis_loss + b_dis_loss).item():3.3f}| "
          f"duration:{time.time() - start_time:3.3f}| "
          f"generator lr:{train.generator_scheduler.get_last_lr()}| "
          f"discriminator lr:{train.discriminator_scheduler.get_last_lr()}")
    # if epoch % 25 == 0:
    #     I = fake_b.squeeze(dim=0).permute([1, 2, 0]).detach().cpu().numpy()
    #     cv2.imwrite(f"step{epoch}.png", I * 0.5 + 0.5)
