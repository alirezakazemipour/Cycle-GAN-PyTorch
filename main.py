import cv2
import glob
import random
from train import Train
import numpy as np

A_images_dir = glob.glob("./dataset/A/trainA/*.jpg")
B_images_dir = glob.glob("./dataset/B/trainB/*.jpg")

trainA = [cv2.imread(dir) / 255.0 for dir in A_images_dir]
trainB = [cv2.imread(dir) / 255.0 for dir in B_images_dir]

train = Train(3)
for epoch in range(200):
    idx_a = random.randint(0, len(trainA) - 1)
    idx_b = random.randint(0, len(trainB) - 1)
    fake_a, recycle_a, fake_b, recycle_b = train.forward(trainA[idx_a], trainB[idx_b])
    generator_loss = train.calculate_generator_loss(trainA[idx_a], fake_a, recycle_a, trainB[idx_b], fake_b, recycle_b)
    train.optimize_generator(generator_loss)

    history_fake_a, history_fake_b = train.get_history()

    if history_fake_a is not None:
        a_dis_loss, b_dis_loss = train.calculate_discriminator_loss(
            trainA[idx_a], np.vstack(history_fake_a),
            trainB[idx_b], np.vstack(history_fake_a))

        train.optimize_discriminator(a_dis_loss, b_dis_loss)

        print(f"Epoch:{epoch}| "
              f"generator_loss:{generator_loss.item():3.3f}| "
              f"discriminator:{0.5 * (a_dis_loss + b_dis_loss).item():3.3f}")
