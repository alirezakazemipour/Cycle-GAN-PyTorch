import cv2
import glob
import random
from train import Train
import numpy as np
import time
import imageio

A_images_dir = glob.glob("./dataset/A/trainA/*.jpg")
B_images_dir = glob.glob("./dataset/B/trainB/*.jpg")

trainA = [cv2.imread(dir) for dir in A_images_dir]
trainB = [cv2.imread(dir) for dir in B_images_dir]


def normalize_img(image):
    return (image.astype(np.float) / 127.5) - 1


def random_crop(image):
    max_x = image.shape[1] - 256
    max_y = image.shape[0] - 256

    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)

    crop = image[y: y + 256, x: x + 256]

    return crop


def random_jitter(image):
    image = cv2.resize(image, (286, 286), interpolation=cv2.INTER_NEAREST)
    image = random_crop(image)
    # image = normalize_img(image)
    p = random.uniform(0, 1)
    if p > 0.5:
        image = cv2.flip(image, 1)
    return image


def preprocess_image_train(image):
    image = random_jitter(image)
    image = normalize_img(image)
    return image


train = Train(3)
for epoch in range(200):
    start_time = time.time()
    idx_a = random.randint(0, len(trainA) - 1)
    idx_b = random.randint(0, len(trainB) - 1)
    # idx_a = epoch
    # idx_b = epoch
    A_image = preprocess_image_train(trainA[idx_a])
    B_image = preprocess_image_train(trainB[idx_b])

    fake_a, recycle_a, fake_b, recycle_b = train.forward(A_image, B_image)
    # train.add_to_history(fake_a.detach().cpu().numpy(), fake_b.detach().cpu().numpy())
    generator_loss = train.calculate_generator_loss(A_image, fake_a, recycle_a, B_image, fake_b, recycle_b)
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
    if epoch % 25 == 0:
        I = fake_b.squeeze(dim=0).permute([1, 2, 0]).detach().cpu().numpy()
        imageio.imwrite(f"step{epoch}.png", I * 0.5 + 0.5)
