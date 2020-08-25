import cv2
import glob
import random
from train import Train
import numpy as np
import time
import imageio
from concurrent import futures
import os
import pickle
from tqdm import tqdm

A_images_dir = glob.glob(
    "horse2zebra/trainA/*.jpg")
B_images_dir = glob.glob(
    "horse2zebra/trainB/*.jpg")

trainA = [cv2.imread(dir) for dir in A_images_dir]
trainB = [cv2.imread(dir) for dir in B_images_dir]
# trainA = []
# trainB = []


def train_processing(dir):
    I = cv2.imread(dir)
    I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
    return I


# with futures.ProcessPoolExecutor() as executor:
#     results = executor.map(train_processing, A_images_dir)
#
#     for result in results:
#         trainA.append(result)
#
# with futures.ProcessPoolExecutor() as executor:
#     results = executor.map(train_processing, B_images_dir)
#
#     for result in results:
#         trainB.append(result)


# with open('trainA_dataset.pickle', 'rb') as f:
#     trainA = pickle.load(f)
#
# with open('trainB_dataset.pickle', 'rb') as f:
#     trainB = pickle.load(f)


def normalize_img(image):
    return (image.astype(np.float) / 127.5) - 1


def random_crop(image):
    max_x = image.shape[1] - 256
    max_y = image.shape[0] - 256

    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)

    crop = image[y: y + 256, x: x + 256]

    return crop


def random_jitter(a, b):
    a = cv2.resize(a, (286, 286), interpolation=cv2.INTER_NEAREST)
    b = cv2.resize(b, (286, 286), interpolation=cv2.INTER_NEAREST)
    a = random_crop(a)
    b = random_crop(b)

    if np.random.random() > 0.5:
        a = np.fliplr(a)
        b = np.fliplr(b)
    return a, b


def preprocess_image_train(a, b):
    a, b = random_jitter(a, b)
    b = normalize_img(b)
    a = normalize_img(a)
    return a, b


train = Train(3)
ep = 1
if os.path.exists("CycleGan.pth"):
    ep = train.load_weights("CycleGan.pth") + 1
    print("Checkpoint loaded")

for epoch in range(ep, 200 + 1):
    for step in tqdm(range(min(len(A_images_dir), len(B_images_dir)))):
        start_time = time.time()
        idx_a = random.randint(0, len(trainA) - 1)
        idx_b = random.randint(0, len(trainB) - 1)

        A_image, B_image = preprocess_image_train(trainA[idx_a], trainB[idx_b])

        fake_a, recycle_a, fake_b, recycle_b = train.forward(A_image, B_image)
        generator_loss, a_gan_loss, a_cycle_loss, idt_A, b_gan_loss, b_cycle_loss, idt_B = \
            train.calculate_generator_loss(A_image, fake_a, recycle_a, B_image, fake_b, recycle_b)
        train.optimize_generator(generator_loss)

        history_fake_a, history_fake_b = train.get_history(fake_a.detach().cpu().numpy(), fake_b.detach().cpu().numpy())

        a_dis_loss, b_dis_loss = train.calculate_discriminator_loss(
            A_image, history_fake_a,
            B_image, history_fake_b)

        train.optimize_discriminator(a_dis_loss, b_dis_loss)
        # print(f"Step:{step}| "
        #       f"Date:{time.time() - start_time:3.3f}")

    print(f"Epoch:{epoch}| "
          f"generator_loss:{generator_loss.item():.3f}| "
          f"discriminator:{0.5 * (a_dis_loss + b_dis_loss).item():.3f}| "
          f"duration:{time.time() - start_time:.3f}| "
          f"generator lr:{train.generator_scheduler.get_last_lr()}| "
          f"discriminator lr:{train.discriminator_scheduler.get_last_lr()}")

    train.schedule_optimizers()
    train.save_weights(epoch)

    if epoch % 3 == 0:
        I = fake_b[0].permute([1, 2, 0]).detach().cpu().numpy()
        image_numpy = (I + 1.0) / 2.0
        imageio.imwrite(f"step_a{epoch}.png", image_numpy)
        imageio.imwrite(f"step_a{epoch}_real.png", trainA[idx_a])
