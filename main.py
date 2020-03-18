import cv2
import glob
import random
from train import Train

A_images_dir = glob.glob("./dataset/A/trainA/*.jpg")
B_images_dir = glob.glob("./dataset/B/trainB/*.jpg")

trainA = [cv2.imread(dir) / 255.0 for dir in A_images_dir]
trainB = [cv2.imread(dir) / 255.0 for dir in B_images_dir]

idx_a = random.randint(0, len(trainA))
idx_b = random.randint(0, len(trainB))

train = Train(3)

fake_a, recycle_a, fake_b, recycle_b = train.forward(trainA[idx_a], trainB[idx_b])








