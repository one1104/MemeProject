# test_inference_final.py
import inference
import os
import random

TEST_DIR = r"C:\Users\meng-\Desktop\MemeProject\dataset_test"
test_imgs = []

for label in os.listdir(TEST_DIR):
    folder = os.path.join(TEST_DIR, label)
    if not os.path.isdir(folder):
        continue
    imgs = [f for f in os.listdir(folder) if f.endswith('.jpg')]
    for img in imgs:
        test_imgs.append((os.path.join(folder, img), label))

random.shuffle(test_imgs)
correct = 0

for i, (img_path, true_label) in enumerate(test_imgs):
    pred, _ = inference.predict_emotion(img_path)
    if pred == true_label:
        correct += 1
    if (i + 1) % 100 == 0:
        print(f"进度: {i+1}/{len(test_imgs)} | 当前准确率: {correct/(i+1):.2%}")

print(f"\n最终准确率: {correct}/{len(test_imgs)} = {correct/len(test_imgs):.2%}")