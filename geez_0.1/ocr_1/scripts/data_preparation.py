import os
import cv2
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class GeezOCRDataset(Dataset):
    def __init__(self, image_dirs, annotation_files, transform=None):
        print("Initializing GeezOCRDataset...")
        self.image_dirs = image_dirs
        self.annotation_files = annotation_files
        self.transform = transform
        self.images = []
        self.annotations = {}

        print(f"Image directories: {image_dirs}")
        print(f"Annotation files: {annotation_files}")

        for folder in image_dirs:
            if os.path.exists(folder):
                print(f"Checking directory: {folder}")
                for root, _, files in os.walk(folder):
                    print(f"Contents of {root}: {files}")
                    for file in files:
                        if file.endswith(('.png', '.jpg', '.jpeg')):
                            self.images.append((root, file))
                            print(f"Found image: {os.path.join(root, file)}")
            else:
                print(f"Directory does not exist: {folder}")

        print(f"Total images found: {len(self.images)}")

        for annotation_file in annotation_files:
            annotation_path = os.path.join(annotation_file)
            if os.path.exists(annotation_path):
                print(f"Loading annotations from: {annotation_path}")
                with open(annotation_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if "annotations" in data:
                        for item in data["annotations"]:
                            print(f"Annotation item: {item}")
                            file_name = item.get("file_name", None)
                            image_id = item.get("image_id", None)
                            label = item["attributes"].get("Label", item["attributes"].get("Lable", None))
                            if file_name and label:
                                self.annotations[file_name] = label
                            elif image_id and label:
                                self.annotations[f"line_{image_id}.png"] = label
                            else:
                                print(f"Invalid annotation entry: {item}")
                    else:
                        print(f"Invalid annotation format in {annotation_file}")
            else:
                print(f"Annotation file does not exist: {annotation_file}")

        self.images = [img for img in self.images if os.path.basename(img[1]) in self.annotations]
        print(f"Total valid images found: {len(self.images)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_folder, img_name = self.images[idx]
        img_path = os.path.join(img_folder, img_name)
        print(f"Loading image: {img_path}")
        image = cv2.imread(img_path)
        if image is None:
            print(f"Failed to load image: {img_path}")
        else:
            print(f"Loaded image: {img_path}")

        image = self.preprocess_image(image)

        if self.transform:
            image = self.transform(image)

        annotation = self.annotations.get(img_name, None)
        if annotation is None:
            print(f"Warning: No annotation found for image {img_name}")

        print(f"Processed image shape: {image.shape}")
        print(f"Annotation: {annotation}")

        return image, annotation

    def preprocess_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        denoised = cv2.fastNlMeansDenoising(thresh, None, 30, 7, 21)
        if np.isnan(denoised).any(): 
            print("NaN detected in preprocessed image!")
        return denoised

def get_dataloader(image_dirs, annotation_files, batch_size=32):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    print("Creating GeezOCRDataset...")
    dataset = GeezOCRDataset(image_dirs, annotation_files, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print("Dataloader created...")

    return dataloader

if __name__ == "__main__":
    image_dirs = [
        "C:/Users/nesre/Desktop/geez_char_recognition/geez_ocr_project/data/images/lines/train",
        "C:/Users/nesre/Desktop/geez_char_recognition/geez_ocr_project/data/images/lines/test",
        "C:/Users/nesre/Desktop/geez_char_recognition/geez_ocr_project/data/images/three"
    ]
    annotation_files = [
        "C:/Users/nesre/Desktop/geez_char_recognition/geez_ocr_project/data/annotations/annotations.json",
        "C:/Users/nesre/Desktop/geez_char_recognition/geez_ocr_project/data/annotations/annotations_copy.json"
    ]

    dataloader = get_dataloader(image_dirs, annotation_files)
