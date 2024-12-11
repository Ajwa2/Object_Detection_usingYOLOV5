# # import os
# # import torch
# # import torch.nn as nn
# # import torch.optim as optim
# # from torch.utils.data import DataLoader
# # from torchvision import transforms

# # from data_preparation import GeezOCRDataset
# # from model_definition import GeezOCRModel

# # def train_model(model, dataloader, criterion, optimizer, num_epochs, device):
# #     model.train()
# #     for epoch in range(num_epochs):
# #         running_loss = 0.0
# #         for images, annotations in dataloader:
# #             images = images.to(device)
# #             labels = []
# #             target_lengths = []

# #             for annotation in annotations:
# #                 if isinstance(annotation, dict):
# #                     label_seq = annotation.get('label', None)
# #                 else:
# #                     label_seq = annotation

# #                 if label_seq:
# #                     label_tensor = torch.tensor([ord(char) for char in label_seq], dtype=torch.long)
# #                     labels.extend(label_tensor)
# #                     target_lengths.append(len(label_tensor))
# #                 else:
# #                     print(f"Warning: No label found for annotation {annotation}")

# #             if len(target_lengths) != images.size(0):
# #                 print(f"Error: Mismatch in lengths - Expected {images.size(0)}, Got {len(target_lengths)}")
# #                 continue

# #             labels = torch.tensor(labels, dtype=torch.long).to(device)
# #             input_lengths = torch.full(size=(images.size(0),), fill_value=images.size(2), dtype=torch.long).to(device)
# #             target_lengths = torch.tensor(target_lengths, dtype=torch.long).to(device)

# #             print(f"Batch size: {images.size(0)}")
# #             print(f"Input lengths: {input_lengths.size()}")
# #             print(f"Target lengths: {target_lengths.size()}")

# #             optimizer.zero_grad()
# #             outputs = model(images)
# #             outputs = outputs.permute(1, 0, 2)

# #             loss = criterion(outputs, labels, input_lengths, target_lengths)
# #             loss.backward()
# #             optimizer.step()

# #             running_loss += loss.item()

# #         print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")

# #     print("Training complete.")
# #     return model

# # if __name__ == "__main__":
# #     image_dirs = [
# #         "C:/Users/nesre/Desktop/geez_char_recognition/geez_ocr_project/data/images/lines/train",
# #         "C:/Users/nesre/Desktop/geez_char_recognition/geez_ocr_project/data/images/lines/test",
# #         "C:/Users/nesre/Desktop/geez_char_recognition/geez_ocr_project/data/images/three"
# #     ]
# #     annotation_files = [
# #         "C:/Users/nesre/Desktop/geez_char_recognition/geez_ocr_project/data/annotations/annotations.json",
# #         "C:/Users/nesre/Desktop/geez_char_recognition/geez_ocr_project/data/annotations/annotations_copy.json"
# #     ]

# #     batch_size = 32
# #     num_classes = 328 + 1
# #     num_epochs = 10
# #     learning_rate = 0.001

# #     transform = transforms.Compose([
# #         transforms.ToPILImage(),
# #         transforms.Resize((128, 128)),
# #         transforms.ToTensor(),
# #         transforms.Normalize((0.5,), (0.5,))
# #     ])

# #     dataset = GeezOCRDataset(image_dirs, annotation_files, transform)
# #     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# #     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# #     model = GeezOCRModel(num_classes).to(device)
# #     criterion = nn.CTCLoss()
# #     optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# #     trained_model = train_model(model, dataloader, criterion, optimizer, num_epochs, device)

# #     model_save_path = "C:/Users/nesre/Desktop/geez_char_recognition/geez_ocr_project/outputs/models/geez_ocr_model.pth"
# #     torch.save(trained_model.state_dict(), model_save_path)
# #     print(f"Model saved to {model_save_path}")


# # that was a good approch 

# # import os
# # import torch
# # import torch.nn as nn
# # import torch.optim as optim
# # from torch.utils.data import DataLoader
# # from torchvision import transforms

# # from data_preparation import GeezOCRDataset
# # from model_definition import GeezOCRModel

# # def train_model(model, dataloader, criterion, optimizer, num_epochs, device):
# #     model.train()
# #     for epoch in range(num_epochs):
# #         running_loss = 0.0
# #         for images, annotations in dataloader:
# #             images = images.to(device)
# #             labels = []
# #             target_lengths = []

# #             batch_size = images.size(0)
# #             for annotation in annotations:
# #                 label_seq = annotation
# #                 if label_seq:
# #                     label_tensor = torch.tensor([ord(char) for char in label_seq], dtype=torch.long)
# #                     labels.extend(label_tensor)
# #                     target_lengths.append(len(label_tensor))
# #                 else:
# #                     print(f"Warning: No label found for annotation {annotation}")

# #             if len(target_lengths) != batch_size:
# #                 print(f"Error: Mismatch in lengths - Expected {batch_size}, Got {len(target_lengths)}")
# #                 continue

# #             labels = torch.tensor(labels, dtype=torch.long).to(device)
# #             input_lengths = torch.full(size=(batch_size,), fill_value=1, dtype=torch.long).to(device)  # Temporal dimension is 1
# #             target_lengths = torch.tensor(target_lengths, dtype=torch.long).to(device)

# #             # Debugging: Check sizes of input_lengths and target_lengths
# #             print(f"Batch size: {batch_size}")
# #             print(f"Input lengths: {input_lengths.size()}")
# #             print(f"Target lengths: {target_lengths.size()}")
# #             print(f"Actual values of input_lengths: {input_lengths}")
# #             print(f"Actual values of target_lengths: {target_lengths}")

# #             optimizer.zero_grad()
# #             outputs = model(images)
# #             outputs = outputs.permute(1, 0, 2)  # Permute for CTC loss: (T, N, C)

# #             # Debug the sizes of outputs
# #             print(f"Outputs size: {outputs.size()}")

# #             loss = criterion(outputs, labels, input_lengths, target_lengths)
# #             loss.backward()
# #             optimizer.step()

# #             running_loss += loss.item()

# #         print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")

# #     print("Training complete.")
# #     return model

# # if __name__ == "__main__":
# #     image_dirs = [
# #         "C:/Users/nesre/Desktop/geez_char_recognition/geez_ocr_project/data/images/lines/train",
# #         "C:/Users/nesre/Desktop/geez_char_recognition/geez_ocr_project/data/images/lines/test",
# #         "C:/Users/nesre/Desktop/geez_char_recognition/geez_ocr_project/data/images/three"
# #     ]
# #     annotation_files = [
# #         "C:/Users/nesre/Desktop/geez_char_recognition/geez_ocr_project/data/annotations/annotations.json",
# #         "C:/Users/nesre/Desktop/geez_char_recognition/geez_ocr_project/data/annotations/annotations_copy.json"
# #     ]

# #     batch_size = 32
# #     num_classes = 328 + 1  # Adding 1 for the blank label required by CTC
# #     num_epochs = 10
# #     learning_rate = 0.001

# #     transform = transforms.Compose([
# #         transforms.ToPILImage(),
# #         transforms.Resize((128, 128)),
# #         transforms.ToTensor(),
# #         transforms.Normalize((0.5,), (0.5,))
# #     ])

# #     dataset = GeezOCRDataset(image_dirs, annotation_files, transform)
# #     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# #     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# #     model = GeezOCRModel(num_classes).to(device)
# #     criterion = nn.CTCLoss()
# #     optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# #     trained_model = train_model(model, dataloader, criterion, optimizer, num_epochs, device)

# #     # Save the model
# #     model_save_path = "C:/Users/nesre/Desktop/geez_char_recognition/geez_ocr_project/outputs/models/geez_ocr_model.pth"
# #     torch.save(trained_model.state_dict(), model_save_path)
# #     print(f"Model saved to {model_save_path}")

# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torchvision import transforms

# from data_preparation import GeezOCRDataset
# from model_definition import GeezOCRModel


# def train_model(model, dataloader, criterion, optimizer, num_epochs, device):
#     model.train()
#     clip_value = 5.0  # Adjust based on your needs
#     for epoch in range(num_epochs):
#         running_loss = 0.0
#         for images, annotations in dataloader:
#             images = images.to(device)
#             labels = []
#             target_lengths = []

#             batch_size = images.size(0)
#             for annotation in annotations:
#                 label_seq = annotation
#                 if label_seq:
#                     label_tensor = torch.tensor([ord(char) for char in label_seq], dtype=torch.long)
#                     labels.extend(label_tensor)
#                     target_lengths.append(len(label_tensor))
#                 else:
#                     print(f"Warning: No label found for annotation {annotation}")

#             if len(target_lengths) != batch_size:
#                 print(f"Error: Mismatch in lengths - Expected {batch_size}, Got {len(target_lengths)}")
#                 continue

#             labels = torch.tensor(labels, dtype=torch.long).to(device)
#             input_lengths = torch.full(size=(batch_size,), fill_value=1, dtype=torch.long).to(device)  # Temporal dimension is 1
#             target_lengths = torch.tensor(target_lengths, dtype=torch.long).to(device)

#             # Debugging: Check sizes of input_lengths and target_lengths
#             print(f"Batch size: {batch_size}")
#             print(f"Input lengths: {input_lengths.size()}")
#             print(f"Target lengths: {target_lengths.size()}")
#             print(f"Actual values of input_lengths: {input_lengths}")
#             print(f"Actual values of target_lengths: {target_lengths}")

#             optimizer.zero_grad()
#             outputs = model(images)
#             outputs = outputs.permute(1, 0, 2)  # Permute for CTC loss: (T, N, C)

#             # Debug the sizes of outputs
#             print(f"Outputs size: {outputs.size()}")
#             print(f"Outputs sample: {outputs[:2]}")  # Print sample outputs for inspection

#             loss = criterion(outputs, labels, input_lengths, target_lengths)
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)  # Gradient clipping
#             optimizer.step()

#             running_loss += loss.item()

#         print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")

#     print("Training complete.")
#     return model

# if __name__ == "__main__":
#     image_dirs = [
#         "C:/Users/nesre/Desktop/geez_char_recognition/geez_ocr_project/data/images/lines/train",
#         "C:/Users/nesre/Desktop/geez_char_recognition/geez_ocr_project/data/images/lines/test",
#         "C:/Users/nesre/Desktop/geez_char_recognition/geez_ocr_project/data/images/three"
#     ]
#     annotation_files = [
#         "C:/Users/nesre/Desktop/geez_char_recognition/geez_ocr_project/data/annotations/annotations.json",
#         "C:/Users/nesre/Desktop/geez_char_recognition/geez_ocr_project/data/annotations/annotations_copy.json"
#     ]

#     batch_size = 32
#     num_classes = 328 + 1  # Adding 1 for the blank label required by CTC
#     num_epochs = 10
#     learning_rate = 0.0001  # Reduced learning rate

#     transform = transforms.Compose([
#         transforms.ToPILImage(),
#         transforms.Resize((128, 128)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,))  # Check if this normalization is appropriate
#     ])

#     dataset = GeezOCRDataset(image_dirs, annotation_files, transform)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model = GeezOCRModel(num_classes).to(device)
#     criterion = nn.CTCLoss()
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#     trained_model = train_model(model, dataloader, criterion, optimizer, num_epochs, device)

#     # Save the model
#     model_save_path = "C:/Users/nesre/Desktop/geez_char_recognition/geez_ocr_project/outputs/models/geez_ocr_model.pth"
#     torch.save(trained_model.state_dict(), model_save_path)
#     print(f"Model saved to {model_save_path}")



import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.init as init
import numpy as np
from PIL import Image
import cv2

from data_preparation import get_dataloader
from model_definition import GeezOCRModel

# Define Geez characters and create mapping
geez_characters = [
    'ሀ', 'ለ', 'ሐ', 'መ', 'ሠ', 'ረ', 'ሰ', 'ሸ', 'ቀ', 'በ', 'ቨ', 'ተ', 'ቸ', 'ኀ', 'ነ', 'ኘ', 'አ', 'ከ', 'ኸ', 'ወ', 'ዐ', 'ዘ', 'ዠ', 'የ', 'ደ', 'ጀ', 'ገ', 'ጠ', 'ጨ', 'ጰ', 'ጸ', 'ፀ', 'ፈ', 'ፐ',
    'ሂ','ሃ','ሄ','ህ','ሆ','ሇ','ሁ','ሉ','ሊ','ላ','ሌ','ል','ሎ','ሏ','.','-',
    'ሑ','ሒ','ሓ','ሔ','ሕ','ሖ','ሙ','ሚ','ማ','ሜ','ም','ሞ','ሡ','ሢ','ሣ','ሤ','ሥ','ሦ','ሧ',
    'ሩ','ሪ','ራ','ሬ','ር','ሮ','ሱ','ሲ','ሳ','ሴ','ስ','ሶ','ሹ','ሺ','ሻ','ሼ','ሽ','ሾ',
    'ቁ','ቂ','ቃ','ቄ','ቅ','ቆ','ቡ','ቢ','ባ','ቤ','ብ','ቦ','ቧ','ቱ','ቲ','ታ','ቴ','ት','ቶ','ቷ',
    'ኁ','ኂ','ኃ','ኄ','ኅ','ኆ','ኋ','ኑ','ኒ','ና','ን','ኖ','ኔ','ኡ','ኢ','ኣ','ኤ','እ','ኦ','ኧ','ኩ','ኪ','ካ','ኬ','ክ','ኮ','ኳ',
    'ዉ','ዊ','ዋ','ዌ','ው','ዎ','ዑ','ዒ','ዓ','ዔ','ዕ','ዖ',
    'ዙ','ዚ','ዛ','ዜ','ዝ','ዞ','ዟ', 'ዩ','ዪ','ያ','ዬ','ይ','ዮ',     
    'ዱ','ዲ','ዳ','ዴ','ድ','ዶ','ዷ',    
    'ጉ','ጊ','ጋ','ጌ','ግ','ጎ','ጓ',
    'ጡ','ጢ','ጣ','ጤ','ጥ','ጦ','ጧ',    
    'ጱ','ጲ','ጳ','ጴ','ጵ','ጶ','ጷ',    
    'ጹ','ጺ','ጻ','ጼ','ጽ','ጾ','ጿ',    
    'ፁ','ፂ','ፃ','ፄ','ፅ','ፆ',     
    'ፉ','ፊ','ፋ','ፌ','ፍ','ፎ','ፏ','ፚ',
    'ፑ','ፒ','ፓ','ፔ','ፕ','ፖ','ፗ',    
    'ቐ','ቑ','ቒ','ቓ','ቔ','ቕ','ቖ',     
    'ቘ','ቚ','ቛ','ቜ','ቝ',     
    'ቩ','ቪ','ቫ','ቬ','ቭ','ቮ','ቯ',
    'ቹ','ቺ','ቻ','ቼ','ች','ቾ','ቿ',
    'ⶓ','ⶔ','ⶕ','ⶖ',     
    'ኙ','ኚ','ኛ','ኜ','ኝ','ኞ','ኟ',
    'ኹ','ኺ','ኻ','ኼ','ኽ','ኾ',     
    'ዀ','ዂ','ዃ','ዄ','ዅ',         
    'ዡ','ዢ','ዣ','ዤ','ዥ','ዦ','ዧ',
    'ጁ','ጂ','ጃ','ጄ','ጅ','ጆ','ጇ',
    'ጘ','ጙ','ጚ','ጛ','ጜ','ጝ','ጞ','ጟ',
    'ጩ','ጪ','ጫ','ጬ','ጭ','ጮ','ጯ',
    'ቈ','ኈ','ጐ','ኰ','ቘ','ዀ',
    'ቈ','ቊ','ቋ','ቌ','ቍ','ጘ','ⶓ',' ',
    'ኈ','ኊ','ኋ','ኌ','ኍ','ጐ','ጒ','ጓ','ጔ','ጕ',
    'ኰ','ኲ','ኳ','ኴ','ኵ','፩','(',')',
    '፪','፫','፬','፭','፮','፯','፰','፱','፲','፳','፴','፵','፶','፷','፸','፹','፺','፻','፼', 'ዽ',
    '፠', '፡', '።', '፣', '፥', '፤', '፦' '፨', '፧','»','«','\t','᎒','‹','…', '፦','n','›','ⷋ','!','”','“'

]

char_to_id = {char: idx for idx, char in enumerate(geez_characters)}
id_to_char = {idx: char for idx, char in enumerate(geez_characters)}



def train_model(model, dataloader, criterion, optimizer, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, annotations in dataloader:
            batch_size = images.size(0)
            images = images.to(device)
            if torch.isnan(images).any():
                print("NaN detected in images!")
                continue

            labels = []
            target_lengths = []

            for annotation in annotations:
                label_seq = annotation
                label_tensor = torch.tensor([char_to_id[char] for char in label_seq], dtype=torch.long)
                labels.extend(label_tensor)
                target_lengths.append(len(label_tensor))

            labels = torch.tensor(labels, dtype=torch.long).to(device)

            seq_len = model(images).size(0)  # Adjust based on the sequence length from model output
            input_lengths = torch.full(size=(batch_size,), fill_value=seq_len, dtype=torch.long).to(device)
            target_lengths = torch.tensor(target_lengths, dtype=torch.long).to(device)

            optimizer.zero_grad()
            outputs = model(images)
            if torch.isnan(outputs).any():
                print("NaN detected in outputs!")
                continue

            loss = criterion(outputs, labels, input_lengths, target_lengths)
            if torch.isnan(loss).any():
                print("NaN detected in loss!")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")

    print("Training complete.")
    return model

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

    batch_size = 32
    num_classes = len(geez_characters) + 1
    num_epochs = 10
    learning_rate = 0.00001

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataloader = get_dataloader(image_dirs, annotation_files, batch_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = GeezOCRModel(num_classes).to(device)
    def initialize_weights(model):
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    model.apply(initialize_weights)
    criterion = nn.CTCLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    trained_model = train_model(model, dataloader, criterion, optimizer, num_epochs, device)

    model_save_path = "C:/Users/nesre/Desktop/geez_char_recognition/geez_ocr_project/outputs/models/geez_ocr_model.pth"
    torch.save(trained_model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
