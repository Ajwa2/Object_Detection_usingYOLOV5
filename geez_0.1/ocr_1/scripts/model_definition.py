import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


geez_characters = [
    'ሀ', 'ለ', 'ሐ', 'መ', 'ሠ', 'ረ', 'ሰ', 'ሸ', 'ቀ', 'በ', 'ቨ', 'ተ', 'ቸ', 'ኀ', 'ነ', 'ኘ', 'አ', 'ከ', 'ኸ', 'ወ', 'ዐ', 'ዘ', 'ዠ', 'የ', 'ደ', 'ጀ', 'ገ', 'ጠ', 'ጨ', 'ጰ', 'ጸ', 'ፀ', 'ፈ', 'ፐ',
    'ሂ','ሃ','ሄ','ህ','ሆ','ሇ','ሁ','ሉ','ሊ','ላ','ሌ','ል','ሎ','ሏ',
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
    '፪','፫','፬','፭','፮','፯','፰','፱','፲','፳','፴','፵','፶','፷','፸','፹','፺','፻','፼', 
    '፠', '፡', '።', '፣', '፥', '፤', '፦' '፨', '፧','»','«'

]

class GeezOCRModel(nn.Module):
    def __init__(self, num_classes):
        super(GeezOCRModel, self).__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        if torch.isnan(x).any():
            print("NaN detected after conv1!")
        
        x = self.pool(F.relu(self.conv2(x)))
        if torch.isnan(x).any():
            print("NaN detected after conv2!")
        
        x = self.pool(F.relu(self.conv3(x)))
        if torch.isnan(x).any():
            print("NaN detected after conv3!")
        
        x = x.view(x.size(0), -1)  # Flatten the tensor while maintaining batch size
        x = F.relu(self.fc1(x))
        if torch.isnan(x).any():
            print("NaN detected after fc1!")
        
        x = self.fc2(x)
        if torch.isnan(x).any():
            print("NaN detected after fc2!")
        
        x = x.view(x.size(0), x.size(1) // self.num_classes, self.num_classes)  # Ensure shape (N, T, C)
        x = x.permute(1, 0, 2)  # (T, N, C) for CTCLoss

        return x

if __name__ == "__main__":
    num_classes = len(geez_characters)  # Number of classes
    model = GeezOCRModel(num_classes)

    def initialize_weights(model):
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    model.apply(initialize_weights)
    print(model)

