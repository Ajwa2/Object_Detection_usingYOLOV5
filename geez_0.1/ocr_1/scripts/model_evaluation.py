import torch
from PIL import Image
from torchvision import transforms
from model_definition import GeezOCRModel

# Define Geez characters and create mapping (use the same as in training)
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


# Load the saved model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_classes = len(geez_characters) + 1  # Number of characters + blank label for CTC
model = GeezOCRModel(num_classes).to(device)
model.load_state_dict(torch.load("C:/Users/nesre/Desktop/geez_char_recognition/geez_ocr_project/outputs/model/geez_ocr_mode_2l.pth"))
model.eval()  # Set the model to evaluation mode

# Prepare the transformation
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Function to decode model predictions using CTC
def decode_predictions(predictions, blank=0):
    predictions = predictions.argmax(dim=-1).squeeze(1).cpu().numpy()
    decoded_text = ""
    for pred in predictions:
        if pred != blank:
            decoded_text += id_to_char.get(pred, '')
    return decoded_text

# Function to test model on a single image
def test_model(image):
    image = transform(image).unsqueeze(0).to(device)  # Transform and add batch dimension
    with torch.no_grad():
        output = model(image)
        print(f"Raw output: {output}")  # Debug print to check raw output
        output = output.permute(1, 0, 2)  # Change to (seq_len, batch, num_classes) for CTC
        predicted_text = decode_predictions(output)
    return predicted_text

# Load a test image
test_image_path = "C:/Users/nesre/Desktop/geez_char_recognition/geez_ocr_project/test_image_3.png"
test_image = Image.open(test_image_path).convert("RGB")  # Load and convert image

# Run inference
predicted_text = test_model(test_image)
print(f"Predicted text: {predicted_text}")
