from flask import Flask, request, jsonify, render_template
from PIL import Image
import torch
import torchvision.transforms as transforms
import io
import timm

app = Flask(__name__)

# Initialize model architecture with correct in_chans parameter
model = timm.create_model(
    'swin_base_patch4_window7_224',
    pretrained=False,
    num_classes=2,
    in_chans=1  # Important: Match the input channels from training
)

# Load model weights
state_dict = torch.load('D:\\Data_tumor_classification\\model.pth', map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.eval()

# Define your class labels
class_names = ['No Tumor', 'Tumor']

# Image transform - convert to grayscale first
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # For single channel
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Read image from memory and convert to grayscale
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('L')  # 'L' mode for grayscale

        # Transform image
        image_tensor = transform(image).unsqueeze(0)

        # Predict
        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)

            prediction_label = class_names[predicted.item()]
            confidence_score = confidence.item()

        return jsonify({
            'prediction': prediction_label,
            'confidence': float(confidence_score),
            'class': 'yes' if prediction_label == 'Tumor' else 'no'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)