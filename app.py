from flask import Flask, request, jsonify, render_template
import os
from google.cloud import vision_v1
import google.generativeai as genai

app = Flask(__name__)

# Set up Google Cloud Vision API
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"C:\Users\91944\Desktop\nodal-fountain-427610-v3-09c5edf8a401.json"
client = vision_v1.ImageAnnotatorClient()

# Configure Gemini API
genai.configure(api_key="AIzaSyCAI14HgK1SJd_jh4XxeVEoSh5Bgf8ydM4")

def ocr_google_vision(file_stream):
    """Google Cloud Vision API request with file stream."""
    content = file_stream.read()
    image = vision_v1.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    if texts:
        return texts[0].description
    return ''

def get_gemini_response(prompt):
    """Get response from Gemini chatbot."""
    modified_prompt = f"Provide a short answer with options: {prompt}"
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(modified_prompt)
    if response:
        return response.text
    return 'Failed to get a response from Gemini API'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    image_file = request.files['image']
    extracted_text = ocr_google_vision(image_file)

    if extracted_text:
        chat_response = get_gemini_response(extracted_text)
        return jsonify({"text": extracted_text, "response": chat_response})
    else:
        return jsonify({"error": "Failed to extract text from image"}), 500

if __name__ == '__main__':
    app.run(debug=True)
