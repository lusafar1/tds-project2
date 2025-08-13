import os
import json
import base64
import tempfile
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from data_analyst import DataAnalystAgent
import logging

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/api/', methods=['POST'])
def analyze_data():
    try:
        # Check if questions.txt is provided
        if 'questions.txt' not in request.files:
            return jsonify({"error": "questions.txt file is required"}), 400
        
        questions_file = request.files['questions.txt']
        questions_content = questions_file.read().decode('utf-8')
        
        # Save uploaded files
        uploaded_files = {}
        for file_key in request.files:
            if file_key != 'questions.txt':
                file = request.files[file_key]
                if file.filename:
                    filename = secure_filename(file.filename)
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(file_path)
                    uploaded_files[file_key] = file_path
        
        # Initialize Data Analyst Agent
        agent = DataAnalystAgent()
        
        # Process the request
        result = agent.process_request(questions_content, uploaded_files)
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
