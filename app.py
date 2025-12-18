from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import os
import warnings

# Suppress scikit-learn version warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication

class JobClassificationAPI:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.skills_database = {}
        self.load_models()
    
    def load_models(self):
        """Load pre-trained models"""
        try:
            model_path = "model/"
            
            # Check if model directory exists
            if not os.path.exists(model_path):
                print(" Model directory not found. Creating demo mode...")
                return False
            
            # Load model components with warning suppression
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                model_files = ['model.pkl', 'vectorizer.pkl', 'label_encoder.pkl']
                for file in model_files:
                    if not os.path.exists(f"{model_path}{file}"):
                        print(f"{file} not found. Using demo mode...")
                        return False
                
                with open(f"{model_path}model.pkl", "rb") as f:
                    self.model = pickle.load(f)
                
                with open(f"{model_path}vectorizer.pkl", "rb") as f:
                    self.vectorizer = pickle.load(f)
                    
                with open(f"{model_path}label_encoder.pkl", "rb") as f:
                    self.label_encoder = pickle.load(f)
                    
                # Load skills database if available
                if os.path.exists(f"{model_path}skills_database.pkl"):
                    with open(f"{model_path}skills_database.pkl", "rb") as f:
                        self.skills_database = pickle.load(f)
            
            print(" Models loaded successfully!")
            return True
            
        except Exception as e:
            print(f" Error loading models: {e}")
            print("Using demo mode...")
            return False
    
    def predict_classification(self, job_title, skills, description=""):
        """Predict job classification"""
        if not self.model:
            return self._demo_prediction(job_title, skills)
        
        try:
            # Suppress warnings during prediction
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Prepare input same as training
                combined_text = f"{job_title} {skills} {description}".strip()
                X_input = self.vectorizer.transform([combined_text])
                
                # Get prediction
                prediction = self.model.predict(X_input)[0]
                probabilities = self.model.predict_proba(X_input)[0]
                
                classification = self.label_encoder.inverse_transform([prediction])[0]
                confidence = max(probabilities)
            
            return classification, confidence
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return self._demo_prediction(job_title, skills)
    
    def _demo_prediction(self, job_title, skills):
        """Demo prediction when model is not available"""
        skills_lower = skills.lower()
        title_lower = job_title.lower()
        
        # AI/Tech roles
        if any(keyword in skills_lower for keyword in ['ai', 'machine learning', 'deep learning', 'blockchain', 'kubernetes', 'devops', 'cloud', 'pytorch']):
            return 'Newly Created', 0.85
        elif any(keyword in title_lower for keyword in ['ai', 'blockchain', 'devops', 'cloud']):
            return 'Newly Created', 0.82
        # Manual/Traditional roles
        elif any(keyword in skills_lower for keyword in ['excel', 'typing', 'manual', 'assembly', 'data entry']):
            return 'Displaced', 0.78
        elif any(keyword in title_lower for keyword in ['clerk', 'operator', 'assembly']):
            return 'Displaced', 0.75
        # Everything else
        else:
            return 'Reshaped', 0.72
    
    def get_skill_recommendations(self, current_skills, classification):
        """Get skill recommendations based on classification"""
        recommendations = {
            'Newly Created': [
                {'name': 'Advanced Machine Learning', 'reason': 'High demand in AI roles', 'frequency': 89},
                {'name': 'Cloud Computing (AWS/Azure)', 'reason': 'Infrastructure for AI systems', 'frequency': 76},
                {'name': 'Data Visualization', 'reason': 'Communicating AI insights', 'frequency': 64},
                {'name': 'MLOps', 'reason': 'Production ML systems', 'frequency': 58}
            ],
            'Displaced': [
                {'name': 'Python Programming', 'reason': 'Automation and data analysis', 'frequency': 92},
                {'name': 'Data Analysis', 'reason': 'Transform manual work to insights', 'frequency': 85},
                {'name': 'Digital Marketing', 'reason': 'Growing online business needs', 'frequency': 71},
                {'name': 'Process Automation', 'reason': 'Replace manual tasks', 'frequency': 68},
                {'name': 'SQL Database Skills', 'reason': 'Data management capabilities', 'frequency': 64}
            ],
            'Reshaped': [
                {'name': 'Digital Tools Proficiency', 'reason': 'Modern workflow requirements', 'frequency': 78},
                {'name': 'Data Analytics', 'reason': 'Data-driven decision making', 'frequency': 73},
                {'name': 'Collaboration Tools', 'reason': 'Remote work capabilities', 'frequency': 65},
                {'name': 'CRM Software', 'reason': 'Customer relationship management', 'frequency': 59}
            ]
        }
        
        return recommendations.get(classification, [])

# Initialize the API
classifier = JobClassificationAPI()

@app.route('/')
def home():
    """Serve the main page with proper Unicode handling"""
    try:
        # Try different encodings
        encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open('index.html', 'r', encoding=encoding) as f:
                    content = f.read()
                return content
            except UnicodeDecodeError:
                continue
        
        # If all encodings fail, return fallback
        return create_fallback_html()
        
    except FileNotFoundError:
        return create_fallback_html()

def create_fallback_html():
    """Create a simple HTML page if index.html can't be read"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Job Classification System</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 40px; border-radius: 10px; }
            h1 { color: #333; text-align: center; }
            .test-form { background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0; }
            input, textarea { width: 100%; padding: 10px; margin: 10px 0; border: 1px solid #ddd; border-radius: 4px; }
            button { background: #007bff; color: white; padding: 12px 24px; border: none; border-radius: 4px; cursor: pointer; }
            button:hover { background: #0056b3; }
            .result { margin: 20px 0; padding: 20px; background: #e9ecef; border-radius: 8px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 Job Classification System</h1>
            <p>API is running! Test the endpoints below:</p>
            
            <div class="test-form">
                <h3>Test Job Classification</h3>
                <input type="text" id="jobTitle" placeholder="Job Title (e.g., Software Developer)">
                <textarea id="skills" placeholder="Skills (e.g., Python, SQL, Machine Learning)"></textarea>
                <textarea id="description" placeholder="Job Description (optional)"></textarea>
                <button onclick="analyzeJob()">Analyze Job</button>
                <div id="result" class="result" style="display: none;"></div>
            </div>
            
            <div style="margin-top: 30px;">
                <h3>API Endpoints:</h3>
                <ul>
                    <li><a href="/api/health">Health Check</a></li>
                    <li><a href="/api/analyze">API Info</a></li>
                </ul>
            </div>
        </div>
        
        <script>
        async function analyzeJob() {
            const jobTitle = document.getElementById('jobTitle').value;
            const skills = document.getElementById('skills').value;
            const description = document.getElementById('description').value;
            
            if (!jobTitle || !skills) {
                alert('Please enter both job title and skills');
                return;
            }
            
            try {
                const response = await fetch('/api/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ jobTitle, skills, description })
                });
                
                const result = await response.json();
                
                document.getElementById('result').style.display = 'block';
                document.getElementById('result').innerHTML = `
                    <h4>Results:</h4>
                    <p><strong>Classification:</strong> ${result.classification}</p>
                    <p><strong>Confidence:</strong> ${(result.confidence * 100).toFixed(1)}%</p>
                    <p><strong>Guidance:</strong> ${result.guidance}</p>
                    <div>
                        <strong>Recommended Skills:</strong>
                        <ul>
                            ${result.recommendations.map(rec => `<li>${rec.name} - ${rec.reason}</li>`).join('')}
                        </ul>
                    </div>
                `;
            } catch (error) {
                document.getElementById('result').innerHTML = '<p style="color: red;">Error: ' + error.message + '</p>';
                document.getElementById('result').style.display = 'block';
            }
        }
        </script>
    </body>
    </html>
    """

@app.route('/api/analyze', methods=['GET', 'POST'])
def analyze_job():
    """API endpoint for job analysis"""
    
    # Handle GET request
    if request.method == 'GET':
        return jsonify({
            'message': 'Job Classification API',
            'usage': 'Send POST request with jobTitle and skills',
            'example': {
                'jobTitle': 'Software Developer',
                'skills': 'Python, JavaScript, SQL',
                'description': 'Optional job description'
            },
            'endpoints': {
                'analyze': '/api/analyze (POST)',
                'health': '/api/health (GET)',
                'home': '/ (GET)'
            }
        })
    
    # Handle POST request
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
            
        job_title = data.get('jobTitle', '').strip()
        skills = data.get('skills', '').strip()
        description = data.get('description', '').strip()
        
        if not job_title or not skills:
            return jsonify({'error': 'Job title and skills are required'}), 400
        
        # Get prediction (with warnings suppressed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            classification, confidence = classifier.predict_classification(job_title, skills, description)
        
        # Get recommendations
        recommendations = classifier.get_skill_recommendations(skills, classification)
        
        # Generate guidance
        guidance_map = {
            'Newly Created': 'This is an emerging role with excellent future prospects. Continue developing cutting-edge skills in this area.',
            'Displaced': 'This role is at high risk of displacement. Focus on acquiring emerging technology skills to transition into new roles.',
            'Reshaped': 'This role is evolving. The recommended skills will help you adapt to changing requirements and stay relevant.'
        }
        
        response = {
            'classification': classification,
            'confidence': confidence,
            'guidance': guidance_map.get(classification, 'Continue developing your skills.'),
            'recommendations': recommendations,
            'jobTitle': job_title,
            'skills': [skill.strip() for skill in skills.split(',')]
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in analyze_job: {e}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': classifier.model is not None,
        'vectorizer_loaded': classifier.vectorizer is not None,
        'label_encoder_loaded': classifier.label_encoder is not None,
        'mode': 'production' if classifier.model else 'demo',
        'endpoints': {
            'home': '/',
            'analyze': '/api/analyze',
            'health': '/api/health'
        }
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🚀 Starting Job Classification API...")
    print("📊 Dashboard: http://localhost:5000")
    print("🔗 API: http://localhost:5000/api/analyze") 
    print("💚 Health Check: http://localhost:5000/api/health")
    print("-" * 50)
    
    # Check model status
    if classifier.model:
        print(" Real model loaded - Production mode")
    else:
        print("  Demo mode - No trained model found")
    
    print("-" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)