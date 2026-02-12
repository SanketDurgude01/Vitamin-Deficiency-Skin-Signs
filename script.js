// frontend/script.js
const API_URL = 'http://127.0.0.1:5000'; // Flask API address

const imageUpload = document.getElementById('imageUpload');
const uploadBox = document.getElementById('uploadBox');
const imagePreview = document.getElementById('imagePreview');
const uploadText = document.getElementById('uploadText');
const predictButton = document.getElementById('predictButton');
const messageElement = document.getElementById('message');
const resultsSection = document.getElementById('resultsSection');
const predictedClassElement = document.getElementById('predictedClass');
const confidenceScoreElement = document.getElementById('confidenceScore');
const signsList = document.getElementById('signsList');
const recommendationsList = document.getElementById('recommendationsList');
const sourcesList = document.getElementById('sourcesList');
const probabilitiesChart = document.getElementById('probabilitiesChart');
const apiStatusElement = document.getElementById('apiStatus');

let uploadedFile = null;

// --- Utility Functions ---

function displayMessage(text, isError = false) {
    messageElement.textContent = text;
    messageElement.style.color = isError ? 'var(--danger-color)' : 'var(--primary-color)';
}

function checkApiHealth() {
    fetch(`${API_URL}/health`)
        .then(response => response.json())
        .then(data => {
            if (data.status === 'healthy') {
                apiStatusElement.textContent = `Online (Model Loaded: ${data.model_loaded})`;
                apiStatusElement.className = 'status-ok';
            } else {
                apiStatusElement.textContent = 'Demo Mode (Model Not Found)';
                apiStatusElement.className = 'status-ok';
            }
        })
        .catch(error => {
            console.error('API Health Check Error:', error);
            apiStatusElement.textContent = 'Offline (Check Flask Server)';
            apiStatusElement.className = 'status-error';
            predictButton.disabled = true;
        });
}

function renderBarChart(probabilities) {
    probabilitiesChart.innerHTML = ''; // Clear previous chart
    
    // Sort probabilities from highest to lowest
    const sortedClasses = Object.entries(probabilities)
        .sort(([, a], [, b]) => b - a);

    sortedClasses.forEach(([className, probability]) => {
        const percentage = (probability * 100).toFixed(2);
        
        const barContainer = document.createElement('div');
        barContainer.className = 'bar-container';

        const label = document.createElement('span');
        label.className = 'class-label';
        label.textContent = className.replace(/_/g, ' ');
        
        const wrapper = document.createElement('div');
        wrapper.className = 'progress-bar-wrapper';

        const bar = document.createElement('div');
        bar.className = 'progress-bar';
        bar.style.width = `${percentage}%`;
        bar.textContent = `${percentage}%`;
        
        wrapper.appendChild(bar);
        
        barContainer.appendChild(label);
        barContainer.appendChild(wrapper);
        probabilitiesChart.appendChild(barContainer);
    });
}

function updateResults(predictionData) {
    const info = predictionData.vitamin_info;
    
    // Main results
    predictedClassElement.textContent = predictionData.predicted_class.replace(/_/g, ' ');
    predictedClassElement.style.color = predictionData.predicted_class === 'Normal_Skin' ? 'var(--success-color)' : 'var(--danger-color)';
    confidenceScoreElement.textContent = `${(predictionData.confidence * 100).toFixed(2)}%`;

    // Clear lists
    signsList.innerHTML = '';
    recommendationsList.innerHTML = '';
    sourcesList.innerHTML = '';

    // Populate lists
    info.signs.forEach(sign => {
        const li = document.createElement('li');
        li.textContent = sign;
        signsList.appendChild(li);
    });

    info.recommendations.forEach(rec => {
        const li = document.createElement('li');
        li.textContent = rec;
        recommendationsList.appendChild(li);
    });

    info.sources.forEach(source => {
        const li = document.createElement('li');
        li.textContent = source;
        sourcesList.appendChild(li);
    });
    
    // Render probabilities
    renderBarChart(predictionData.all_predictions);

    resultsSection.classList.remove('hidden');
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

// --- Event Handlers ---

function handleFileSelect(file) {
    if (!file) return;

    if (!file.type.match('image.*')) {
        displayMessage('File must be an image (png, jpg, jpeg).', true);
        predictButton.disabled = true;
        return;
    }

    uploadedFile = file;
    displayMessage(`File selected: ${file.name}`, false);
    predictButton.disabled = false;
    
    // Display preview
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        imagePreview.classList.remove('hidden');
        uploadText.classList.add('hidden');
        uploadBox.style.border = '3px solid var(--primary-color)';
    };
    reader.readAsDataURL(file);
}

// Click event
uploadBox.addEventListener('click', () => imageUpload.click());
imageUpload.addEventListener('change', (event) => {
    handleFileSelect(event.target.files[0]);
});

// Drag and drop events
uploadBox.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadBox.style.borderColor = 'var(--primary-color)';
});
uploadBox.addEventListener('dragleave', (e) => {
    e.preventDefault();
    uploadBox.style.borderColor = 'var(--border-color)';
});
uploadBox.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadBox.style.borderColor = 'var(--border-color)';
    handleFileSelect(e.dataTransfer.files[0]);
});

// Prediction button click
predictButton.addEventListener('click', () => {
    if (!uploadedFile) {
        displayMessage('Please upload an image first.', true);
        return;
    }

    resultsSection.classList.add('hidden');
    predictButton.disabled = true;
    displayMessage('Analyzing image... Please wait.', false);

    const formData = new FormData();
    formData.append('file', uploadedFile);

    fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData,
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        if (data.success) {
            updateResults(data.prediction);
            displayMessage('Analysis complete!', false);
        } else {
            displayMessage(`Prediction failed: ${data.error}`, true);
        }
    })
    .catch(error => {
        console.error('Error during prediction:', error);
        displayMessage(`An error occurred: ${error.message}. Is the API server running?`, true);
    })
    .finally(() => {
        predictButton.disabled = false;
    });
});

// Initial check when the page loads
document.addEventListener('DOMContentLoaded', checkApiHealth); 
