# AI-Powered Symptom Checker and Triage Tool

## Project Overview
This project is a simple AI-driven healthcare application created for a course assignment. It accepts a short symptom description written in natural language and uses machine learning to suggest:

1. A **triage level**: Emergency, Urgent, or Self-care
2. A **broad symptom pattern** such as respiratory infection, allergy, or cardiac warning pattern

The application is an educational prototype only. It is **not a medical device** and does not provide a real diagnosis.

## Why I Selected This Innovation
AI-based symptom assessment is a useful example of how natural language processing can support healthcare access. Recent work in conversational medical AI, including systems such as Google Research's AMIE, shows how large language models can engage in diagnostic-style conversations. At the same time, healthcare AI needs strong safety, privacy, bias, and governance controls.

For this class project, I created a much smaller and safer demonstration using TF-IDF and Logistic Regression rather than a large language model. I also added an emergency red-flag rule layer.

## Main Features
- Free-text symptom input
- NLP text vectorization using TF-IDF
- Machine-learning classification using Logistic Regression
- Triage categories: Emergency / Urgent / Self-care
- Broad symptom-pattern classification
- Emergency red-flag safety rules
- Streamlit web interface
- Clear medical disclaimer and limitations

## How the AI Works
The application contains a small demonstration training set. Symptom sentences are converted into numerical features using TF-IDF. Two Logistic Regression models are trained:

- Model 1 predicts the triage level
- Model 2 predicts a broad symptom pattern

Before the model result is displayed, the app checks for a small set of emergency phrases. If an emergency phrase is found, the app displays an emergency warning.

## Installation

### 1. Clone or download this repository
```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd ai-healthcare-symptom-triage
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

### 3. Install packages
```bash
pip install -r requirements.txt
```

### 4. Run the application
```bash
streamlit run app.py
```

The browser should open automatically.

## Example Inputs
Try:
- `I have a high fever, cough, and I feel very weak.`
- `I have itchy eyes, sneezing and a runny nose.`
- `I have chest pain, shortness of breath and sweating.`

## Ethical and Data-Centric Considerations
### Data quality
The accuracy of a healthcare AI system depends on the quality and representativeness of its data. This course demo uses a tiny hand-built dataset, so it is not suitable for clinical use.

### Bias
A training dataset can underrepresent certain populations, symptom descriptions, ages, languages, or health conditions. A real system would require systematic fairness testing.

### Privacy
Real health applications must protect sensitive health information. This demo does not intentionally save user-entered symptom text.

### Safety and uncertainty
Medical AI can make incorrect predictions. A production system would require clinical validation, monitoring, human oversight, and careful handling of uncertainty.

### Regulation
AI systems used as medical devices may be subject to regulatory requirements. The U.S. FDA maintains a public list of authorized AI-enabled medical devices.

## Research Sources
1. Google Research. **AMIE: A research AI system for diagnostic medical reasoning and conversations** (2024).  
   https://research.google/blog/amie-a-research-ai-system-for-diagnostic-medical-reasoning-and-conversations/

2. World Health Organization. **Ethics and governance of artificial intelligence for health** (2021).  
   https://www.who.int/publications/i/item/9789240029200

3. World Health Organization. **Ethics and governance of artificial intelligence for health: Guidance on large multi-modal models** (2025).  
   https://www.who.int/publications-detail-redirect/9789240084759

4. U.S. Food and Drug Administration. **Artificial Intelligence-Enabled Medical Devices**.  
   https://www.fda.gov/medical-devices/digital-health-center-excellence/artificial-intelligence-enabled-medical-devices

## Disclaimer
This application is for educational purposes only. It does not diagnose disease, replace medical advice, or provide a clinically validated triage decision. For a medical emergency, call 911 in the United States or the appropriate local emergency number.
