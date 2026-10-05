# Research Summary: AI-Powered Symptom Checking and Triage

## Selected Innovation
For this assignment, I selected an **AI-powered symptom checker and triage tool**. The idea is to let a user describe symptoms in normal language and use AI to categorize the description into a level of urgency.

This type of application connects to recent developments in conversational medical AI. Google Research introduced AMIE, a research AI system designed to study diagnostic medical reasoning and medical conversations. It shows how modern AI can process complex symptom descriptions and ask clinically relevant questions. My project is much simpler and does not attempt to reproduce AMIE. Instead, it demonstrates the basic idea using natural language processing and machine learning.

## Why It Matters
Healthcare information is often difficult for patients to understand. A symptom-triage tool could help organize information and guide a person toward an appropriate next step. However, an AI output can be wrong, so it should support rather than replace professional healthcare judgment.

## How My Prototype Works
The prototype uses two common machine-learning techniques:

- **TF-IDF** converts the user's text into numbers based on important words and phrases.
- **Logistic Regression** classifies the text into a triage level and a broad symptom category.

I also added a small rule-based safety layer for several emergency warning phrases. This is important because the training dataset in a class project is far too small to safely detect every medical emergency.

## Data-Centric Issues
### Data quality
Good healthcare AI requires high-quality and representative medical data. A small or inaccurate dataset can produce unreliable results.

### Bias
Bias can occur when training data does not represent different populations or different ways people describe symptoms. This can affect fairness and accuracy.

### Privacy
Health information is sensitive. A real application should minimize data collection, use secure storage and transmission, and follow applicable privacy requirements.

### Safety and human oversight
The World Health Organization emphasizes that AI for health should be designed and governed with ethics and human rights in mind. Recent WHO guidance also addresses governance of large multimodal and generative AI systems in healthcare.

### Regulation
The U.S. FDA maintains a list of AI-enabled medical devices authorized for marketing. This demonstrates that medical AI intended for real clinical use requires a much higher level of evidence and oversight than a classroom prototype.

## Conclusion
This project demonstrates how AI can process natural-language symptom descriptions and provide a basic triage classification. The technical part is relatively simple, but the healthcare context makes safety, privacy, bias, data quality and human oversight very important. The prototype is therefore educational only and should never be treated as a medical diagnosis tool.

## References
- Google Research. “AMIE: A research AI system for diagnostic medical reasoning and conversations.” 2024.
  https://research.google/blog/amie-a-research-ai-system-for-diagnostic-medical-reasoning-and-conversations/
- World Health Organization. “Ethics and governance of artificial intelligence for health.” 2021.
  https://www.who.int/publications/i/item/9789240029200
- World Health Organization. “Ethics and governance of artificial intelligence for health: Guidance on large multi-modal models.” 2025.
  https://www.who.int/publications-detail-redirect/9789240084759
- U.S. Food and Drug Administration. “Artificial Intelligence-Enabled Medical Devices.”
  https://www.fda.gov/medical-devices/digital-health-center-excellence/artificial-intelligence-enabled-medical-devices
