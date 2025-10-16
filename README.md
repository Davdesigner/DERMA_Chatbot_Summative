# DERMA_ChatBot

A healthcare chatbot specialized in dermatology information using t5-small transformer model to provide accurate and helpful information to user queries about skin diseases.

## Overview

This chatbot is designed to provide reliable information about Skin diseases, symptoms, treatments, and prevention strategies. It uses a fine-tuned t5-small model to understand and respond to user questions with medically accurate information.

## Project Demo Video

Link to Demo presentation: [Click Here.](https://youtu.be/fA5zi3bdkCs)

## Features

- Accurate question and response matching using T5
- Comprehensive dermatological information
- Medical terminology explanations
- Prevention and lifestyle advice
- Emergency symptoms awareness
- Interactive and user-friendly interface

## Tech Stack

- **Model**: T5-small
- **Framework**: PyTorch
- **Frontend**: Gradio
- **Language**: Python 3.8+

## Installation

```bash
# Clone the repository
git clone https://github.com/M-Pascal/Chatbot_Summative.git
cd Notebook

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

## Dataset

The Q&A dataset includes:
- Medical questions about skin diseases
- Detailed, accurate answers
- Question patterns for better matching
- Categorical tags for organization

Dataset categories cover:
- Basic skin disease concepts
- Symptoms and diagnosis
- Treatment options
- Prevention strategies
- Emergency situations
- Risk factors

## Project Structure

```
Summative-chatbot/
├── app.py                  # Gradio application
├── Notebook/
│   └── Healthcare_Chatbot_Summative.ipynb 
└── README.md             # Project documentation
```


## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b new_branch`)
3. Commit your changes (`git commit -m 'Add some'`)
4. Push to the branch (`git push origin new_branch`)
5. Open a Pull Request

## Future Improvements

- [ ] Add multi-language support
- [ ] Implement confidence scoring
- [ ] Add more specialized medical topics
- [ ] Add speech-to-text capability
- [ ] Implement regular dataset updates

## Done By:

Owner - [@PascalMugisha](mailto:p.mugisha@alustudent.com)
