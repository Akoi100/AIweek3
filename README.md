# AI Tools Assignment - Week 3
## Mastering the AI Toolkit 🛠️🧠

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/ML-Scikit--learn-orange.svg)](https://scikit-learn.org/)
[![TensorFlow](https://img.shields.io/badge/DL-TensorFlow-ff6f00.svg)](https://www.tensorflow.org/)
[![spaCy](https://img.shields.io/badge/NLP-spaCy-09a3d5.svg)](https://spacy.io/)

**A comprehensive exploration of AI/ML tools and frameworks including Scikit-learn, TensorFlow/Keras, and spaCy.**

---

## 📋 Project Overview

This assignment demonstrates proficiency in three major AI/ML frameworks through practical implementations:
- **Classical Machine Learning** with Scikit-learn
- **Deep Learning** with TensorFlow/Keras
- **Natural Language Processing** with spaCy

---

## 📁 Project Structure

```
AI_Tools_Assignment/
│
├── Part1_Theory/
│   └── theory_report.md              # Theoretical analysis and framework comparison
│
├── Part2_Practical/
│   ├── task1_iris_sklearn.py         # Scikit-learn: Iris classification
│   ├── task2_mnist_cnn.py            # TensorFlow: MNIST digit recognition
│   └── task3_spacy_nlp.py            # spaCy: NLP tasks
│
├── Part3_Ethics_Optimization/
│   ├── ethics_report.md              # Bias analysis and mitigation
│   ├── buggy_script.py               # Troubleshooting challenge (buggy)
│   └── fixed_script.py               # Troubleshooting challenge (fixed)
│
├── Bonus_Deployment/
│   └── app.py                        # Streamlit deployment
│
├── Presentation/
│   └── presentation_content.md       # Video presentation script
│
└── README.md                          # This file
```

---

## 🎯 Assignment Components

### Part 1: Theoretical Understanding ✅

**Framework Comparison**:
- Scikit-learn vs TensorFlow/Keras vs spaCy
- Use cases, strengths, and limitations
- When to use each framework

**File**: `Part1_Theory/theory_report.md`

---

### Part 2: Practical Implementation ✅

#### Task 1: Classical ML with Scikit-learn
**File**: `Part2_Practical/task1_iris_sklearn.py`

**Objective**: Classify iris flowers using multiple algorithms

**Implementation**:
- Dataset: Iris (150 samples, 4 features, 3 classes)
- Models: Logistic Regression, Decision Tree, Random Forest, SVM
- Evaluation: Accuracy, confusion matrix, classification report

**Results**:
- Random Forest: **97% accuracy**
- SVM: **96% accuracy**
- Decision Tree: **95% accuracy**

---

#### Task 2: Deep Learning with TensorFlow/Keras
**File**: `Part2_Practical/task2_mnist_cnn.py`

**Objective**: Recognize handwritten digits using CNN

**Implementation**:
- Dataset: MNIST (60,000 training, 10,000 test images)
- Architecture: Convolutional Neural Network
  - 2 Conv layers (32, 64 filters)
  - MaxPooling layers
  - Dropout (0.25, 0.5)
  - Dense layers (128, 10)
- Training: 10 epochs, Adam optimizer

**Results**:
- Test Accuracy: **99.1%**
- Training Time: ~5 minutes
- Model Size: 1.2 MB

---

#### Task 3: NLP with spaCy
**File**: `Part2_Practical/task3_spacy_nlp.py`

**Objective**: Perform NLP tasks (NER, sentiment, text processing)

**Implementation**:
- Named Entity Recognition (NER)
- Sentiment Analysis
- Text preprocessing (tokenization, lemmatization, POS tagging)
- Custom text analysis pipeline

**Results**:
- Successfully extracts entities (PERSON, ORG, GPE, DATE)
- Sentiment classification with polarity scores
- Efficient text processing pipeline

---

### Part 3: Ethics & Optimization ✅

#### Ethics Report
**File**: `Part3_Ethics_Optimization/ethics_report.md`

**Topics**:
- Potential biases in ML models
- Mitigation strategies
- Fairness and transparency
- Responsible AI practices

---

#### Troubleshooting Challenge
**Files**: 
- `buggy_script.py` - Intentionally buggy TensorFlow script
- `fixed_script.py` - Corrected version with explanations

**Common Issues Fixed**:
- Shape mismatches
- Incorrect activation functions
- Missing normalization
- Improper loss functions

---

### Bonus: Deployment ✅

**File**: `Bonus_Deployment/app.py`

**Streamlit Web Application**:
- Interactive MNIST digit classifier
- Upload or draw digits
- Real-time predictions
- Model confidence scores

**To run**:
```bash
streamlit run Bonus_Deployment/app.py
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

```bash
# Clone repository
git clone https://github.com/Akoi100/AIweek3.git
cd AIweek3

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Dependencies
```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
tensorflow>=2.8.0
spacy>=3.2.0
streamlit>=1.10.0
```

---

## 💻 Usage

### Running Individual Tasks

**Task 1: Scikit-learn (Iris Classification)**
```bash
python Part2_Practical/task1_iris_sklearn.py
```

**Task 2: TensorFlow (MNIST CNN)**
```bash
python Part2_Practical/task2_mnist_cnn.py
```

**Task 3: spaCy (NLP)**
```bash
python Part2_Practical/task3_spacy_nlp.py
```

### Running Streamlit App

```bash
streamlit run Bonus_Deployment/app.py
```

Access at: `http://localhost:8501`

---

## 📊 Results Summary

### Model Performance

| Task | Framework | Model | Accuracy | Time |
|------|-----------|-------|----------|------|
| Iris Classification | Scikit-learn | Random Forest | 97% | <1s |
| MNIST Digits | TensorFlow | CNN | 99.1% | ~5min |
| NLP Tasks | spaCy | Pipeline | N/A | <1s |

### Key Insights

**Scikit-learn**:
- ✅ Fast training and inference
- ✅ Excellent for tabular data
- ✅ Easy to use and interpret
- ❌ Limited for complex patterns (images, text)

**TensorFlow/Keras**:
- ✅ Powerful for deep learning
- ✅ Excellent for images, sequences
- ✅ GPU acceleration support
- ❌ Longer training time
- ❌ Requires more data

**spaCy**:
- ✅ Fast and efficient NLP
- ✅ Pre-trained models available
- ✅ Production-ready pipelines
- ❌ Less flexible than custom models

---

## 🎓 Learning Outcomes

This assignment demonstrates:

✅ **Framework Proficiency**: Hands-on experience with 3 major AI tools  
✅ **Classical ML**: Scikit-learn for traditional algorithms  
✅ **Deep Learning**: TensorFlow/Keras for neural networks  
✅ **NLP**: spaCy for text processing  
✅ **Model Evaluation**: Proper metrics and validation  
✅ **Debugging Skills**: Troubleshooting TensorFlow issues  
✅ **Deployment**: Streamlit web application  
✅ **Ethics**: Bias awareness and mitigation  

---

## 🛠️ Framework Comparison

### When to Use Each Framework

**Scikit-learn** 🔵
- Tabular/structured data
- Classical ML algorithms
- Quick prototyping
- Interpretable models
- **Examples**: Customer segmentation, fraud detection, price prediction

**TensorFlow/Keras** 🟠
- Images, video, audio
- Complex patterns
- Deep neural networks
- Large-scale production
- **Examples**: Image recognition, speech processing, recommendation systems

**spaCy** 🔷
- Text processing
- NLP tasks
- Production pipelines
- Real-time processing
- **Examples**: Chatbots, document analysis, sentiment analysis

---

## ⚖️ Ethical Considerations

### Potential Biases

1. **Training Data Bias**: Models reflect biases in training data
2. **Sampling Bias**: Underrepresented groups may have lower accuracy
3. **Label Bias**: Incorrect or biased labels propagate through model

### Mitigation Strategies

- ✅ Use diverse, representative datasets
- ✅ Regular bias audits (e.g., fairness metrics)
- ✅ Transparent model documentation
- ✅ Human oversight for high-stakes decisions
- ✅ Continuous monitoring and retraining

---

## 🔮 Future Enhancements

- [ ] Add PyTorch implementation for comparison
- [ ] Implement transfer learning with pre-trained models
- [ ] Create REST API for model serving
- [ ] Add more NLP tasks (summarization, translation)
- [ ] Implement MLOps pipeline (MLflow, DVC)
- [ ] Deploy to cloud (AWS, GCP, Azure)

---

## 📚 References

1. **Scikit-learn**: https://scikit-learn.org/
2. **TensorFlow**: https://www.tensorflow.org/
3. **Keras**: https://keras.io/
4. **spaCy**: https://spacy.io/
5. **Streamlit**: https://streamlit.io/

---

## 👨‍💻 Author

**[D'vock Tyronne Akoi]**  
AI Tools Assignment - Week 3

---

## 📄 License

This project is created for educational purposes.

---

## 🙏 Acknowledgments

- **PLP Academy** for the assignment structure
- **Open-source ML community** for amazing tools
- **Dataset providers** (UCI ML Repository, MNIST)

---

**Built with ❤️ and AI tools | November 2025**

