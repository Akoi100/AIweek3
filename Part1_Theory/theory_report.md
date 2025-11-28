# Part 1: Theoretical Understanding

## 1. Short Answer Questions

### Q1: Explain the primary differences between TensorFlow and PyTorch. When would you choose one over the other?

**TensorFlow:**
*   **Paradigm:** Historically used a static computation graph (Define-and-Run), though TensorFlow 2.0 introduced eager execution (Define-by-Run) to be more pythonic. It is often seen as more production-ready with a mature ecosystem for deployment (TFX, TF Serving).
*   **API:** Keras is the high-level API, making it very accessible for beginners.
*   **Debugging:** Can be harder to debug in graph mode, though eager execution has improved this.

**PyTorch:**
*   **Paradigm:** Uses a dynamic computation graph (Define-by-Run). The graph is built on the fly as code is executed.
*   **API:** More "Pythonic" and flexible. It feels like writing standard Python code.
*   **Debugging:** Easier to debug using standard Python tools (pdb, print statements) because of its dynamic nature.

**When to choose which:**
*   **Choose PyTorch:** For research, rapid prototyping, and when you need dynamic graphs (e.g., for complex NLP models or varying input lengths). It is highly favored in the academic and research community.
*   **Choose TensorFlow:** For large-scale production deployments, mobile/edge devices (TensorFlow Lite), and web (TensorFlow.js). If you prefer the Keras high-level abstraction for quick model building, TF is also a good choice.

### Q2: Describe two use cases for Jupyter Notebooks in AI development.

1.  **Exploratory Data Analysis (EDA):** Jupyter Notebooks allow data scientists to interactively load data, visualize it using libraries like Matplotlib or Seaborn, and clean it step-by-step. The ability to see immediate output (charts, tables) after running a cell makes it ideal for understanding the data before modeling.
2.  **Model Prototyping and Documentation:** They are excellent for experimenting with different model architectures and hyperparameters. You can combine code, markdown text, and mathematical equations (LaTeX) in a single document, making it a "computational narrative" that explains the *why* and *how* of the experiment, which is crucial for collaboration and reproducibility.

### Q3: How does spaCy enhance NLP tasks compared to basic Python string operations?

*   **Linguistic Knowledge:** Basic Python string operations (split, replace, regex) treat text as raw characters. spaCy understands the *structure* and *meaning* of language. It has built-in models for tokenization, part-of-speech (POS) tagging, dependency parsing, and named entity recognition (NER).
*   **Efficiency:** spaCy is written in Cython and is designed for production use. It is incredibly fast and efficient at processing large volumes of text compared to writing custom Python loops.
*   **Contextual Understanding:** spaCy can identify that "Apple" is an organization in "Apple bought a startup" but a fruit in "I ate an apple," whereas basic string matching would struggle with this context.

## 2. Comparative Analysis: Scikit-learn vs. TensorFlow

| Feature | Scikit-learn | TensorFlow |
| :--- | :--- | :--- |
| **Target Applications** | **Classical Machine Learning:** Best for structured/tabular data. Algorithms include Regression, SVM, Random Forests, K-Means, PCA. Not designed for deep learning. | **Deep Learning & Neural Networks:** Best for unstructured data (images, text, audio). Supports CNNs, RNNs, Transformers, and complex custom architectures. |
| **Ease of Use for Beginners** | **High:** Very consistent API (`fit`, `predict`, `transform`). Excellent documentation and fewer hyperparameters to tune to get a working baseline. Ideal for learning ML fundamentals. | **Moderate to Hard:** TensorFlow 2.0 (via Keras) is easier than v1, but deep learning concepts (tensors, gradients, layers) are inherently more complex. Setting up environments (CUDA/cuDNN) can also be challenging. |
| **Community Support** | **Mature & Stable:** Huge community for traditional data science. extensive tutorials, and integration with the PyData stack (Pandas, NumPy). | **Massive & Active:** Dominates the deep learning space (alongside PyTorch). extensive resources for state-of-the-art models, pre-trained models (TensorFlow Hub), and enterprise support from Google. |
