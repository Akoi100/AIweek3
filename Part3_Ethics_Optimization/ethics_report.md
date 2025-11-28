# Part 3: Ethics & Optimization

## 1. Ethical Considerations

### Potential Biases
**MNIST (Handwritten Digits):**
*   **Geographic/Cultural Bias:** The MNIST dataset was collected from American Census Bureau employees and high school students. Handwriting styles vary significantly across cultures (e.g., the way the number '7' or '1' is written in Europe vs. the US). A model trained only on MNIST might perform poorly on digits written by people from other regions.
*   **Representation:** While less critical for digits, if the dataset were faces, lack of diversity (race, gender, age) would be a major ethical issue leading to discriminatory performance.

**Amazon Reviews (NLP):**
*   **Selection Bias:** Reviews are often written by a specific demographic of users who are motivated to write reviews (either very happy or very angry). This doesn't represent the average user.
*   **Language Bias:** Models trained on standard English might fail to understand AAVE (African American Vernacular English), dialects, or non-native English speakers, potentially misclassifying their sentiment or flagging them as "toxic" incorrectly.

### Mitigation Strategies
**TensorFlow Fairness Indicators:**
*   This tool allows us to compute performance metrics (accuracy, false positive rate) sliced by different groups (e.g., "European handwriting" vs "American handwriting" if we had those labels).
*   We can visualize if the model is underperforming for a specific group and take action (e.g., collecting more data for that group).

**spaCy's Rule-Based Systems:**
*   **Transparency:** Unlike "black box" deep learning models, rule-based systems are transparent. We know exactly *why* a sentence was classified as negative (e.g., it contained the word "terrible").
*   **Bias Correction:** If we find the model is biased against a certain term (e.g., incorrectly flagging a cultural term as negative), we can manually adjust the rules or stop-lists to fix it immediately without retraining a massive model.

## 2. Troubleshooting Challenge

See `buggy_script.py` for the code with errors and `fixed_script.py` for the corrected version.
