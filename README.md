# Efficient User Intent Classification with Machine Learning and Embeddings

This repository contains the implementation of an efficient and fast user intent classification system using an ensemble of logistic regression, SVM, and k-NN classifiers. The model leverages text embeddings from the `jinaai/jina-embeddings-v2-base-es` to provide high accuracy while being resource-efficient compared to large language models (LLMs).

[Medium (free) story link (for more details)](https://medium.com/@prudant/d83ef2251f1f)

## Limitations
This model was train on spanish text corpus and only for text that represents question or requests, it wont perform well with multi turn chathistory or multiline long texts, it's recommended for 128< len token texts.

`If you build a larger dataset with longer text secuences or multi turn conversations and train the model, it should be work pretty well, the Jina embedding model support up to 8k tokens =)`

## Performance Note
One of the key advantages of this model is that it does not require a GPU to run efficiently. The ensemble classifier, leveraging logistic regression, SVM, and k-NN, provides more than decent performance on a CPU. This makes it a viable and cost-effective alternative to running large language models or transformers in production, which typically require significant computational resources and higher costs. This approach ensures that the model is both accessible and practical for real-time applications without the need for specialized hardware.

This approach is also useful and more practical than training a BERT or SBERT classifier, as the embeddings and this ensemble do not require significant computational power. By leveraging pre-trained embeddings and an efficient ensemble of traditional machine learning models, we achieve high accuracy without the need for extensive computational resources.

## Overview

In this project, I address the challenge of user intent classification in conversational AI pipelines, particularly for *spanish* language retrieve-augmented generation (RAG) systems. By combining multiple machine learning algorithms and calibrating their probabilities, the ensemble model achieves remarkable performance. This approach is designed to be significantly faster and more cost-effective than LLMs, making it suitable for real-time applications.

### Intents Supported

- Censorship
- Others
- Lead
- Contact
- Directions
- Meet
- Negation
- Affirmation
- Casual Chat

### Model Results

| Intent        | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| Afirmación    | 1.00      | 1.00   | 1.00     | 14      |
| Censura       | 0.99      | 1.00   | 0.99     | 539     |
| Charla        | 1.00      | 0.67   | 0.80     | 15      |
| Contacto      | 0.97      | 1.00   | 0.99     | 38      |
| Direcciones   | 1.00      | 1.00   | 1.00     | 71      |
| Lead          | 0.99      | 0.99   | 0.99     | 140     |
| Meet          | 0.97      | 1.00   | 0.98     | 29      |
| Negación      | 1.00      | 0.94   | 0.97     | 18      |
| Otros         | 0.98      | 0.97   | 0.98     | 171     |
| **Micro Avg** | 0.99      | 0.99   | 0.99     | 1035    |
| **Macro Avg** | 0.99      | 0.95   | 0.97     | 1035    |
| **Weighted Avg** | 0.99  | 0.99   | 0.99     | 1035    |

### Conclusion
This project showcases a fast and efficient method for user intent classification using an ensemble of machine learning models and text embeddings. While the current model achieves high accuracy, there is always room for improvement, especially in enhancing the training dataset for better generalization.

### Contributions
I invite you to clone the repository, test the model, and contribute to improving the dataset and model performance. Your feedback and suggestions are highly appreciated.

### Follow and Support
If you found this project helpful, please give it a star and follow me for more insights on efficient machine learning techniques and natural language understanding innovations. Let's collaborate and push the boundaries of what's possible in user intent classification.

From Latam with ❤️
