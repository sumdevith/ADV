# Gemma 3 Fine-Tuning

In this homework, I worked with Gemma 3 4B to learn how to fine-tune a large language model using Khmer banking questions and answers. I used two datasets, ACLEDA and ABA, to understand how different training data affects model performance.

### 1. Course Summary

I learned the basic workflow of LLM fine-tuning, including data preparation, tokenization, training and evaluation. I also learned about LoRA and PEFT, which allow us to fine-tune a pretrained model without updating all its parameters.

### 2. Notebook Configuration

I used Google Colab with a GPU and configured important hyperparameters such as learning rate, batch size, LoRA rank and training steps. I trained the model on ACLEDA first and then trained a separate model on ABA using the same configuration.

### 3. Loss Analysis

After training, I analyzed the training and validation losses. Both models showed decreasing losses, which means they learned from the training data. ACLEDA finished with a validation loss of **1.2106**, while ABA finished with **1.6450**. I also learned how differences between training and validation loss may help identify overfitting or underfitting.

### 4. Practical Experiment

I compared the two trained models using the same test questions and evaluation metrics. The ACLEDA-trained model achieved an overall F1 score of **0.227**, while the ABA-trained model achieved **0.196**. Although ABA had a lower final training loss, ACLEDA performed slightly better in this small evaluation.

### Conclusion

This homework helped me understand that fine-tuning is not only about training a model until the loss decreases. Dataset quality, model configuration and evaluation are also important. I learned how to compare models using real results and identify areas that still need improvement.
