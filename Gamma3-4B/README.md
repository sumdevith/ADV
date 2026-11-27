# My Gemma 3 Fine-Tuning Homework: ACLEDA and ABA

_A study note to go with `Gemma3.ipynb`_

ent_Final## What I am doing in this homework

For this assignment, I am using **Gemma 3 4B** to work with Khmer banking questions and answers. I have two datasets: one from **ACLEDA** and another from **ABA**. I want to see what happens when I fine-tune the same starting model on different data.

I am **not training one model on both banks together**. My plan is to finish the ACLEDA experiment first, save its results, and then start a new training run with ABA. The second run uses a fresh copy of Gemma 3, not the model already trained on ACLEDA. After that, I compare their loss graphs and the answers they give to the same questions.

This is the order I follow:

```text
Upload ACLEDA.csv + ABA.csv
           |
           v
Check and clean the two datasets
           |
           v
Make train / validation / test splits
           |
           v
Gemma 3 4B + LoRA --> Train on ACLEDA --> Save model + losses
           |
           v
Fresh Gemma 3 4B + LoRA --> Train on ABA --> Save model + losses
           |
           v
Plot the loss graphs for both runs
           |
           v
Ask both models the same 24 test questions
           |
           v
Compare scores, Khmer answers, and write my conclusion
```

The notebook has **four objectives**, which I explain below. Some parts of this guide describe what the code is set up to do. I have now completed both training runs and recorded the real losses and evaluation scores below. The results are from my Colab experiment, not example numbers.

---

## Objective 1 — Understand the fine-tuning workflow

### 1.1 What is an LLM, and why do I use Gemma 3?

An LLM is a language model that learns patterns from a large amount of text. It can read a question and generate an answer. I am using the **Gemma 3 4B Instruct** model because I want to try a real language-model fine-tuning experiment without training a large model from zero.

The **4B** means the model has roughly four billion parameters. Those parameters already contain knowledge learned during pretraining. In this homework, I only want to adapt the model to the format and language of my banking Q&A data.

### 1.2 What does fine-tuning mean?

I think of fine-tuning as giving the model more examples of the kind of task I want it to perform. In my case, the task is simple: **read a Khmer banking question and generate a Khmer answer**.

Each training example has two main columns:

```text
Question: តើ...?
Answer:   [the related Khmer banking answer]
```

The model uses the question as input and tries to learn the reference answer. It does not mean the model will become a reliable bank assistant immediately. I still need to test its answers, especially if the information is about money or banking services.

### 1.3 Why am I using PEFT and LoRA?

I learned that fully fine-tuning all the weights in a four-billion-parameter model can use a lot of GPU memory. That is why this notebook uses **PEFT**, or _Parameter-Efficient Fine-Tuning_.

**LoRA** (_Low-Rank Adaptation_) is one PEFT method. Instead of changing every original weight, it trains a much smaller set of adapter parameters. This is useful for my Colab experiment because:

- it needs less training memory than updating all model weights;
- I can save the ACLEDA and ABA adapters separately;
- I can keep the original Gemma 3 model as the common starting point.

The notebook also loads the base model in **4-bit** to save memory. LoRA and 4-bit loading help make the experiment more practical on a Colab GPU.

### 1.4 My two CSV datasets

I am using these files:

| Dataset | My CSV filename                    | Q&A rows | Unique answers |
| ------- | ---------------------------------- | -------: | -------------: |
| ACLEDA  | `dataasilida_fully_cleaned(2).csv` |      518 |            508 |
| ABA     | `aba_query(1).csv`                 |    1,344 |            224 |

Both files have `Question` and `Answer` columns. The ABA dataset has more rows, but many of its questions are paraphrases or lead to the same answer. This is something I need to remember later: **more rows do not automatically mean more independent information or a better model**.

### 1.5 Uploading and cleaning the files

In the notebook, I select **both CSV files in the same upload window**. I do not need to upload a ZIP file. The upload code recognizes ACLEDA from `acleda` or `asilida` in the filename and ABA from `aba`.

Before training, the code checks that both columns exist. It removes missing questions/answers, trims extra spaces, and removes repeated question-and-answer pairs.

I clean the data because an empty answer or an accidental duplicate is not useful training information. I also display a small summary table so I can check that the correct files were loaded.

### 1.6 Why do I split the data?

I do not want to train the model and then test it only on the questions it has already seen. That would not be a meaningful test. So each dataset is split into approximately:

| Split      | Target share | How I use it                          |
| ---------- | -----------: | ------------------------------------- |
| Training   |          80% | The model learns from these examples. |
| Validation |          10% | I check the loss while training.      |
| Test       |          10% | I evaluate the model after training.  |

The notebook uses a **group-based split**, so the exact percentages can be a little different.

For ACLEDA, identical questions are kept together. For ABA, rows with the **same answer** are kept together. I use this rule for ABA because several questions can be different ways of asking about the same information. If near-duplicate examples appeared in both training and testing, the result could look better than it really is.

The first graph in the notebook shows how many records went into each split.

---

## Objective 2 — Understand the notebook settings

Before training, I need to understand the settings instead of just pressing **Run all**.

These are the main settings in my notebook:

```python
MODEL = 'unsloth/gemma-3-4b-it'
MAX_LEN, MAX_STEPS, LR = 1024, 100, 1e-4
LORA_R, BATCH, ACCUM = 8, 1, 8
EVAL_STEPS = 10
```

| Setting               |               Value | How I understand it                                    |
| --------------------- | ------------------: | ------------------------------------------------------ |
| Base model            | Gemma 3 4B Instruct | The original model used for both banks.                |
| Maximum length        |        1,024 tokens | Limits the amount of text processed in one example.    |
| Learning rate         |              0.0001 | Controls the size of the training updates.             |
| Training steps        |                 100 | Number of optimizer update steps for each run.         |
| LoRA rank             |                   8 | Controls the size/capacity of the LoRA update.         |
| LoRA alpha            |                  16 | Controls LoRA's scaling.                               |
| Batch size            |                   1 | Processes one example per device batch.                |
| Gradient accumulation |                   8 | Collects gradients across eight small batches.         |
| Validation frequency  |      Every 10 steps | Checks validation loss regularly.                      |
| Training-loss logging |       Every 5 steps | Records the training-loss values for graphs.           |
| Random seed           |                  42 | Helps me repeat the random split and sampling choices. |

My **effective batch size** is `1 × 8 = 8`. I use gradient accumulation because processing a larger batch directly may not fit into the GPU memory.

I keep these hyperparameters **the same for ACLEDA and ABA**. Otherwise, I would not know whether the results changed because of the dataset or because I changed the training configuration.

One limitation is that 100 training steps do **not** represent the same number of full dataset passes for both banks. ABA has more rows than ACLEDA, so the models do not necessarily see the same proportion of their respective datasets.

### How the training code works

The notebook has a function called `train_bank()`. I use the same function twice rather than writing two long training scripts. Each time it:

1. Loads a **fresh** Gemma 3 4B base model.
2. Adds a new LoRA adapter.
3. Converts the CSV questions and answers into the Gemma chat format.
4. Creates a supervised fine-tuning trainer.
5. Trains on the bank's training split and checks the validation split.
6. Saves the loss history and LoRA adapter in that bank's own folder.
7. Clears the model from GPU memory before the next run.

The **Gemma chat template** is important because it marks where a user's question ends and the model's answer begins. The notebook uses response-only training, so the loss is focused on the assistant's answer tokens.

---

## My training order: ACLEDA first, then ABA

### Step 1: Train on ACLEDA

First, I run:

```python
train_bank('acleda', train_a, val_a)
```

During this step, the model learns from ACLEDA training questions. The notebook checks validation loss every 10 steps and records training loss every 5 steps. At the end, it saves:

```text
/content/gemma_bank_homework/acleda_history.csv
/content/gemma_bank_homework/acleda_adapter/
```

The history CSV is needed for my graph. The adapter contains the trained LoRA weights.

### Step 2: Train on ABA

Only after ACLEDA finishes, I run:

```python
train_bank('aba', train_b, val_b)
```

This does **not** continue training the ACLEDA adapter. The code loads the **original Gemma 3 model again**, attaches a new LoRA adapter, and trains using ABA examples. It saves:

```text
/content/gemma_bank_homework/aba_history.csv
/content/gemma_bank_homework/aba_adapter/
```

Now I have two different adapters trained from the same starting model. This is the setup I need for my practical experiment.

**Small note:** If the notebook finds a completed history file and adapter for a bank, it skips training that bank again. This helps prevent me from repeating a finished run by accident.

---

## Objective 3 — Study the training and validation loss

### What does loss tell me?

Loss is a way of measuring how far the model's predictions are from the reference answers during training. In general, I want the training loss to decrease. But I also need to look at **validation loss**, because a model can fit its training questions well and still perform badly on unseen questions.

Here is how I plan to read my graphs:

| What I see                                          | What it might mean                                           |
| --------------------------------------------------- | ------------------------------------------------------------ |
| Training and validation loss both decrease          | The model may be learning and generalizing.                  |
| Training loss decreases but validation loss goes up | Possible **overfitting**.                                    |
| Both losses remain high and barely improve          | Possible **underfitting** or a training problem.             |
| Loss changes up and down a lot                      | I should inspect the data, learning rate, and training logs. |

I should not decide based on one point. These are patterns to investigate, not a guarantee of what is happening.

### The loss graphs in my notebook

After **both training runs** are complete, the notebook makes:

- **`02_each_bank_loss.png`** — one ACLEDA panel and one ABA panel, each showing training and validation loss.
- **`03_loss_comparison.png`** — one graph with ACLEDA and ABA curves together.
- **`loss_summary.csv`** — the first, last, and best recorded loss values.

The combined chart is useful for comparing how the curves change across training steps. But I need to be careful: **ACLEDA loss and ABA loss are calculated on different datasets**. The lower loss number does not automatically mean the better overall banking assistant. That is why I also test both models on the _same_ questions in Objective 4.

### My observations after running Colab

Both training runs finished successfully on the T4 GPU. I used separate LoRA adapters, so the second run did not continue from the ACLEDA-trained model. The numbers in this section are the values recorded by my experiment.

| Dataset | First training loss | Last training loss | First validation loss | Last validation loss | Best validation loss |
| ------- | ------------------: | -----------------: | --------------------: | -------------------: | -------------------: |
| ACLEDA  |              3.5569 |             1.2764 |                2.0905 |               1.2106 |               1.2106 |
| ABA     |              5.3398 |             0.9956 |                2.9427 |               1.6450 |               1.6424 |

**What I noticed with ACLEDA:** Its training loss went down from 3.5569 to 1.2764. Its validation loss also fell from 2.0905 to 1.2106. The last validation value was also the best recorded value. For this training run, I do not see clear evidence of overfitting from these summary numbers. The model was still improving on validation examples at the end.

**What I noticed with ABA:** Its training loss went down from 5.3398 to 0.9956, which is a large decrease. Validation loss also improved, from 2.9427 to 1.6450. The best validation loss was 1.6424 at step 90, followed by 1.6450 at step 100. That small increase is worth checking, but it is too small on its own to prove overfitting. The larger gap between the final training loss and validation loss also suggests I should watch generalization carefully.

**Overfitting or underfitting?** I cannot say either model has clearly overfit or underfit just from these numbers. ABA shows a possible early sign of a training/validation gap. Both models reduced their validation loss, so there is evidence that they learned something from the data. But a decreasing loss does not automatically mean the generated Khmer answers are accurate. I need to look at the test results as well.

Another point I learned: ACLEDA and ABA have different questions, answers, and data distributions. Their loss values are **not directly comparable as a model-quality ranking**. I can compare the _shapes_ of the learning curves, but I use the same held-out questions for a more meaningful comparison of generated answers.

---

## Objective 4 — Fine-tune on another dataset and compare

This is the part where I answer my teacher's practical-experiment question. I have trained Gemma 3 with ACLEDA and then with ABA, using separate adapters. Now I want to find out how their answers differ.

### Why do both models answer the same questions?

If I test the ACLEDA model only on ACLEDA questions and the ABA model only on ABA questions, the scores would be hard to compare. So the notebook chooses a small **24-question test set**:

- **12 ACLEDA questions** from its held-out test split.
- **12 ABA questions** from its held-out test split, with different reference answers.

**Both models answer all 24 questions.** In other words, the ACLEDA-trained model also tries ABA questions, and the ABA-trained model also tries ACLEDA questions.

### How I measure their answers

The notebook gives three scores:

**Exact match:** Does the generated answer match the reference answer exactly after basic spacing cleanup? This is strict; a correct answer written in different words can still get zero.

**Character 3-gram recall:** How many of the reference answer's three-character chunks appear in the generated answer?

**Character 3-gram F1:** A balance of precision and recall for those three-character chunks:

\[
F1 = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
\]

These are useful for a small class experiment, but they mainly measure **text overlap**. They do not prove an answer is factually correct, especially for banking information. I still need to read a few Khmer answers myself.

### The comparison graphs

The notebook makes two model-performance charts:

**`04_F1_by_bank.png`** shows both models' F1 scores separately on ACLEDA questions and ABA questions. I can use it to see whether each model performs better in its own banking domain.

**`05_overall_scores.png`** shows the average exact match, recall, and F1 of the two models on the same 24 questions.

I also get `answers_side_by_side.csv`, which lets me compare what each model actually wrote for the same customer question.

### What would count as a reasonable conclusion?

I should look at the **graphs, the numbers, and actual answers together**. For example, a model might get a lower exact-match score simply because it uses different Khmer wording. On the other hand, it might have a high text-overlap score while still giving a wrong banking detail.

This experiment is quite small: only 24 comparison questions, 100 training steps per model, and datasets that differ in both size and topics. So I should explain what I observed **in this experiment**, not claim that one model is always better in every situation.

### My actual model comparison results

For this part, I used 24 held-out questions: 12 from ACLEDA and 12 from ABA. Both adapters answered all 24 questions. The metrics are exact text match, character 3-gram recall, and character 3-gram F1. F1 here is a **text-overlap score**, not a percentage of factually correct banking answers.

| Test question source | Trained model     | Exact match | Recall |        F1 |
| -------------------- | ----------------- | ----------: | -----: | --------: |
| ACLEDA               | Trained on ACLEDA |       0.000 |  0.158 | **0.151** |
| ACLEDA               | Trained on ABA    |       0.000 |  0.079 |     0.119 |
| ABA                  | Trained on ACLEDA |       0.000 |  0.361 | **0.302** |
| ABA                  | Trained on ABA    |       0.000 |  0.268 |     0.272 |

| Model             | Overall exact match | Overall recall | Overall F1 |
| ----------------- | ------------------: | -------------: | ---------: |
| Trained on ACLEDA |               0.000 |      **0.260** |  **0.227** |
| Trained on ABA    |               0.000 |          0.173 |      0.196 |

**What surprised me:** The ACLEDA-trained model did slightly better on **both** question groups, including the ABA questions. Its overall F1 was 0.227 compared with 0.196 for the ABA-trained model. I expected the ABA model to perform better on its own questions, especially because its CSV has more rows. My experiment did not show that. A larger number of rows does not automatically mean better generalization. The ABA dataset contains 1,344 rows but only 224 distinct reference answers, with many paraphrases, so the row count may overstate the amount of different information the model learned.

**What the exact-match score means:** Both models got 0 exact match. That means none of the generated answers matched the full reference answer exactly after basic whitespace cleanup. It does **not** mean all 24 answers were necessarily completely wrong: a correct answer with different Khmer wording can also get zero. Still, the F1 scores are fairly low, so the quality needs more checking.

**Looking at example answers:** In question Q02 about the functions of an electronic wallet, neither model's displayed answer seems to directly list the functions in the reference answer. This is a reminder that generated text can sound fluent but still miss the question. I should manually review more full-length Khmer predictions and check any banking facts before using a model in a real service.

### My final conclusion

For this homework, I fine-tuned Gemma 3 4B with LoRA twice. I started with the ACLEDA dataset and, after saving that adapter, trained another fresh model with ABA. I used the same hyperparameters for both runs. Their training and validation losses decreased, which tells me the fine-tuning process was learning from each dataset.

When I tested both adapters on the same 24 held-out questions, the ACLEDA-trained model had the higher overall character 3-gram F1: **0.227**, compared with **0.196** for the ABA-trained model. It also scored higher on both ACLEDA and ABA questions in this particular test. However, neither model matched any reference answer exactly, and the text-overlap scores were not strong. I cannot conclude that either model is ready to answer real customer questions reliably.

What I learned is that **training loss alone is not enough to judge a model**. I need validation loss, a common test set, and a manual check of the generated answers. If I improve this project later, I want to review the quality of the Q&A data, test with more questions, and compare a few training settings. I would also check factual correctness rather than depending only on text overlap.

_These observations are drafted from my recorded results. Before submitting, I should read them and edit them to match what I personally noticed in the full graphs and Khmer answers, and follow my course's AI-assistance policy._

---

## How I run everything in Google Colab

I follow these steps in order:

1. Open `Gemma3_ACLEDA_Then_ABA_Short_Student_Final.ipynb` in Google Colab and select a **GPU runtime** (such as T4).
2. Run the installation cell. The notebook installs Unsloth, TRL, PEFT, datasets and the other packages it needs.
3. At the upload cell, select **both CSV files at the same time**: `dataasilida_fully_cleaned(2).csv` and `aba_query(1).csv`.
4. Check the dataset table. Make sure both files have loaded with `Question` and `Answer` columns.
5. Run the split and settings cells. Check the first dataset-size graph.
6. Run the **ACLEDA training** cell and confirm its history and adapter were saved.
7. Run the **ABA training** cell. This starts a fresh model and saves a separate history and adapter.
8. Run the loss-analysis cells. Look at both individual graphs and the combined loss graph.
9. Run the evaluation cells. Both adapters answer the same 24 held-out questions.
10. Check F1, exact match, recall, and side-by-side Khmer answers. Write my own observations and conclusion.
11. Download the result files and the executed `.ipynb` notebook for my submission.
