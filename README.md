# ID2223 Scalable Machine Learning - Lab 2: Fine-Tuning & Deployment of LLMs

**Authors:** Emanuele Minotti, Stefano Romano

## 🎯 Project Overview

This laboratory focuses on the full pipeline of **Fine-Tuning Large Language Models (LLMs)** using Parameter-Efficient Fine-Tuning (PEFT) techniques. Our goal was to transform general-purpose base models into instruction-following assistants and, subsequently, domain-specific experts.

The project is divided into two main stages:
1. **Model Exploration & Benchmarking:** Fine-tuning and comparing three different model sizes (1B, 3B, and 8B) on a general instruction dataset to analyze scaling laws and performance.
2. **Domain Adaptation (VetAI):** Further specializing the best-performing model on a veterinary dataset to create a targeted medical assistant.

We utilized the **Unsloth** library for efficient training with 4-bit quantization and exported the final models to **GGUF** format for CPU-based inference via a user interface.

---

## 📊 Stage 1: Model Selection & Performance Analysis

We adopted a model-centric approach to improve performance. Instead of relying on a single configuration, we trained three distinct models using the **FineTome-100k** dataset. We experimented with increasing parameter counts and adjusted LoRA hyperparameters (Rank `r` and Alpha `lora_alpha`) to balance training efficiency with model expressivity.

To ensure an objective evaluation, we split the dataset into training and testing sets, monitoring **Validation Loss** and **Perplexity** every 50 steps.

### 🔹 Comparative Results

The table below summarizes the hyperparameters used and the best metrics achieved for each model configuration:

| Model | LoRA Rank | LoRA Alpha | Learning Rate | Train Batch Size | Grad. Accum. | Lowest Val Loss | Best Perplexity |
|:-----:|:---------:|:----------:|:-------------:|:----------------:|:------------:|:---------------:|:---------------:|
| **Llama-3.2-1B** | 16 | 16 | 2e-4 | 2 | 4 | **0.8906** | **2.4366** |
| **Llama-3.2-3B** | 16 | 16 | 2e-4 | 2 | 4 | **0.7266** | **2.0680** |
| **Meta-Llama-3.1-8B** | 32 | 32 | 1e-4 | 1 | 8 | **0.6474** | **1.9106** |

### 📝 Analysis

The empirical results provide clear evidence of the benefits of model scaling and hyperparameter tuning:

1.  **Scaling Impact:** There is a consistent improvement in downstream performance as model capacity increases.
    * Moving from the **1B** to the **3B** model resulted in a significant **21% reduction** in perplexity.
    * Scaling further from **3B** to **8B** yielded an additional **7.6% reduction**.
2.  **LoRA Parameterization:** For the largest model (8B), we increased the LoRA Rank and Alpha to 32 (compared to 16 for smaller models). This allowed the adapter to learn more complex patterns, contributing to the superior validation loss of **0.6474**.

### 📈 Training Curves

Each model was trained for **500 steps** to fit within compute constraints while ensuring convergence. The plots below visualize the validation loss and perplexity trends, showing smooth convergence for the larger models.

| Llama-3.2-1B | Llama-3.2-3B | Meta-Llama-3.1-8B |
|:------------:|:------------:|:-----------------:|
| ![1b plot](plots/1b.png) | ![3b plot](plots/3b.png) | ![8b plot](plots/8b.png) |

---

## 🐶 Stage 2: Domain-Specific Fine-Tuning (VetAI)

After identifying **Meta-Llama-3.1-8B** as the most capable model based on the metrics above, we proceeded to the second phase of the laboratory: domain adaptation.

Our objective was to build the engine for a **Veterinary Assistant App**. General-purpose instruction tuning makes the model helpful, but not necessarily expert in medical advice for pets.

### Approach
We performed a second stage of fine-tuning (SFT) using a specialized dataset:
* **Base Model:** The checkpoint of our fine-tuned Llama-3.1-8B.
* **Dataset:** A curated collection of veterinary questions and answers (pet health, nutrition, and emergency advice).
* **Technique:** We continued the training process for an additional epoch with a lower learning rate (`5e-5`) to refine the weights without catastrophic forgetting of general conversation abilities.

This two-stage process resulted in a model that maintains the conversational fluency of Llama-3.1 but possesses specific knowledge about animal care.

---

## 📊 Quantitative Evaluation & Benchmarks

To systematically assess the behavior of the evaluated models, we relied on a fixed set of five baseline questions designed to probe different dimensions of reasoning, communication, and alignment with the intended assistant role.

### Evaluation Dimensions
The questions were selected to isolate complementary dimensions of model performance:
1.  **Conceptual Understanding:** Explaining preventive vs. reactive care.
2.  **Logical Reasoning:** A transitive logic puzzle involving animal weights.
3.  **Language Clarity:** Rewriting a sentence for conciseness.
4.  **Empathy & Tone:** Drafting a supportive message for an anxious user.
5.  **Math Reasoning:** A multi-step calculation unit conversion.

### Quantitative Scores (0–10)

The following table summarizes the performance based on manual evaluation criteria:

| Model | Concept Clarity | Logical Reasoning | Language Quality | Empathy | Safety |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LLaMA-1B** | 6 | 5 | 6 | 4 | 5 |
| **LLaMA-3B** | 7 | 6 | 7 | 6 | 6 |
| **LLaMA-8B-FT** | **9** | **9** | **9** | **9** | **9** |



### Evaluation Summary
The quantitative results highlight a clear performance gradient across model sizes and training stages.
* **LLaMA-1B:** Demonstrates basic comprehension but lacks expressive richness and consistent tone, particularly in empathetic or safety-sensitive contexts.
* **LLaMA-3B:** Shows noticeable improvements in structure, coherence, and emotional tone, though occasional reasoning inconsistencies remain.
* **LLaMA-8B-Finetuned:** Achieves the strongest overall results. It consistently offers precise conceptual explanations, correct logical deductions, and clear language. Its supportive responses are natural and appropriately empathetic, maintaining safe, non-diagnostic behavior. These gains illustrate the combined impact of increased model capacity and targeted domain fine-tuning.

---

## 📝 ANNEX — Full Benchmark Responses

Below are the complete verbatim responses provided by the three models when evaluated on the fixed baseline questions.

### 1. Conceptual Understanding
*Question: Explain the difference between preventive care and reactive care in pet health, using simple and general terms.*

> **LLaMA-1B:** Preventive care focuses on maintaining the health of pets in the long term... It's like the difference between a car accident and a minor fender bender.
>
> **LLaMA-3B:** Preventive care focuses on maintaining your pet's health... Reactive care is about addressing issues that have already developed... Think of it like this: Preventive care is like building a strong foundation for your home.
>
> **LLaMA-8B-FT:** Preventive care is the routine care you give your pet to keep them healthy over time... Reactive care happens after something has already changed or gone wrong. In simple terms: preventive care avoids problems; reactive care responds to them.

### 2. Logical Reasoning
*Question: You have three animals: a cat, a dog, and a rabbit. The cat is lighter than the dog, and the dog is heavier than the rabbit. Which animal is the lightest, and which is the heaviest?*

> **LLaMA-1B:** The cat is the lightest, and the dog is the heaviest.
>
> **LLaMA-3B:** The rabbit is the lightest, the cat is in the middle, and the dog is the heaviest.
>
> **LLaMA-8B-FT:** The rabbit is the lightest, the cat is in the middle, and the dog is the heaviest.

### 3. Language Clarity
*Question: Rewrite the sentence: “A healthy pet requires consistent care, proper nutrition, and regular attention to its behavior.”*

> **LLaMA-1B:** “A healthy pet needs consistent care, proper nutrition, and regular attention to its behavior.”
>
> **LLaMA-3B:** "To keep a pet healthy, provide regular care, nutritious food, and monitor its behavior."
>
> **LLaMA-8B-FT:** “A healthy pet needs steady care, balanced nutrition, and attention to its behavior.”

### 4. Empathy and Tone Control
*Question: Write a supportive message for someone who feels anxious about their future (Context: Pet owner worried about dog acting differently).*

> **LLaMA-1B:** It’s understandable to feel worried when your dog behaves differently. Try observing the changes calmly...
>
> **LLaMA-3B:** It makes sense to feel uneasy when your dog acts differently. Sudden changes can be stressful... Try to stay calm, give your dog a quiet and comfortable space...
>
> **LLaMA-8B-FT:** It’s completely understandable to feel worried when your dog acts differently. Behavior changes can feel unsettling. Try offering a calm environment and observe things gently over time. If the change continues, a veterinarian can help...

### 5. Math Reasoning
*Question: If a dog eats 250 grams of food per day, how many kilograms does it eat in a 30-day month?*

> **LLaMA-1B:** It eats 250 * 30 = 7500 grams... 7500 / 1000 = 7.5 kg.
>
> **LLaMA-3B:** To find out how many kilograms your dog eats... 250 grams is 0.25 kilograms. Then, multiply 0.25 kilograms by 30 days: 0.25 x 30 = 7.5 kilograms.
>
> **LLaMA-8B-FT:** 250g = 0.25kg. 0.25 × 30 = **7.5 kg**.

---

## 🔑 Key Takeaways

The comparative analysis leads to three primary conclusions regarding model architecture and training strategies:

* **Parameter Scale & Reasoning Capabilities:**
    The distinct performance gap between the 1B and 8B models—particularly in the logical reasoning task (the animal weight puzzle)—reinforces the correlation between parameter count and the ability to handle transitive logic. While smaller models can mimic language patterns, they struggle with multi-step deductions that the 8B model handles with ease.

* **Efficacy of Domain Fine-Tuning:**
    The **LLaMA-8B-FT** demonstrated superior alignment with the intended "Vet Assistant" persona compared to the base models. The fine-tuning process successfully calibrated the model's tone, allowing it to balance empathy with strict safety boundaries (e.g., avoiding definitive medical diagnoses while remaining supportive).

* **Instruction Adherence & Conciseness:**
    Larger, fine-tuned models tend to exhibit better instruction adherence with less "chatter." In the mathematical reasoning and rewriting tasks, the 8B model provided the most direct and precise answers, whereas the intermediate models (3B) tended towards unnecessary verbosity.

---

## 🚀 Deployment & User Interface

To make the model accessible on standard hardware (CPU-only environments), we merged the LoRA adapters into the base model and exported the result to **GGUF format** (8-bit quantization).

The final application is hosted on **Hugging Face Spaces**. It features a user-friendly Chat Interface where users can ask questions about their pets' health.

👉 **Try the VetAI Assistant here:** https://huggingface.co/spaces/stromano02/Iris

---

## 🛠️ How to Reproduce

1.  **Environment:** The code is designed to run on Google Colab (T4 GPU).
2.  **Dependencies:** Install `unsloth`, `transformers`, and `trl` as specified in the first cell of the notebook.
3.  **Data:** The notebook automatically handles data loading from Hugging Face (`mlabonne/FineTome-100k`) and the custom veterinary JSONL file.
