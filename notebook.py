# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "accelerate==1.12.0",
#     "bert-score==0.3.13",
#     "datasets==4.4.1",
#     "evaluate==0.4.6",
#     "marimo",
#     "matplotlib==3.10.8",
#     "numpy==2.2.6",
#     "pandas==2.3.3",
#     "scikit-learn==1.8.0",
#     "tensorflow==2.20.0",
#     "torch==2.9.1",
#     "tqdm==4.67.1",
#     "transformers[torch]==4.57.3",
# ]
# ///

import marimo

__generated_with = "0.18.4"
app = marimo.App(
    width="medium",
    css_file="/usr/local/_marimo/custom.css",
    auto_download=["html"],
)


@app.cell
def _():
    import marimo as mo
    import torch
    from transformers import (
        T5ForConditionalGeneration,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        TrainerCallback,
        AutoModelForSeq2SeqLM
    )
    from datasets import Dataset
    import matplotlib.pyplot as plt
    from pathlib import Path
    import warnings

    mrmm_model="mrm8488/t5-small-finetuned-text-simplification"

    warnings.filterwarnings("ignore")
    return (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        Dataset,
        T5ForConditionalGeneration,
        Trainer,
        TrainerCallback,
        TrainingArguments,
        mo,
        mrmm_model,
        plt,
        torch,
    )


@app.cell
def _(mo):
    mo.md("""
    # T5 Fine-tuning for Text Simplification

    This notebook demonstrates fine-tuning T5-base on the ASSET text simplification dataset using HuggingFace Trainer.
    """)
    return


@app.cell
def _(mo):
    asset_folder_path = mo.ui.file_browser(
        selection_mode="directory",
        multiple=False,
        label="Asset folder",
        initial_path="./data"
    )
    asset_folder_path
    return (asset_folder_path,)


@app.cell
def _(Dataset):
    def load_asset_data(asset_folder_path, split="valid"):
        """Load ASSET dataset from folder path and return HuggingFace Dataset"""
        import os

        src_sentences = []
        tgt_sentences = []


        for file_name in os.listdir(asset_folder_path):
            if file_name.endswith(f".{split}.orig"):
                base_name = file_name[: -(len(split) + 6)]  # Remove .split.orig extension
                orig_path = os.path.join(asset_folder_path, file_name)

                with open(orig_path, "r", encoding="utf-8") as f:
                    orig_sentences = [line.strip() for line in f if line.strip()]

                simp_files = [
                    os.path.join(asset_folder_path, simp_file_name)
                    for simp_file_name in os.listdir(asset_folder_path)
                    if simp_file_name.startswith(base_name) and f".{split}.simp." in simp_file_name
                ]

                simp_sentences_list = []
                for simp_file in simp_files:
                    with open(simp_file, "r", encoding="utf-8") as f:
                        simp_sentences = [line.strip() for line in f if line.strip()]
                        simp_sentences_list.append(simp_sentences)


                for i, orig_sentence in enumerate(orig_sentences):
                    for simp_sentences in simp_sentences_list:
                        if i < len(simp_sentences):
                            src_sentences.append("simplify: " + orig_sentence)
                            tgt_sentences.append(simp_sentences[i])

        if not src_sentences or not tgt_sentences:
            print("Warning: No data loaded. Check the folder structure and file naming conventions.")

        # Create HuggingFace Dataset
        data_dict = {
            "source": src_sentences,
            "target": tgt_sentences,
        }

        return Dataset.from_dict(data_dict)
    return (load_asset_data,)


@app.cell
def _(AutoTokenizer, asset_folder_path, load_asset_data, mo, mrmm_model):
    tokenizer = AutoTokenizer.from_pretrained(mrmm_model, legacy=False)

    train_dataset_t5 = load_asset_data(asset_folder_path.path(index=0), split="valid")
    test_dataset_t5 = load_asset_data(asset_folder_path.path(index=0), split="test")

    mo.md(f"""
    ### Dataset Loaded!

    - **Training samples**: {len(train_dataset_t5):,}
    - **Test samples**: {len(test_dataset_t5):,}
    - **Model**: t5-small-finetuned-text-simplification
    - **Tokenizer vocabulary size**: {len(tokenizer):,}

    **Example pair:**
    - Complex: {train_dataset_t5[0]['source']}
    - Simple: {train_dataset_t5[0]['target']}
    """)
    return test_dataset_t5, tokenizer, train_dataset_t5


@app.cell
def _(tokenizer):
    def preprocess_function(examples, max_length=128):
        """
        Tokenizes the source and target text and prepares the specific 
        inputs required for a T5 Encoder-Decoder architecture.
        """

        # 1. ENCODER INPUTS: Tokenize the complex source text.
        # These are the inputs the 'Encoder' sees to understand the original context.
        model_inputs = tokenizer(
            examples["source"],
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )

        # 2. TARGET ANNOTATION: Tokenize the simplified target text.
        # We use this to create both the 'labels' (the answers) and the 'decoder inputs'.
        labels = tokenizer(
            examples["target"],
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )

        # 3. TEACHER FORCING: Manually create decoder_input_ids.
        # T5 requires the decoder to see the target tokens shifted by one position.
        # This prevents the model from 'cheating' by seeing the word it's supposed to predict.
        decoder_input_ids = []
        for label_ids in labels["input_ids"]:
            # T5 uses the pad_token_id (0) as the 'Start-of-Sequence' signal.
            # We prepend '0' and remove the last token to keep the length at 128.
            # Shifted: [Start, Token1, Token2, ...] instead of [Token1, Token2, Token3, ...]
            shifted = [tokenizer.pad_token_id] + label_ids[:-1]
            decoder_input_ids.append(shifted)

        model_inputs["decoder_input_ids"] = decoder_input_ids

        # 4. LOSS CALCULATION: Prepare the labels and mask padding.
        # We replace all padding token IDs (0) with -100.
        # PyTorch loss functions ignore -100, so the model is only penalized for 
        # mistakes on actual words, not on the empty padding at the end.
        model_inputs["labels"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label_ids]
            for label_ids in labels["input_ids"]
        ]

        return model_inputs
    return (preprocess_function,)


@app.cell
def _(preprocess_function, test_dataset_t5, train_dataset_t5):
    tokenized_train = train_dataset_t5.map(
        preprocess_function, batched=True, remove_columns=["source", "target"]
    )
    tokenized_test = test_dataset_t5.map(
        preprocess_function, batched=True, remove_columns=["source", "target"]
    )

    print(f"Tokenized training samples: {len(tokenized_train):,}")
    print(f"Tokenized test samples: {len(tokenized_test):,}")
    return tokenized_test, tokenized_train


@app.cell
def _(mo):
    mo.md("""
    ## Training Hyperparameters

    Configure the training parameters:
    """)
    return


@app.cell
def _(mo):
    learning_rate_t5 = mo.ui.slider(
        1e-5, 10e-5, value=5e-5, step=1e-6, label="Learning Rate"
    )
    n_epochs_t5 = mo.ui.slider(1, 10, value=5, step=1, label="Number of Epochs")
    batch_size_t5 = mo.ui.slider(8, 64, value=16, step=8, label="Batch Size")
    warmup_ratio_t5 = mo.ui.slider(0.0, 0.2, value=0.1, step=0.01, label="Warmup Ratio")

    mo.hstack([
        mo.vstack([learning_rate_t5, n_epochs_t5]),
        mo.vstack([batch_size_t5, warmup_ratio_t5])
    ])
    return batch_size_t5, learning_rate_t5, n_epochs_t5, warmup_ratio_t5


@app.cell
def _(mo):
    train_button = mo.ui.run_button(label="Start Training")
    train_button
    return (train_button,)


@app.cell
def _(
    T5ForConditionalGeneration,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    batch_size_t5,
    learning_rate_t5,
    mo,
    mrmm_model,
    n_epochs_t5,
    plt,
    tokenized_test,
    tokenized_train,
    torch,
    train_button,
    warmup_ratio_t5,
):
    # Determine device
    if torch.cuda.is_available():
        device_t5 = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device_t5 = torch.device("mps")
    else:
        device_t5 = torch.device("cpu")

    # Load model
    model_t5 = T5ForConditionalGeneration.from_pretrained(mrmm_model)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=n_epochs_t5.value,
        per_device_train_batch_size=batch_size_t5.value,    
        per_device_eval_batch_size=batch_size_t5.value,
        warmup_ratio=warmup_ratio_t5.value,
        learning_rate=learning_rate_t5.value,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=300,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=300,
        save_steps=300,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        optim="adamw_torch_fused",
        label_smoothing_factor=0.1,
        max_grad_norm=0.5,
        use_mps_device=(device_t5.type == "mps"),
    )

    # Custom callback to track losses for plotting
    class LossCallback(TrainerCallback):
        def __init__(self):
            self.train_losses = []
            self.eval_losses = []
            self.train_steps = []
            self.eval_steps = []

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is not None:
                if "loss" in logs:
                    self.train_losses.append(logs["loss"])
                    self.train_steps.append(state.global_step)
                if "eval_loss" in logs:
                    self.eval_losses.append(logs["eval_loss"])
                    self.eval_steps.append(state.global_step)

    loss_callback = LossCallback()

    # Initialize trainer
    trainer = Trainer(
        model=model_t5,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        callbacks=[loss_callback],
    )

    fig_t5, ax_t5 = plt.subplots(figsize=(10, 6))

    if train_button.value:
        print(f"Training on device: {device_t5}")
        print(f"Model parameters: {sum(p.numel() for p in model_t5.parameters()):,}\n")
        print("Training Configuration:")
        print("=" * 40)
        print(f"Learning Rate:    {learning_rate_t5.value}")
        print(f"Number of Epochs: {n_epochs_t5.value}")
        print(f"Batch Size:       {batch_size_t5.value}")
        print(f"Warmup Ratio:     {warmup_ratio_t5.value}")
        print("=" * 40 + "\n")

        # Train the model
        train_result = trainer.train()

        # Get final evaluation
        eval_result = trainer.evaluate()

        # Plot losses
        ax_t5.clear()
        if loss_callback.train_losses:
            ax_t5.plot(loss_callback.train_steps, loss_callback.train_losses, 'b-', label='Train Loss', linewidth=2, alpha=0.7)

        if loss_callback.eval_losses:
            ax_t5.plot(loss_callback.eval_steps, loss_callback.eval_losses, 'r-', label='Eval Loss', linewidth=2, marker='o', markersize=6)

        ax_t5.set_xlabel("Training Steps", fontsize=12)
        ax_t5.set_ylabel("Loss", fontsize=12)
        ax_t5.set_title("T5 Training Progress", fontsize=14, fontweight="bold")
        ax_t5.legend(fontsize=11)
        ax_t5.grid(True, alpha=0.3)
        plt.tight_layout()

        mo.md(f"""
        ### Training Complete!

        **Final Training Loss**: {train_result.training_loss:.4f}

        **Final Evaluation Loss**: {eval_result['eval_loss']:.4f}

        **Training Time**: {train_result.metrics['train_runtime']:.2f} seconds ({train_result.metrics['train_runtime']/60:.1f} minutes)
        """)

        mo.output.append(ax_t5)
    return (device_t5,)


@app.cell
def _(mo):
    mo.md("""
    ## Interactive Simplification

    Test the fine-tuned model and compare with pre-trained models:
    """)
    return


@app.cell
def _(mo):
    test_input = mo.ui.text_area(
        label="Enter a complex sentence to simplify:",
        value="Although the original manuscript was widely believed to have been lost during the chaotic events of the library fire in 1845, a partially preserved copy was unexpectedly discovered within the dusty archives of a remote monastery nearly a century later",
        rows=3,
    )
    test_input
    return (test_input,)


@app.cell
def _(torch):
    def simplify_with_t5(text, model, tokenizer, device, max_length=64):
        """Simplify text using T5 model"""
        model.eval()

        input_text = "simplify: " + text

        input_ids = tokenizer(
            input_text,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)

        # Most of the arguments in the following definition are based on the model not shortening the simplified sentences.
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_length=max_length,
                min_length=10,         # Prevent it from being too short
                num_beams=8, # More beams in hope of finding a better short path 
                length_penalty=-1,   # Need an aggressive negative value, because output didn't change at all
                repetition_penalty=2.5, # Penalizing for too much repetition
                no_repeat_ngram_size=2, # Block even 2-word repetitions
                early_stopping=True,
                do_sample=True,        # Allow for more simpler word choices
                top_p=0.90             # Only consider the most likely 90% of words
            )

        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    return (simplify_with_t5,)


@app.cell
def _(
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    T5ForConditionalGeneration,
    device_t5,
    mo,
    simplify_with_t5,
    test_input,
    tokenizer,
):
    # 1. Define the path to your best checkpoint
    checkpoint_path = "./results/checkpoint-3000"

    # 2. Load the specific fine-tuned weights 
    our_t5_model = T5ForConditionalGeneration.from_pretrained(checkpoint_path).to(device_t5)

    # 3. Use the loaded model in your generation step
    input_text_test = test_input.value.strip()

    simplified_ours = simplify_with_t5(
        input_text_test, our_t5_model, tokenizer, device_t5
    )

    input_text_test = test_input.value.strip()

    # Load pre-trained models
    tokenizer_eilamc = AutoTokenizer.from_pretrained(
        "eilamc14/t5-base-text-simplification", legacy=False
    )
    model_eilamc = AutoModelForSeq2SeqLM.from_pretrained(
        "eilamc14/t5-base-text-simplification"
    ).to(device_t5)

    # tokenizer_mrm = T5Tokenizer.from_pretrained(
    #     "mrm8488/t5-small-finetuned-text-simplification", legacy=False
    # )
    # model_mrm = T5ForConditionalGeneration.from_pretrained(
    #     "mrm8488/t5-small-finetuned-text-simplification"
    # ).to(device_t5)

    # Simplify with pre-trained models
    simplified_eilamc = simplify_with_t5(
        input_text_test, model_eilamc, tokenizer_eilamc, device_t5
    )
    # simplified_mrm = simplify_with_t5(
    #     input_text_test, model_mrm, tokenizer_mrm, device_t5
    # )

    mo.md(f"""
    ### Model Comparison

    **Original (Complex):**
    > {input_text_test}

    ---

    | Model | Simplified Output |
    |-------|-------------------|
    | **Our Fine-tuned T5** | {simplified_ours} |
    | **eilamc14/t5-base-text-simplification** | {simplified_eilamc} |
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Evaluation with Tsar Trialdata
    """)
    return


@app.cell
def _(AutoTokenizer, T5ForConditionalGeneration, torch):
    def _():
        import os, json, random
        import numpy as np
        import pandas as pd
        from sklearn.metrics import f1_score, root_mean_squared_error
        from transformers import pipeline
        import evaluate
        from tqdm import tqdm

        # ---------------- Config ----------------
        GOLD_FILE = "tsar2025_trialdata.jsonl"   # gold file next to this script
        SEED = 42                           # for reproducibility
        BATCH_SIZE = 32                     # adjust for your GPU
        CHECKPOINT_PATH = "./results/checkpoint-3000" # Your local fine-tuned model
        BASE_MODEL_NAME = "mrm8488/t5-small-finetuned-text-simplification"

        # ---------------- Seed ------------------
        random.seed(SEED)
        np.random.seed(SEED)
        try:
            torch.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass

        device = 0 if torch.cuda.is_available() else -1

        # Loading the Model
        print(f"Loading fine-tuned model from {CHECKPOINT_PATH}...")
        tokenizer_eval = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, legacy=False)
        model = T5ForConditionalGeneration.from_pretrained(CHECKPOINT_PATH).to("cuda" if device == 0 else "cpu")

        def simplify_with_t5_live(text):
            """Function to generate simplification using the loaded model"""
            model.eval()
            input_text = "simplify: " + text
            inputs = tokenizer_eval(input_text, max_length=64, truncation=True, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_length=64, 
                    min_length=10,
                    num_beams=8,
                    length_penalty=-3,
                    repetition_penalty=2.5,
                    no_repeat_ngram_size=2,
                    early_stopping=True,
                    do_sample=True,
                    top_p=0.90
                )
            return tokenizer_eval.decode(outputs[0], skip_special_tokens=True)

        # ---------------- IO --------------------
        def read_jsonl(path: str):
            with open(path, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f]

        def read_gold(path: str):
            data = read_jsonl(path)
            if not data:
                raise ValueError(f"Gold file is empty: {path}")
            try:
                original = [e["original"] for e in data]
                reference = [e["reference"] for e in data]
                target   = [e["target_cefr"] for e in data]
                text_ids = [e["text_id"] for e in data]
            except KeyError as ke:
                raise KeyError(f"Gold file missing key {ke}.")
            return original, reference, target, text_ids

        # Align system outputs
        def align_intersection(hyps, sys_ids, gold_ids, gold_orig, gold_ref, gold_tgt):
            gid2idx = {g:i for i,g in enumerate(gold_ids)}
            pairs = [(gid2idx[sid], hyp) for hyp, sid in zip(hyps, sys_ids) if sid in gid2idx]
            if not pairs:
                return None
            pairs.sort(key=lambda x: x[0])
            sel_idx = [i for i,_ in pairs]
            aligned_hyps = [h for _,h in pairs]
            aligned_orig = [gold_orig[i] for i in sel_idx]
            aligned_ref  = [gold_ref[i]  for i in sel_idx]
            aligned_tgt  = [gold_tgt[i]  for i in sel_idx]
            return {
                "hyps": aligned_hyps,
                "orig": aligned_orig,
                "ref":  aligned_ref,
                "tgt":  aligned_tgt,
                "coverage_n": len(sel_idx),
                "coverage_pct": round(100.0 * len(sel_idx) / len(gold_ids), 2)
            }

        # ------------- Models/Metrics -----------
        cefr_labeler1 = pipeline("text-classification", model="AbdullahBarayan/ModernBERT-base-doc_en-Cefr", device=device)
        cefr_labeler2 = pipeline("text-classification", model="AbdullahBarayan/ModernBERT-base-doc_sent_en-Cefr", device=device)
        cefr_labeler3 = pipeline("text-classification", model="AbdullahBarayan/ModernBERT-base-reference_AllLang2-Cefr2", device=device)

        meaning_bert = evaluate.load("davebulaval/meaningbert")
        bertscore    = evaluate.load("bertscore")

        CEFR = ["A1","A2","B1","B2","C1","C2"]
        L2I  = {l:i for i,l in enumerate(CEFR)}

        def cefr_labels(hyps, models, batch_size=BATCH_SIZE):
            p1 = models[0](hyps, batch_size=batch_size, truncation=True)
            p2 = models[1](hyps, batch_size=batch_size, truncation=True)
            p3 = models[2](hyps, batch_size=batch_size, truncation=True)
            def top1(x):
                if isinstance(x, dict): return x
                if isinstance(x, list) and x: return max(x, key=lambda d: d["score"])
            outs = []
            for d1, d2, d3 in zip(p1, p2, p3):
                best = max((top1(d1), top1(d2), top1(d3)), key=lambda d: d["score"])
                outs.append(best["label"].strip().upper())
            return outs

        def score_cefr(hyps, ref_lvls, models):
            gold  = [str(l).strip().upper() for l in ref_lvls]
            preds = [str(l).strip().upper() for l in cefr_labels(hyps, models)]
            f1 = f1_score(gold, preds, average="weighted")
            t  = np.array([L2I[l] for l in gold])
            p  = np.array([L2I[l] for l in preds])
            adj  = (np.abs(t - p) <= 1).mean()
            rmse = root_mean_squared_error(t, p)
            return {"weighted_f1": round(float(f1),4), "adj_accuracy": round(float(adj),4), "rmse": round(float(rmse),4)}

        def score_meaningbert(hyps, refs):
            res = meaning_bert.compute(predictions=hyps, references=refs)
            return round(float(np.mean(res["scores"])) / 100.0, 4)

        def score_bertscore(hyps, refs, scoretype="f1"):
            res = bertscore.compute(references=refs, predictions=hyps, lang="en")
            return round(float(np.mean(res[scoretype])), 4)

        # ------------- Main ---------------------
        if not os.path.isfile(GOLD_FILE):
            raise FileNotFoundError(f"Gold file not found: {GOLD_FILE}")

        gold_orig, gold_ref, gold_tgt, gold_ids = read_gold(GOLD_FILE)

        print(f"Generating simplifications for {len(gold_orig)} sentences...")
        # Actual generation step
        hyps = [simplify_with_t5_live(text) for text in tqdm(gold_orig)]

        print("\n=== Random Generation Samples ===")
        sample_indices = random.sample(range(len(gold_orig)), min(15, len(gold_orig)))
        for idx in sample_indices:
            print(f"\n[ID: {gold_ids[idx]}]")
            print(f"Original:  {gold_orig[idx]}")
            print(f"Simple:    {hyps[idx]}")
            print("-" * 50)

        results = []
        aligned = align_intersection(hyps, gold_ids, gold_ids, gold_orig, gold_ref, gold_tgt)

        if aligned:
            print("Computing metrics...")
            hyps_i, orig_i, ref_i, tgt_i = aligned["hyps"], aligned["orig"], aligned["ref"], aligned["tgt"]

            cefr = score_cefr(hyps_i, tgt_i, [cefr_labeler1, cefr_labeler2, cefr_labeler3])
            mb_o = score_meaningbert(hyps_i, orig_i)
            bs_o = score_bertscore(hyps_i, orig_i, "f1")
            mb_r = score_meaningbert(hyps_i, ref_i)
            bs_r = score_bertscore(hyps_i, ref_i, "f1")

            results.append({
                "modelname": "Our-Fine-Tuned-T5-Checkpoint",
                "weighted_f1": cefr["weighted_f1"],
                "adj_accuracy": cefr["adj_accuracy"],
                "rmse": cefr["rmse"],
                "meaningbert-orig": mb_o,
                "bertscore-orig": bs_o,
                "meaningbert-ref": mb_r,
                "bertscore-ref": bs_r
            })

        df = pd.DataFrame(results)
        print("\n=== Validation Results ===")
        print(df.to_string(index=False))
        output_file = "validation_results.csv"
        df.to_csv(output_file, index=False, encoding='utf-8')
        return print(f"\nSaved: {output_file}")


    _()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
