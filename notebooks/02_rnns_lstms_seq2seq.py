import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    import warnings

    warnings.filterwarnings("ignore")
    return mo, nn, plt, torch


@app.cell
def _(nn, torch):
    class RNN(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(RNN, self).__init__()
            self.hidden_size = hidden_size

            # Input to hidden weights
            self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
            # Hidden to output weights
            self.h2o = nn.Linear(hidden_size, output_size)

        def forward(self, input, hidden):
            """
            One step of RNN
            input: shape (batch_size, input_size)
            hidden: shape (batch_size, hidden_size)
            """
            # Concatenate input and hidden state
            # This is mathematically equivalent to having two matrices and adding their results
            combined = torch.cat((input, hidden), 1)
            # Compute new hidden state
            hidden = torch.tanh(self.i2h(combined))
            # Compute output
            output = self.h2o(hidden)
            return output, hidden

        def init_hidden(self, batch_size):
            """Initialize hidden state with zeros"""
            return torch.zeros(batch_size, self.hidden_size)


    # Example: Create a simple RNN
    rnn_demo = RNN(input_size=50, hidden_size=128, output_size=10)
    print(rnn_demo)
    return


@app.cell
def _(nn, torch):
    # Seq2Seq Encoder
    class Encoder(nn.Module):
        def __init__(
            self, input_size, embedding_dim, hidden_dim, n_layers: int = 1, dropout=0.5
        ):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.n_layers: int = n_layers

            self.embedding = nn.Embedding(input_size, embedding_dim)
            self.lstm = nn.LSTM(
                embedding_dim,
                hidden_dim,
                n_layers,
                dropout=dropout if n_layers > 1 else 0,
                batch_first=True,
            )
            self.dropout = nn.Dropout(dropout)

        def forward(self, src):
            # src shape: (batch_size, src_len)
            embedded = self.dropout(self.embedding(src))
            # embedded shape: (batch_size, src_len, embedding_dim)

            _, (hidden, cell) = self.lstm(embedded)
            # hidden shape: (n_layers, batch_size, hidden_dim)
            # cell shape: (n_layers, batch_size, hidden_dim)

            return hidden, cell


    # Seq2Seq Decoder
    class Decoder(nn.Module):
        def __init__(
            self, output_size, embedding_dim, hidden_dim, n_layers: int=1, dropout=0.5
        ):
            super().__init__()
            self.output_size = output_size
            self.hidden_dim = hidden_dim
            self.n_layers: int = n_layers

            self.embedding = nn.Embedding(output_size, embedding_dim)
            self.lstm = nn.LSTM(
                embedding_dim,
                hidden_dim,
                n_layers,
                dropout=dropout if n_layers > 1 else 0,
                batch_first=True,
            )
            self.fc_out = nn.Linear(hidden_dim, output_size)
            self.dropout = nn.Dropout(dropout)

        def forward(self, input, hidden, cell):
            # input shape: (batch_size, 1)  ← One token at a time!
            input = input.unsqueeze(1)
            embedded = self.dropout(self.embedding(input))
            # embedded shape: (batch_size, 1, embedding_dim)

            output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
            # output shape: (batch_size, 1, hidden_dim)

            prediction = self.fc_out(output.squeeze(1))
            # prediction shape: (batch_size, output_size)

            return prediction, hidden, cell


    # Complete Seq2Seq Model
    class Seq2Seq(nn.Module):
        def __init__(self, encoder, decoder, device):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder
            self.device = device

        def forward(self, src, trg, teacher_forcing_ratio=0.5):
            # src shape: (batch_size, src_len)
            # trg shape: (batch_size, trg_len)

            batch_size = src.shape[0]
            trg_len = trg.shape[1]
            trg_vocab_size = self.decoder.output_size

            # Store decoder outputs
            outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(
                self.device
            )

            # Encode the source
            hidden, cell = self.encoder(src)

            # First input to decoder is <SOS> token
            input = trg[:, 0]

            for t in range(1, trg_len):
                # Decode one step
                output, hidden, cell = self.decoder(input, hidden, cell)
                outputs[:, t] = output

                # Teacher forcing: use actual next token vs predicted
                teacher_force = torch.rand(1).item() < teacher_forcing_ratio
                top1 = output.argmax(1)
                input = trg[:, t] if teacher_force else top1

            return outputs
    return Decoder, Encoder, Seq2Seq


@app.cell
def _(nn, torch):
    # Simple LSTM Example: Sentiment Classification
    class SimpleLSTMClassifier(nn.Module):
        def __init__(
            self,
            vocab_size,
            embedding_dim,
            hidden_dim,
            output_dim,
            n_layers=1,
            dropout=0.5,
        ):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.lstm = nn.LSTM(
                embedding_dim,
                hidden_dim,
                n_layers,
                dropout=dropout if n_layers > 1 else 0,
                batch_first=True,
            )
            self.fc = nn.Linear(hidden_dim, output_dim)
            self.dropout = nn.Dropout(dropout)

        def forward(self, text):
            # text shape: (batch_size, seq_len)
            embedded = self.dropout(self.embedding(text))
            # embedded shape: (batch_size, seq_len, embedding_dim)

            # Pass through LSTM
            _, (hidden, _) = self.lstm(embedded)
            # output shape: (batch_size, seq_len, hidden_dim)
            # hidden shape: (n_layers, batch_size, hidden_dim)
            # cell shape: (n_layers, batch_size, hidden_dim)

            # Use the final hidden state for classification
            hidden = self.dropout(hidden[-1])  # Take last layer
            # hidden shape: (batch_size, hidden_dim)

            return self.fc(hidden)


    # Create a simple LSTM classifier
    VOCAB_SIZE = 10000
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 256
    OUTPUT_DIM = 2  # Binary classification (e.g., positive/negative)
    N_LAYERS = 2
    DROPOUT = 0.5

    lstm_classifier = SimpleLSTMClassifier(
        VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, DROPOUT
    )

    # Example forward pass
    batch_size = 4
    seq_len = 10
    example_input = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len))
    example_output = lstm_classifier(example_input)

    print(f"LSTM Classifier Created:")
    print(f"  Vocabulary size: {VOCAB_SIZE:,}")
    print(f"  Embedding dimension: {EMBEDDING_DIM}")
    print(f"  Hidden dimension: {HIDDEN_DIM}")
    print(f"  Number of layers: {N_LAYERS}")
    print(f"  Output dimension: {OUTPUT_DIM} (binary classification)")
    print(f"\nExample:")
    print(f"  Input shape: {example_input.shape} (batch_size, seq_len)")
    print(f"  Output shape: {example_output.shape} (batch_size, output_dim)")
    print(
        f"  Total parameters: {sum(p.numel() for p in lstm_classifier.parameters()):,}"
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Training on ASSET Dataset

    Now let's train a real Seq2Seq model on the ASSET text simplification dataset!

    ### Dataset Preparation

    First, we need to:
    1. Load the parallel sentences (complex → simple)
    2. Build vocabularies for source and target
    3. Convert sentences to token indices
    4. Create data loaders
    """)
    return


@app.cell
def _(torch):
    from collections import Counter
    from pathlib import Path

    # Special tokens
    PAD_TOKEN = "<PAD>"
    SOS_TOKEN = "<SOS>"
    EOS_TOKEN = "<EOS>"
    UNK_TOKEN = "<UNK>"


    class Vocabulary:
        def __init__(self, min_freq=2):
            self.min_freq = min_freq
            self.word2idx = {
                PAD_TOKEN: 0,
                SOS_TOKEN: 1,
                EOS_TOKEN: 2,
                UNK_TOKEN: 3,
            }
            self.idx2word = {
                0: PAD_TOKEN,
                1: SOS_TOKEN,
                2: EOS_TOKEN,
                3: UNK_TOKEN,
            }
            self.word_counts = Counter()

        def build_vocab(self, sentences):
            """Build vocabulary from list of sentences"""
            for sentence in sentences:
                self.word_counts.update(sentence.lower().split())

            # Add words that appear at least min_freq times
            idx = len(self.word2idx)
            for word, count in self.word_counts.items():
                if count >= self.min_freq and word not in self.word2idx:
                    self.word2idx[word] = idx
                    self.idx2word[idx] = word
                    idx += 1

        def encode(self, sentence, max_len=None):
            """Convert sentence to indices"""
            tokens = [SOS_TOKEN] + sentence.lower().split() + [EOS_TOKEN]
            indices = [
                self.word2idx.get(token, self.word2idx[UNK_TOKEN])
                for token in tokens
            ]

            if max_len:
                indices = indices[:max_len]

            return indices

        def decode(self, indices):
            """Convert indices back to sentence"""
            words = []
            for idx in indices:
                word = self.idx2word.get(idx, UNK_TOKEN)
                if word == EOS_TOKEN:
                    break
                if word not in [PAD_TOKEN, SOS_TOKEN]:
                    words.append(word)
            return " ".join(words)

        def __len__(self):
            return len(self.word2idx)


    def load_data(
        data_path_str, split="valid", simplification_idx=0, max_samples=None
    ):
        """Load ASSET dataset"""
        src_file = Path(data_path_str) / f"asset.{split}.orig"
        tgt_file = Path(data_path_str) / f"asset.{split}.simp.{simplification_idx}"

        with open(src_file, "r", encoding="utf-8") as f:
            src_sentences = [line.strip() for line in f if line.strip()]

        with open(tgt_file, "r", encoding="utf-8") as f:
            tgt_sentences = [line.strip() for line in f if line.strip()]

        # Ensure matching pairs
        data = list(zip(src_sentences, tgt_sentences))

        if max_samples:
            data = data[:max_samples]

        return data


    class SimplificationDataset(torch.utils.data.Dataset):
        def __init__(self, data, src_vocab, tgt_vocab, max_len=50):
            self.data = data
            self.src_vocab = src_vocab
            self.tgt_vocab = tgt_vocab
            self.max_len = max_len

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            src_sentence, tgt_sentence = self.data[idx]
            src_indices = self.src_vocab.encode(src_sentence, self.max_len)
            tgt_indices = self.tgt_vocab.encode(tgt_sentence, self.max_len)
            return torch.LongTensor(src_indices), torch.LongTensor(tgt_indices)


    def collate_fn(batch):
        """Pad sequences to same length in batch"""
        src_batch, tgt_batch = zip(*batch)

        # Pad source sequences
        src_lens = [len(s) for s in src_batch]
        max_src_len = max(src_lens)
        src_padded = torch.zeros(len(src_batch), max_src_len, dtype=torch.long)
        for i, s in enumerate(src_batch):
            src_padded[i, : len(s)] = s

        # Pad target sequences
        tgt_lens = [len(t) for t in tgt_batch]
        max_tgt_len = max(tgt_lens)
        tgt_padded = torch.zeros(len(tgt_batch), max_tgt_len, dtype=torch.long)
        for i, t in enumerate(tgt_batch):
            tgt_padded[i, : len(t)] = t

        return src_padded, tgt_padded


    print("Data loading utilities created!")
    return (
        PAD_TOKEN,
        SOS_TOKEN,
        SimplificationDataset,
        Vocabulary,
        collate_fn,
        load_data,
    )


@app.cell
def _(mo):
    data_path_str = "./data/asset"
    mo.md(f"**Data Path for ASSET Dataset:** `{data_path_str}`")
    return (data_path_str,)

@app.cell
def _(data_path_str): 
    print(data_path_str) 
    return


@app.cell
def _(
    SimplificationDataset,
    Vocabulary,
    collate_fn,
    data_path_str,
    load_data,
    mo,
    torch,
):
    train_data = load_data(
        data_path_str, split="valid", simplification_idx=0, max_samples=None
    )

    test_data = load_data(data_path_str, split="test", simplification_idx=0)

    # Build vocabularies from training data only
    src_vocab = Vocabulary(min_freq=2)
    tgt_vocab = Vocabulary(min_freq=2)

    src_vocab.build_vocab([src for src, _ in train_data])
    tgt_vocab.build_vocab([tgt for _, tgt in train_data])

    # Create datasets
    # Test data also uses training vocabularies 
    # This causes OUT-OF-VOCAB issues if new words appear in test set (see later)
    train_dataset = SimplificationDataset(train_data, src_vocab, tgt_vocab)
    test_dataset = SimplificationDataset(test_data, src_vocab, tgt_vocab)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn
    )

    mo.md(f"""
    ### Dataset Loaded!

    - **Training samples** (from valid split): {len(train_data):,}
    - **Test samples**: {len(test_data):,}
    - **Source vocabulary size**: {len(src_vocab):,}
    - **Target vocabulary size**: {len(tgt_vocab):,}

    **Example pair:**
    - Complex: {train_data[0][0]}
    - Simple: {train_data[0][1]}
    """)
    return (
        src_vocab,
        test_data,
        test_loader,
        tgt_vocab,
        train_data,
        train_loader,
    )


@app.cell
def _(mo, test_data, train_data):
    train_words = set()
    for src, tgt in train_data:
        train_words.update(src.lower().split())
        train_words.update(tgt.lower().split())

    # Extract all words from test data
    test_words = set()
    for src, tgt in test_data:
        test_words.update(src.lower().split())
        test_words.update(tgt.lower().split())

    # Calculate overlap
    overlap_words = train_words.intersection(test_words)
    train_only = train_words - test_words
    test_only = test_words - train_words

    # Calculate percentages
    overlap_pct_train = (
        (len(overlap_words) / len(train_words)) * 100 if train_words else 0
    )
    overlap_pct_test = (
        (len(overlap_words) / len(test_words)) * 100 if test_words else 0
    )
    mo.md(f"""
    ### Vocabulary Overlap Analysis

    **Training Set:**
    - Unique words: {len(train_words):,}
    - Words only in training: {len(train_only):,}

    **Test Set:**
    - Unique words: {len(test_words):,}
    - Words only in test: {len(test_only):,}

    **Overlap:**
    - Shared words: {len(overlap_words):,}
    - Overlap (% of train vocab): {overlap_pct_train:.1f}%
    - Overlap (% of test vocab): {overlap_pct_test:.1f}%

    **Out-of-vocabulary (OOV) rate:**
    - Test words not seen in training: {len(test_only):,} ({(len(test_only) / len(test_words) * 100) if test_words else 0:.1f}%)
        """)
    return


@app.cell
def _(mo):
    # Training hyperparameters as UI inputs
    enc_emb_dim = mo.ui.slider(
        64, 256, value=128, step=64, label="Encoder Embedding Dim"
    )
    dec_emb_dim = mo.ui.slider(
        64, 256, value=128, step=64, label="Decoder Embedding Dim"
    )
    hid_dim = mo.ui.slider(64, 512, value=128, step=64, label="Hidden Dimension")
    n_layers = mo.ui.slider(1, 3, value=1, step=1, label="Number of Layers")
    enc_dropout = mo.ui.slider(
        0.0, 0.7, value=0.5, step=0.1, label="Encoder Dropout"
    )
    dec_dropout = mo.ui.slider(
        0.0, 0.7, value=0.5, step=0.1, label="Decoder Dropout"
    )
    learning_rate = mo.ui.slider(
        0.0001, 0.001, value=0.0005, step=0.0001, label="Learning Rate"
    )
    n_epochs = mo.ui.slider(50, 500, value=100, step=50, label="Number of Epochs")

    mo.hstack(
        [
            mo.vstack([enc_emb_dim, dec_emb_dim, hid_dim, n_layers]),
            mo.vstack([enc_dropout, dec_dropout, learning_rate, n_epochs]),
        ]
    )
    return (
        dec_dropout,
        dec_emb_dim,
        enc_dropout,
        enc_emb_dim,
        hid_dim,
        learning_rate,
        n_epochs,
        n_layers,
    )


@app.cell
def _(mo):
    button = mo.ui.run_button()
    button
    return (button,)


@app.cell
def _(
    Decoder,
    Encoder,
    PAD_TOKEN,
    Seq2Seq,
    button,
    dec_dropout,
    dec_emb_dim,
    enc_dropout,
    enc_emb_dim,
    hid_dim,
    learning_rate,
    mo,
    n_epochs,
    n_layers,
    plt,
    src_vocab,
    test_loader,
    tgt_vocab,
    torch,
    train_loader,
):
    import time

    # Model hyperparameters from UI inputs
    INPUT_DIM_TRAIN = len(src_vocab)
    OUTPUT_DIM_TRAIN = len(tgt_vocab)
    ENC_EMB_DIM_TRAIN = enc_emb_dim.value
    DEC_EMB_DIM_TRAIN = dec_emb_dim.value
    HID_DIM_TRAIN = hid_dim.value
    N_LAYERS_TRAIN = n_layers.value
    ENC_DROPOUT_TRAIN = enc_dropout.value
    DEC_DROPOUT_TRAIN = dec_dropout.value
    LEARNING_RATE = learning_rate.value
    N_EPOCHS = n_epochs.value

    # Use MPS (Metal Performance Shaders) on Apple Silicon, CUDA on NVIDIA GPUs, or CPU
    if torch.backends.mps.is_available():
        device_train = torch.device("mps")
    elif torch.cuda.is_available():
        device_train = torch.device("cuda")
    else:
        device_train = torch.device("cpu")

    # Create model
    enc_train = Encoder(
        INPUT_DIM_TRAIN,
        ENC_EMB_DIM_TRAIN,
        HID_DIM_TRAIN,
        N_LAYERS_TRAIN,
        ENC_DROPOUT_TRAIN,
    )
    dec_train = Decoder(
        OUTPUT_DIM_TRAIN,
        DEC_EMB_DIM_TRAIN,
        HID_DIM_TRAIN,
        N_LAYERS_TRAIN,
        DEC_DROPOUT_TRAIN,
    )
    model_train = Seq2Seq(enc_train, dec_train, device_train).to(device_train)

    # Loss and optimizer
    PAD_IDX = src_vocab.word2idx[PAD_TOKEN]
    criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(model_train.parameters(), lr=LEARNING_RATE)


    def train_epoch(model, iterator, optimizer, criterion, clip=1):
        model.train()
        epoch_loss = 0

        for src, trg in iterator:
            src, trg = src.to(device_train), trg.to(device_train)

            optimizer.zero_grad()

            # Forward pass
            output = model(src, trg, teacher_forcing_ratio=0.5)

            # Calculate loss (ignore first token which is <SOS>)
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)

            loss = criterion(output, trg)
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

            optimizer.step()
            epoch_loss += loss.item()

        return epoch_loss / len(iterator)


    def evaluate(model, iterator, criterion):
        model.eval()
        epoch_loss = 0

        with torch.no_grad():
            for src, trg in iterator:
                src, trg = src.to(device_train), trg.to(device_train)

                # No teacher forcing during evaluation
                output = model(src, trg, teacher_forcing_ratio=0)

                output_dim = output.shape[-1]
                output = output[:, 1:].reshape(-1, output_dim)
                trg = trg[:, 1:].reshape(-1)

                loss = criterion(output, trg)
                epoch_loss += loss.item()

        return epoch_loss / len(iterator)


    # Train for a few epochs

    best_test_loss = float("inf")

    training_log = []
    train_losses_list = []
    test_losses_list = []

    # Create figure for live plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    if button.value:
        print(f"Training on device: {device_train}")
        print(
            f"Model parameters: {sum(p.numel() for p in model_train.parameters()):,}\n"
        )
        print("Training Configuration:")
        print("=" * 40)
        print(f"Encoder Embedding Dim:  {ENC_EMB_DIM_TRAIN}")
        print(f"Decoder Embedding Dim:  {DEC_EMB_DIM_TRAIN}")
        print(f"Hidden Dimension:       {HID_DIM_TRAIN}")
        print(f"Number of Layers:       {N_LAYERS_TRAIN}")
        print(f"Encoder Dropout:        {ENC_DROPOUT_TRAIN}")
        print(f"Decoder Dropout:        {DEC_DROPOUT_TRAIN}")
        print(f"Learning Rate:          {LEARNING_RATE}")
        print(f"Number of Epochs:       {N_EPOCHS}")
        print("=" * 40 + "\n")
        for epoch in range(N_EPOCHS):
            start_time = time.time()

            train_loss = train_epoch(
                model_train, train_loader, optimizer, criterion
            )
            test_loss = evaluate(model_train, test_loader, criterion)

            end_time = time.time()
            epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

            # Track best test loss
            if test_loss < best_test_loss:
                best_test_loss = test_loss

            # Store losses for plotting
            train_losses_list.append(train_loss)
            test_losses_list.append(test_loss)

            log_entry = f"Epoch: {epoch + 1:02} | Time: {int(epoch_mins)}m {int(epoch_secs)}s | Train Loss: {train_loss:.3f} | Test Loss: {test_loss:.3f}"
            training_log.append(log_entry)
            print(log_entry)

            # Update plot every epoch
            ax.clear()
            epochs_so_far = range(1, len(train_losses_list) + 1)
            ax.plot(
                epochs_so_far,
                train_losses_list,
                "b-",
                label="Train Loss",
                linewidth=2,
            )
            ax.plot(
                epochs_so_far,
                test_losses_list,
                "r-",
                label="Test Loss",
                linewidth=2,
            )
            ax.set_xlabel("Epoch", fontsize=12)
            ax.set_ylabel("Loss", fontsize=12)
            ax.set_title(
                "Training and Test Loss over Epochs",
                fontsize=14,
                fontweight="bold",
            )
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            # Display the plot
            mo.output.replace(ax)

        mo.md(f"""
        ### Training Complete!

        ```
        {chr(10).join(training_log)}
        ```

        Best test loss: **{best_test_loss:.3f}**
        """)

        # Return the figure to keep plot visible
        mo.output.append(ax)
    return device_train, model_train


@app.cell
def _(mo):
    mo.md("""
    ## Part 6: Interactive Simplification Demo

    Now let's use our trained model to simplify sentences!
    """)
    return


@app.cell
def _(mo):
    simplify_input = mo.ui.text_area(
        label="Enter a complex sentence to simplify:",
        value="A Georgian inscription around the drum attests his name.",
        rows=3,
    )
    simplify_input
    return (simplify_input,)


@app.cell
def _(
    SOS_TOKEN,
    device_train,
    mo,
    model_train,
    simplify_input,
    src_vocab,
    tgt_vocab,
    torch,
):
    def simplify_sentence(
        sentence, model, src_vocab, tgt_vocab, device, max_len=50
    ):
        """Simplify a sentence using the trained model"""
        model.eval()

        # Encode input sentence
        src_indices = src_vocab.encode(sentence, max_len)
        src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device)

        # Encode
        with torch.no_grad():
            hidden, cell = model.encoder(src_tensor)

        # Start decoding with <SOS> token
        trg_indices = [tgt_vocab.word2idx[SOS_TOKEN]]

        # Debug: track predictions
        pred_tokens_debug = []

        for i in range(max_len):
            trg_tensor = torch.LongTensor([trg_indices[-1]]).to(device)

            with torch.no_grad():
                output, hidden, cell = model.decoder(trg_tensor, hidden, cell)

            # Get top-5 predictions for debugging
            top5_indices = torch.topk(output, 5).indices
            pred_token = top5_indices[0, 0].item()

            pred_tokens_debug.append(
                {
                    "token_id": pred_token,
                    "word": tgt_vocab.idx2word.get(pred_token, "???"),
                    "top5_words": [
                        tgt_vocab.idx2word.get(idx.item(), "???")
                        for idx in top5_indices[0]
                    ],
                }
            )

            trg_indices.append(pred_token)

            # Stop if we predict <EOS>
            if pred_token == tgt_vocab.word2idx.get("<EOS>", 2):
                break

        # Decode to text
        simplified = tgt_vocab.decode(trg_indices)
        return simplified, pred_tokens_debug


    # Simplify the input
    input_sentence = simplify_input.value.strip()


    simplified_output, pred_tokens_debug = simplify_sentence(
        input_sentence, model_train, src_vocab, tgt_vocab, device_train
    )
    mo.md(f"""
    ### Simplification Result

    **Original (Complex):**
    > {input_sentence}

    **Simplified by Model:**
    > {simplified_output}

    **predictions**
    {pred_tokens_debug}

    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Why do we have no decrease in test loss? Too many out of vocabulary words.
    """)
    return


if __name__ == "__main__":
    app.run()
