import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune BERT classification on rolling-profile data and plot last predictions"
    )
    parser.add_argument('--data',         default='filtered_reviews.csv', help='CSV from prep step')
    parser.add_argument('--model_name',   default='bert-base-uncased')
    parser.add_argument('--output_dir',   default='results_class')
    parser.add_argument('--max_length',   type=int, default=128)
    parser.add_argument('--batch_size',   type=int, default=8)
    parser.add_argument('--learning_rate',type=float, default=5e-5)
    parser.add_argument('--epochs',       type=int, default=3)
    parser.add_argument('--val_frac',     type=float, default=0.2)
    parser.add_argument('--seed',         type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load and sort reviews
    df = pd.read_csv(args.data).sort_values(['user_id','date']).reset_index(drop=True)

    # Build rolling-profile classification examples
    examples = []
    for uid, group in df.groupby('user_id'):
        history = ""
        for _, row in group.iterrows():
            # map continuous rating to class 0-4
            raw = float(row['label'])
            # round to nearest integer star (1–5), then shift to 0–4
            cls = int(round(raw)) - 1
            # clamp into [0,4]
            cls = max(0, min(cls, 4))
            examples.append({
                'menu_text': row['menu_text'],
                'user_text': history,
                'label':     cls,
                'user_id':   uid,
                'date':      row['date']
            })
            snippet = f"{row['menu_text']} ({int(row['label'])}★): {row['review_text']}"
            history = (history + ' ' + snippet)[-2000:]

    ex_df = pd.DataFrame(examples)

    # Split train/val by user
    from sklearn.model_selection import GroupShuffleSplit
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.val_frac, random_state=args.seed)
    train_idx, val_idx = next(splitter.split(ex_df, groups=ex_df['user_id']))
    train_df = ex_df.iloc[train_idx].reset_index(drop=True)
    val_df   = ex_df.iloc[val_idx].reset_index(drop=True)

    # Convert to HF Dataset (drop user_id,date)
    train_ds = Dataset.from_pandas(train_df.drop(columns=['user_id','date']))
    val_ds   = Dataset.from_pandas(val_df.drop(columns=['user_id','date']))

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    def preprocess(batch):
        enc = tokenizer(
            batch['menu_text'], batch['user_text'],
            truncation=True, padding='max_length', max_length=args.max_length
        )
        enc['labels'] = batch['label']
        return enc

    train_ds = train_ds.map(preprocess, batched=True, remove_columns=['menu_text','user_text','label'])
    val_ds   = val_ds.map(preprocess, batched=True, remove_columns=['menu_text','user_text','label'])
    train_ds.set_format('torch')
    val_ds.set_format('torch')

    # Model config for classification
    config = AutoConfig.from_pretrained(
        args.model_name,
        num_labels=5,
        problem_type='single_label_classification'
    )
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, config=config)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy='epoch',
        save_strategy='epoch',
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        seed=args.seed
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer
    )

    trainer.train()
    trainer.save_model(args.output_dir)

    # Plot last prediction per user
    last_df = val_df.groupby('user_id').tail(1)
    inputs = tokenizer(
        list(last_df['menu_text']), list(last_df['user_text']),
        truncation=True, padding='max_length', max_length=args.max_length,
        return_tensors='pt'
    ).to(trainer.model.device)
    model.eval()
    with torch.no_grad():
        logits = trainer.model(**inputs).logits
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
    true = last_df['label'].values
    plt.figure(figsize=(6,6))
    plt.scatter(true, preds, alpha=0.5)
    plt.plot([0,4],[0,4],'r--')
    plt.xlabel('True Class (0–4)')
    plt.ylabel('Predicted Class')
    plt.title('Last Review: True vs Predicted Class per User')
    plt.tight_layout()
    plt.savefig('last_predictions_class.png')
    print('Saved last_predictions_class.png')

if __name__=='__main__':
    main()
