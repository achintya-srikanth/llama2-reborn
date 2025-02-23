import argparse
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from llama import Llama, load_pretrained
from tokenizer import Tokenizer
from optimizer import AdamW
from tqdm import tqdm
import os
import time


class JohnWickDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = f.read().split('\n')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        encoding = self.tokenizer.encode(text, bos=True, eos=True)
        if len(encoding) > self.max_len:
            encoding = encoding[:self.max_len]
        return torch.tensor(encoding, dtype=torch.long)


def train(model, train_dataloader, optimizer, device, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch = batch.to(device)
            optimizer.zero_grad()
            logits, _ = model(batch[:, :-1])
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch[:, 1:].contiguous().view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_dataloader):.4f}")


def generate_text(model, tokenizer, prompt, max_new_tokens, temperature, device):
    model.eval()
    input_ids = tokenizer.encode(prompt, bos=True, eos=False)
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=temperature)
    return tokenizer.decode(output[0].tolist())


def save_file_with_unique_name(base_name):
    """
    Save a file with a unique name by appending a timestamp if the file exists.
    """
    base_name = base_name.replace(".txt", "_jw.txt").replace(".pt", "_jw.pt")
    if not os.path.exists(base_name):
        return base_name

    # Append timestamp to avoid overwriting
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    name_parts = os.path.splitext(base_name)
    unique_name = f"{name_parts[0]}_{timestamp}{name_parts[1]}"
    return unique_name


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    tokenizer = Tokenizer(args.max_sentence_len)
    model = load_pretrained(args.pretrained_model_path).to(device)

    if args.option == 'finetune_generate':
        # Fine-tuning step
        dataset = JohnWickDataset(args.train_file, tokenizer, args.max_sentence_len)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        optimizer = AdamW(model.parameters(), lr=args.lr)

        train(model, dataloader, optimizer, device, args.epochs)

        # Save the fine-tuned model with a unique name
        output_model_file = save_file_with_unique_name(args.output_model)
        torch.save(model.state_dict(), output_model_file)
        print(f"Finetuned model saved to {output_model_file}")

        # Generation step
        generated_text = generate_text(model, tokenizer, args.prompt, args.max_new_tokens,
                                       args.temperature, device)

        # Save generated text with a unique name
        generated_output_file = save_file_with_unique_name(args.generated_output_file)
        with open(generated_output_file, "w") as f:
            f.write(generated_text)
        
        print("Generated text:")
        print(generated_text)
        print(f"Generated text saved to {generated_output_file}")

    elif args.option == 'generate':
        # Load fine-tuned model and generate text
        model.load_state_dict(torch.load(args.output_model))
        generated_text = generate_text(model, tokenizer, args.prompt,
                                       args.max_new_tokens,
                                       args.temperature,
                                       device)

        # Save generated text with a unique name
        generated_output_file = save_file_with_unique_name(args.generated_output_file)
        with open(generated_output_file, "w") as f:
            f.write(generated_text)

        print("Generated text:")
        print(generated_text)
        print(f"Generated text saved to {generated_output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--option", choices=['finetune_generate', 'generate'], required=True)
    parser.add_argument("--pretrained_model_path", type=str, required=True)
    parser.add_argument("--train_file", type=str)
    parser.add_argument("--output_model", type=str,
                        default="finetuned_model.pt")  # Default file for saving the model
    parser.add_argument("--generated_output_file", type=str,
                        default="generated_text.txt")  # Default file for saving generated text
    parser.add_argument("--max_sentence_len", type=int,
                        default=512)  # Maximum sentence length for tokenization
    parser.add_argument("--batch_size", type=int,
                        default=4)  # Batch size for fine-tuning
    parser.add_argument("--lr", type=float,
                        default=2e-5)  # Learning rate for fine-tuning
    parser.add_argument("--epochs", type=int,
                        default=3)  # Number of epochs for fine-tuning
    parser.add_argument("--use_gpu", action='store_true')  # Use GPU if available
    parser.add_argument("--prompt", type=str)  # Prompt for text generation
    parser.add_argument("--max_new_tokens", type=int,
                        default=100)  # Maximum number of tokens to generate
    parser.add_argument("--temperature", type=float,
                        default=0.7)  # Sampling temperature for generation

    args = parser.parse_args()
    main(args)
