from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1")

# Save all tokenizer files to a local folder
tokenizer.save_pretrained("./nomic-tokenizer")