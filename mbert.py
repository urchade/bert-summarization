from transformers import AutoTokenizer, AutoModelWithLMHead

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-uncased")

model = AutoModelWithLMHead.from_pretrained("bert-base-multilingual-uncased")