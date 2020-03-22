from transformers import BertTokenizer, BertForQuestionAnswering
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
print('text:', text)
print()
print('question:', question)

input_ids = tokenizer.encode(question, text)
token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]
start_scores, end_scores = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))

print('start_scores:', start_scores)
print('end_scores:', end_scores)

all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
answer = ' '.join(all_tokens[torch.argmax(start_scores): torch.argmax(end_scores)+1])

print('answer:', answer)
assert answer == "a nice puppet"
