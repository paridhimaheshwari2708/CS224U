import torch
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, util


# load english dataset
def load_data(path):
	# return words list and labels
	with open(path, 'r', encoding='utf-8') as f:
		lines = f.readlines()
		lines = [line.strip().lower().split('\t') for line in lines]
		train_former = [line[0] for line in lines[:101171]]
		train_quote = [line[1] for line in lines[:101171]]
		train_latter = [line[2] for line in lines[:101171]]
		valid_former = [line[0] for line in lines[101171:113942]]
		valid_quote = [line[1] for line in lines[101171:113942]]
		valid_latter = [line[2] for line in lines[101171:113942]]
		test_former = [line[0] for line in lines[113942:]]
		test_quote = [line[1] for line in lines[113942:]]
		test_latter = [line[2] for line in lines[113942:]]
		all_quotes = train_quote + valid_quote + test_quote
	all_quotes = list(set(all_quotes))
	all_quotes.sort()
	y_train = [all_quotes.index(q) for q in train_quote]
	y_valid = [all_quotes.index(q) for q in valid_quote]
	y_test = [all_quotes.index(q) for q in test_quote]

	return train_former, train_latter, train_quote, \
			valid_former, valid_latter, valid_quote, \
			test_former, test_latter, test_quote, \
			torch.LongTensor(y_train), torch.LongTensor(y_valid), torch.LongTensor(y_test), all_quotes


def mean_pooling(model_output, attention_mask):
	# Take attention mask into account for correct averaging
	token_embeddings = model_output[0] # First element of model_output contains all token embeddings
	input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
	return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def get_sentence_transformer_embeddings(sentences, pooling):
	# Load model from HuggingFace Hub
	tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
	model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
	# Tokenize sentences
	encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
	# Compute token embeddings
	with torch.no_grad():
		model_output = model(**encoded_input)
	if pooling == 'mean':
		sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
	elif pooling == 'cls':
		sentence_embeddings = model_output[0][:, 0, :]
	return sentence_embeddings


def compute_similarity_matrix(sentence_embeddings):
	similarities = cosine_similarity(sentence_embeddings)
	return similarities


if __name__ == '__main__':

	# Loading dataset
	data_path = './data/english.txt'
	train_former, train_latter, train_quote, valid_former, valid_latter, valid_quote, test_former, test_latter, test_quote, y_train, y_valid, y_test, all_quotes = load_data(data_path)

	# Compute embedding for all quotes
	model = SentenceTransformer('all-MiniLM-L6-v2')
	all_quote_embeddings = model.encode(all_quotes, convert_to_tensor=True)

	# Compute cosine similarities
	all_quote_similarities = util.cos_sim(all_quote_embeddings, all_quote_embeddings).cpu().numpy()

	final = {
		'quotes' : all_quotes,
		'similarities' : all_quote_similarities,
	}
	with open('data/similarities.pkl', 'wb') as f:
		pickle.dump(final, f)
