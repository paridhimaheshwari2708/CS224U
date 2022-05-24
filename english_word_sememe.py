import nltk
import torch
import argparse
import OpenHowNet
import numpy as np
from transformers import AutoTokenizer


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


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("--base", dest="base", action="store", type=str, choices=['bert', 'roberta', 'distilbert', 'mobilebert'])
	args = parser.parse_args()

	if args.base == 'bert':
		PRETRAINED_MODEL_NAME = 'bert-base-uncased'
	elif args.base == 'roberta':
		PRETRAINED_MODEL_NAME = 'roberta-base'
	elif args.base == 'distilbert':
		PRETRAINED_MODEL_NAME = 'distilbert-base-uncased'
	elif args.base == 'mobilebert':
		PRETRAINED_MODEL_NAME = 'google/mobilebert-uncased'

	print("loading dataset......")
	data_path = "./data/english.txt"
	train_former, train_latter, train_quote, valid_former, valid_latter, valid_quote, test_former, test_latter, test_quote, y_train, y_valid, y_test, all_quotes = load_data(data_path)
	print("tran  valid  test:", len(train_former), len(valid_former), len(test_former))
	print("all quotes: ", len(all_quotes))
	print("train quote:", len(list(set(train_quote))))
	print("valid quote:", len(list(set(valid_quote))))
	print("test quote:", len(list(set(test_quote))))

	# load pretrained model
	tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
	pad_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
	cls_id = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
	sep_id = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
	hownet_dict = OpenHowNet.HowNetDict()

	all_quote_words = []
	for s in all_quotes:
		q_words = [w for w in nltk.word_tokenize(s)]
		all_quote_words.extend(q_words)
	all_quote_words = list(set(all_quote_words))
	all_quote_words.sort()
	# Insert in increasing order of index
	for special_id in sorted([pad_id, cls_id, sep_id]):
		all_quote_words.insert(special_id, '<INS>')
	print("all quote words ", len(all_quote_words))
	print(all_quote_words[pad_id])
	print(all_quote_words[cls_id])
	print(all_quote_words[sep_id])

	# get all english sememes
	all_semems = []
	for i in range(len(all_quote_words)):
		semems = hownet_dict.get_sememes_by_word(all_quote_words[i], structured=False, lang="en", merge=True)
		for s in semems:
			all_semems.append(s)
	all_semems = list(set(all_semems))
	all_semems.sort()
	print("all sememes: ", len(all_semems))

	# Generate sememes for each word
	sememe_onehot = np.zeros((len(all_quote_words), len(all_semems)))
	for i in range(len(all_quote_words)):
		semems = hownet_dict.get_sememes_by_word(all_quote_words[i], structured=False, lang="en", merge=True)
		if len(semems) > 0:
			word2sememe = [s for s in semems]
			for s in word2sememe:
				sememe_onehot[i][all_semems.index(s)] = 1
	print("word2sememe one hot: ", sememe_onehot.shape)
	np.save(f"./data/{args.base}_english_word_sememe.npy", sememe_onehot)

	# Generate the word index for each quote
	all_word_ids = []
	for quote in all_quotes:
		quote_words = [w for w in nltk.word_tokenize(quote)]
		token2word = []
		for w in quote_words:
			tokens = tokenizer.tokenize(w)
			ids = tokenizer.convert_tokens_to_ids(tokens)
			for id in ids:
				token2word.append([id, all_quote_words.index(w)])

		encoded_dict = tokenizer.encode_plus(quote,
											add_special_tokens=True,
											max_length=80,
											pad_to_max_length=True,
											truncation=True,
											return_attention_mask=True,
											return_tensors='pt')
		input_ids = encoded_dict['input_ids']
		input_ids = input_ids.squeeze().tolist()
		word_ids = []
		for id in input_ids:
			if (id == cls_id) or (id == sep_id) or (id == pad_id):
				word_ids.append(id)
			else:
				for i in range(len(token2word)):
					if id == token2word[i][0]:
						word_ids.append(token2word[i][1])
						token2word = token2word[i+1:]
						break
		if len(word_ids) != 80:
			word_ids.extend(pad_id for _ in range(80-len(word_ids)))
		all_word_ids.append(torch.LongTensor(word_ids))

	all_word_ids = torch.stack(all_word_ids, dim=0)
	print("all word ids: ", all_word_ids.shape)

	torch.save(all_word_ids, f"./data/{args.base}_quote2word.pt")
