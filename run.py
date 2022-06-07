import os
import time
import torch
import pickle
import random
import argparse
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, AutoTokenizer, AutoModel, BertSememeModel, DistilBertSememeModel

random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

learning_rate = 3e-5

SIMILARITIES_TEMPERATURE = 1

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


def make_context_tensors(former, latter):
	sentences = [ f + PRETRAINED_MASK_TOKEN + l for f, l in zip(former, latter)]
	encoded_dict = tokenizer.batch_encode_plus(sentences,
											add_special_tokens=True,
											max_length=150,
											pad_to_max_length=True,
											truncation=True,
											return_attention_mask=True,
											return_token_type_ids=True,
											return_tensors='pt')
	input_ids = encoded_dict['input_ids']
	token_type_ids = encoded_dict['token_type_ids']
	attention_masks = encoded_dict['attention_mask']
	mask_ids = torch.tensor([row.tolist().index(PRETRAINED_MASK_TOKEN_ID) for row in encoded_dict['input_ids']])
	return input_ids, token_type_ids, attention_masks, mask_ids


# Dataset and DataLoader
class Dataset(Dataset):
	def __init__(self, input_ids, token_type_ids, attention_masks, mask_ids, quote):
		self.input_ids = input_ids
		self.token_type_ids = token_type_ids
		self.attention_masks = attention_masks
		self.mask_ids = mask_ids
		self.quote = quote

	def __len__(self):
		return len(self.input_ids)

	def __getitem__(self, idx):
		if self.quote is None:
			return self.input_ids[idx], self.token_type_ids[idx], self.attention_masks[idx], self.mask_ids[idx]
		return self.input_ids[idx], self.token_type_ids[idx], self.attention_masks[idx], self.mask_ids[idx], self.quote[idx]


#  Generate negative examples according to num
def generate_quotes(quote, num, method):
	pos_quote = quote
	if method == 'random':
		quotes_select = all_quotes[:]
		quotes_select.remove(quote)
		neg_quotes = random.sample(quotes_select, num)
	elif method == 'probability':
		pos_idx = all_quotes.index(quote)
		similarities = all_quote_similarities[pos_idx, :]
		neg_probs = (1-similarities) / np.sum(1-similarities) # range is [0, 1]
		neg_idx = np.random.choice(np.arange(neg_probs.shape[0]), size=num, p=neg_probs)
		neg_quotes = [all_quotes[i] for i in neg_idx]
	elif method == 'extreme':
		pos_idx = all_quotes.index(quote)
		similarities = all_quote_similarities[pos_idx]
		sorted_idx = np.argsort(similarities)
		neg_idx = sorted_idx[:num]
		neg_quotes = [all_quotes[i] for i in neg_idx]
	quotes = neg_quotes + [pos_quote]
	random.shuffle(quotes)
	return pos_quote, quotes


def make_quote_tensors(quote):
	pos_quote, quotes = generate_quotes(quote, num=NUM_NEGATIVES, method=SAMPLING_STRATEGY)
	label = quotes.index(pos_quote)
	input_ids = []
	encoded_dict = tokenizer.batch_encode_plus(quotes,
											add_special_tokens=True,
											max_length=80,
											pad_to_max_length=True,
											truncation=True,
											return_tensors='pt')
	input_ids = encoded_dict['input_ids']  # [num, 80]
	quote_ids = [all_quotes_dict[q] for q in quotes]
	return input_ids, label, torch.LongTensor(quote_ids)


# Define network
class Context_Encoder(nn.Module):
	def __init__(self):
		super().__init__()
		self.bert_model = AutoModel.from_pretrained(PRETRAINED_MODEL_NAME)
		self.dropout = nn.Dropout(0.5)

	def forward(self, context_input_ids, context_token_type_ids, context_attention_masks, mask_ids):
		outputs = self.bert_model(input_ids=context_input_ids,
								  token_type_ids=context_token_type_ids,
								  attention_mask=context_attention_masks)
		last_hidden_state = outputs[0]  # [batch_size, sequence_length, hidden_size]
		all_context = []
		for i in range(len(last_hidden_state)):
			hidden_state = last_hidden_state[i]  # [sequence_length, hidden_size]
			mask = hidden_state[mask_ids[i]]
			mask = self.dropout(mask)
			context = mask.unsqueeze(dim=0)  # context: [1, hidden_size]
			all_context.append(context)
		all_context = torch.cat(all_context, dim=0)  # all_context: [batch, hidden_size]
		return all_context


class Quote_Encoder(nn.Module):
	def __init__(self, base):
		super().__init__()

		if base == 'bert':
			self.bert_model = BertSememeModel.from_pretrained(PRETRAINED_MODEL_NAME)
		elif base == 'distilbert':
			self.bert_model = DistilBertSememeModel.from_pretrained(PRETRAINED_MODEL_NAME)
		# self.dropout = nn.Dropout(0.5)

	def forward(self, quotes):
		quote_tensor = []
		labels = []
		for quote in quotes:
			quote_input_ids, label, quote_ids = make_quote_tensors(quote)
			quote_input_ids = quote_input_ids.to(device)
			quote_ids = quote_ids.to(device)
			outputs = self.bert_model(input_ids=quote_input_ids, quote_ids=quote_ids)
			last_hidden_state = outputs[0]  # (num, sequence_length, hidden_size))
			# last_hidden_state = self.dropout(last_hidden_state)
			output = torch.mean(last_hidden_state, dim=1)  # (num, hidden_size))
			quote_tensor.append(output)
			labels.append(label)
		quote_tensor = torch.stack(quote_tensor, dim=0)  # (batch, num, hidden_size))
		return quote_tensor, labels


class QuotRecNet1(nn.Module):
	def __init__(self, contex_model, quote_model):
		super().__init__()
		self.contex_model = contex_model
		self.quote_model = quote_model

	def forward(self, input_ids, token_type_ids, attention_masks, mask_ids, quotes):
		# context_output: [batch, hidden_size]
		context_output = self.contex_model(input_ids, token_type_ids, attention_masks, mask_ids)
		context_output = context_output.unsqueeze(dim=1)  # [batch, 1, hidden_size]

		# quote_output: [batch, num, hidden_size]  labels: [batch]
		quote_output, labels = self.quote_model(quotes)
		quote_output = quote_output.permute(0, 2, 1)

		outputs = torch.matmul(context_output, quote_output).squeeze(dim=1)  # output: [batch, num_quotes]
		return outputs, torch.LongTensor(labels)


def training(model, epoch, train, valid, device):
	total = sum(p.numel() for p in model.parameters())
	trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print('start training, parameter total:{}, trainable:{}'.format(total, trainable))
	t_batch = len(train)
	v_batch = len(valid)

	model.train()
	criterion = nn.CrossEntropyLoss()
	optimizer = AdamW(model.parameters(), lr=learning_rate)
	best_acc = 0
	count = 0
	for curr_epoch in range(epoch):
		start = time.perf_counter()
		total_loss, total_acc = 0, 0
		print('epoch: ', curr_epoch + 1)
		# Train
		for i, (input_ids, token_type_ids, attention_masks, mask_ids, quotes) in tqdm(enumerate(train), total=len(train)):
			input_ids = input_ids.to(device)
			token_type_ids = token_type_ids.to(device)
			attention_masks = attention_masks.to(device)
			mask_ids = mask_ids.to(device, dtype=torch.long)
			optimizer.zero_grad()
			outputs, labels = model(input_ids, token_type_ids, attention_masks, mask_ids, quotes)
			labels = labels.to(device, dtype=torch.long)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
			_, pred = torch.max(outputs.cpu().data, 1)
			acc = accuracy_score(pred, labels.cpu())
			total_loss += loss.item()
			total_acc += acc
		print('Train | Loss: {:.5f} Accuracy: {:.3f}'.format(total_loss, total_acc / t_batch))

		# Validation
		model.eval()
		with torch.no_grad():
			total_loss, total_acc = 0, 0
			for i, (input_ids, token_type_ids, attention_masks, mask_ids, quotes) in tqdm(enumerate(valid), total=len(valid)):
				input_ids = input_ids.to(device)
				token_type_ids = token_type_ids.to(device)
				attention_masks = attention_masks.to(device)
				mask_ids = mask_ids.to(device, dtype=torch.long)
				outputs, labels = model(input_ids, token_type_ids, attention_masks, mask_ids, quotes)
				labels = labels.to(device, dtype=torch.long)
				loss = criterion(outputs, labels)
				_, pred = torch.max(outputs.cpu().data, 1)
				acc = accuracy_score(pred, labels.cpu())
				total_loss += loss.item()
				total_acc += acc
			print('Valid | Loss: {:.5f} Accuracy: {:.3f}'.format(total_loss, total_acc / v_batch))

			if total_acc > best_acc:
				best_acc = total_acc
				torch.save(model.quote_model.state_dict(), f'./model/{MODEL_SAVE_PATH}/english_quote.pth')
				torch.save(model.contex_model.state_dict(), f'./model/{MODEL_SAVE_PATH}/english_context.pth')
				print('saving model with Acc {:.3f} '.format(total_acc / v_batch))
				count = 0
			else:
				count += 1

		model.train()
		end = time.perf_counter()
		print('epoch running time: {:.0f}s'.format(end - start))
		# early stopping
		if count == 3:
			break


# make quotes to bert tensor
def make_tensors(quotes):
	input_ids = []
	encoded_dict = tokenizer.batch_encode_plus(quotes,
											 add_special_tokens=True,
											 max_length=80,
											 pad_to_max_length=True,
											 truncation=True,
											 return_tensors='pt')
	input_ids = encoded_dict['input_ids']
	return input_ids


# Use the mask method for training
class QuotRecNet2(nn.Module):
	def __init__(self):
		super().__init__()
		self.bert_model = AutoModel.from_pretrained(PRETRAINED_MODEL_NAME)
		self.dropout = nn.Dropout(0.5)

	def forward(self, input_ids, token_type_ids, attention_masks, mask_ids, quote_tensor):
		outputs = self.bert_model(input_ids=input_ids,
								  token_type_ids=token_type_ids,
								  attention_mask=attention_masks)
		last_hidden_state = outputs[0]  # [batch_size, sequence_length, hidden_size]
		all_outputs = []
		for i in range(len(last_hidden_state)):
			hidden_state = last_hidden_state[i]  # [sequence_length, hidden_size]
			mask = hidden_state[mask_ids[i]]
			context = self.dropout(mask)
			context = context.unsqueeze(dim=0)  # context: [1, hidden_size]
			# quote_tensor: [num_class, hidden_size]
			output = torch.mm(context, quote_tensor.t())  # outputs: [1, num_class]
			all_outputs.append(output)
		all_outputs = torch.cat(all_outputs, dim=0)  # all_outputs: [batch, num_class]
		return all_outputs


def generate_quote_tensors(all_quotes):
	quote_input_ids = make_tensors(all_quotes)
	print(f'quote bert input: {quote_input_ids.shape}')

	# Generate sentence vector for quotes
	quote_model = AutoModel.from_pretrained(PRETRAINED_MODEL_NAME)
	model_dict = quote_model.state_dict()
	save_model_state = torch.load(f'./model/{MODEL_SAVE_PATH}/english_quote.pth')
	state_dict = {k[11:]: v for k, v in save_model_state.items() if k[11:] in model_dict.keys()}
	model_dict.update(state_dict)
	quote_model.load_state_dict(model_dict)
	quote_model = quote_model.to(device)
	quote_input_ids = quote_input_ids.to(device)

	quote_embeddings = []
	quote_model.eval()
	with torch.no_grad():
		for input_ids in quote_input_ids:
			input_ids = input_ids.unsqueeze(dim=0)
			outputs = quote_model(input_ids=input_ids)
			hidden_states = outputs[0]  # hidden_states:[batch_size, sequence_length, hidden_size]
			quote_tensor = torch.mean(hidden_states, dim=1)  # quote_tensor: [batch_size, hidden_size]
			quote_embeddings.append(quote_tensor)
		quote_embeddings = torch.cat(quote_embeddings, dim=0)
	return quote_embeddings


# get rank
def rank_gold(predicts, golds):
	ranks = []
	ps = predicts.data.cpu().numpy()
	gs = golds.cpu().numpy()
	for i in range(len(ps)):
		predict = ps[i]
		gold_index = gs[i]
		predict_value = predict[gold_index]
		predict_sort = sorted(predict, reverse=True)
		predict_index = predict_sort.index(predict_value)
		if predict_index == -1:
			break
		ranks.append(predict_index)
	return ranks


# get NDCG@5
def get_NDCG(ranks):
	total = 0.0
	for r in ranks:
		if r < 5:  # k=5
			total += 1.0 / np.log2(r + 2)
	return total / len(ranks)


# get recall@k
def recall(predicts, golds):
	predicts = predicts.data.cpu().numpy()
	golds = golds.cpu().numpy()
	predicts_index = np.argsort(-predicts, axis=1)
	recall_1, recall_3, recall_5, recall_10, recall_20, recall_30 = 0, 0, 0, 0, 0, 0
	recall_100, recall_300, recall_500, recall_1000 = 0, 0, 0, 0
	for i in range(len(golds)):
		if golds[i] in predicts_index[i][:1000]:
			recall_1000 += 1
			if golds[i] in predicts_index[i][:500]:
				recall_500 += 1
				if golds[i] in predicts_index[i][:300]:
					recall_300 += 1
					if golds[i] in predicts_index[i][:100]:
						recall_100 += 1
						if golds[i] in predicts_index[i][:30]:
							recall_30 += 1
							if golds[i] in predicts_index[i][:20]:
								recall_20 += 1
								if golds[i] in predicts_index[i][:10]:
									recall_10 += 1
									if golds[i] in predicts_index[i][:5]:
										recall_5 += 1
										if golds[i] in predicts_index[i][:3]:
											recall_3 += 1
											if golds[i] in predicts_index[
													i][:1]:
												recall_1 += 1
	return recall_1, recall_3, recall_5, recall_10, recall_20, recall_30, recall_100, recall_300, recall_500, recall_1000


def training_mask(model, epoch, train, valid, quote_tensor, device):
	total = sum(p.numel() for p in model.parameters())
	trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print('start training, parameter total:{}, trainable:{}'.format(total, trainable))
	t_batch = len(train)
	v_batch = len(valid)
	learning_rate = 5e-5
	model.train()
	criterion = nn.CrossEntropyLoss()
	optimizer = AdamW(model.parameters(), lr=learning_rate)
	best_MRR = 0
	count = 0
	quote_tensor = quote_tensor.to(device)
	for curr_epoch in range(epoch):
		start = time.perf_counter()
		print('epoch: ', curr_epoch + 1)
		total_loss, total_MRR, total_NDCG = 0, 0, 0
		# Train
		for i, (input_ids, token_type_ids, attention_masks, mask_ids, labels) in tqdm(enumerate(train), total=len(train)):
			input_ids = input_ids.to(device)
			token_type_ids = token_type_ids.to(device)
			attention_masks = attention_masks.to(device)
			mask_ids = mask_ids.to(device, dtype=torch.long)
			labels = torch.tensor([all_quotes_dict[q] for q in labels])
			labels = labels.to(device, dtype=torch.long)
			optimizer.zero_grad()
			outputs = model(input_ids, token_type_ids, attention_masks, mask_ids, quote_tensor)  # outputs: (batch, num_class)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
			ranks = rank_gold(outputs, labels)
			MRR = np.average([1.0 / (r + 1) for r in ranks])
			NDCG = get_NDCG(ranks)
			total_loss += loss.item()
			total_MRR += MRR
			total_NDCG += NDCG
		end = time.perf_counter()
		print('Epoch running time : {:.0f}'.format(end - start))
		print('Train | Loss: {:.3f} MRR: {:.3f} NDCG: {:.3f}'.format(total_loss, total_MRR/t_batch, total_NDCG/t_batch))

		# Validation
		model.eval()
		with torch.no_grad():
			total_loss, total_MRR, total_NDCG = 0, 0, 0
			for i, (input_ids, token_type_ids, attention_masks, mask_ids, labels) in tqdm(enumerate(valid), total=len(valid)):
				input_ids = input_ids.to(device)
				token_type_ids = token_type_ids.to(device)
				attention_masks = attention_masks.to(device)
				mask_ids = mask_ids.to(device, dtype=torch.long)
				labels = torch.tensor([all_quotes_dict[q] for q in labels])
				labels = labels.to(device, dtype=torch.long)
				outputs = model(input_ids, token_type_ids, attention_masks, mask_ids, quote_tensor)
				loss = criterion(outputs, labels)
				ranks = rank_gold(outputs, labels)
				MRR = np.average([1.0 / (r + 1) for r in ranks])
				NDCG = get_NDCG(ranks)
				total_loss += loss.item()
				total_MRR += MRR
				total_NDCG += NDCG
			print('Valid | Loss: {:.5f} MRR: {:.3f} NDCG: {:.3f}'.format(total_loss, total_MRR / v_batch, total_NDCG / v_batch))

		if total_MRR > best_MRR:
			best_MRR = total_MRR
			torch.save(model, f'./model/{MODEL_SAVE_PATH}/model_english.model')
			print('saving model with MRR {:.3f} NDCG: {:.3f}'.format(total_MRR / v_batch, total_NDCG / v_batch))
			count = 0
		else:
			learning_rate = learning_rate * 0.9
			count += 1

		# Early stopping
		if count == 3:
			break
		model.train()


def test(model, test_loader, quote_tensor, device):
	print('start test')
	model.eval()
	t_batch = len(test_loader)
	criterion = nn.CrossEntropyLoss()
	quote_tensor = quote_tensor.to(device)
	with torch.no_grad():
		total_loss, total_MRR, total_NDCG, total_ranks = 0, 0, 0, 0
		total_recall_1, total_recall_3, total_recall_5, total_recall_10, total_recall_20, total_recall_30 = 0, 0, 0, 0, 0, 0
		total_recall_100, total_recall_300, total_recall_500, total_recall_1000 = 0, 0, 0, 0
		all_ranks = []
		for i, (input_ids, token_type_ids, attention_masks, mask_ids, labels) in tqdm(enumerate(test_loader), total=len(test_loader)):
			input_ids = input_ids.to(device)
			token_type_ids = token_type_ids.to(device)
			attention_masks = attention_masks.to(device)
			mask_ids = mask_ids.to(device, dtype=torch.long)
			labels = torch.tensor([all_quotes_dict[q] for q in labels])
			labels = labels.to(device, dtype=torch.long)
			outputs = model(input_ids, token_type_ids, attention_masks, mask_ids, quote_tensor)
			loss = criterion(outputs, labels)
			ranks = rank_gold(outputs, labels)
			all_ranks.extend(ranks)
			MRR = np.average([1.0 / (r + 1) for r in ranks])
			NDCG = get_NDCG(ranks)
			recall_1, recall_3, recall_5, recall_10, recall_20, recall_30, recall_100, recall_300, recall_500, recall_1000 = recall(outputs, labels)
			total_loss += loss.item()
			total_MRR += MRR
			total_NDCG += NDCG
			total_ranks += np.sum(ranks)
			total_recall_1 += recall_1
			total_recall_3 += recall_3
			total_recall_5 += recall_5
			total_recall_10 += recall_10
			total_recall_20 += recall_20
			total_recall_30 += recall_30
			total_recall_100 += recall_100
			total_recall_300 += recall_300
			total_recall_500 += recall_500
			total_recall_1000 += recall_1000
		print(
			'Test | Loss: {:.5f} MRR: {:.3f} NDCG: {:.3f} Mean Rank: {:.0f} Median Rank: {:.0f} Variance: {:.0f}'
			.format(total_loss, total_MRR / t_batch,
					total_NDCG / t_batch, np.mean(all_ranks),
					np.median(all_ranks)+1,
					np.std(all_ranks)))
		print(
			'Recall@1: {:.4f} Recall@3: {:.4f} Recall@5: {:.4f} Recall@10: {:.4f} Recall@20: {:.4f} Recall@30: {:.4f} Recall@100: {:.4f} Recall@300: {:.4f} Recall@500: {:.4f} Recall@1000: {:.4f}'
			.format(
				total_recall_1 / len(y_test), total_recall_3 / len(y_test),
				total_recall_5 / len(y_test), total_recall_10 / len(y_test),
				total_recall_20 / len(y_test), total_recall_30 / len(y_test),
				total_recall_100 / len(y_test), total_recall_300 / len(y_test),
				total_recall_500 / len(y_test),
				total_recall_1000 / len(y_test)))


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("--phase", dest="phase", action="store", type=str)
	parser.add_argument("--base", dest="base", action="store", type=str, choices=['bert', 'distilbert'])
	parser.add_argument("--sampling", dest="sampling", action="store", type=str, default='random', choices=['random', 'extreme', 'probability'])
	parser.add_argument("--batch_size", dest="batch_size", action="store", type=int, default=4)
	parser.add_argument("--num_epochs", dest="num_epochs", action="store", type=int, default=40)
	parser.add_argument("--num_negatives", dest="num_negatives", action="store", type=int, default=19)
	args = parser.parse_args()

	NUM_NEGATIVES = args.num_negatives
	BATCH_SIZE = args.batch_size
	NUM_EPOCHS = args.num_epochs
	SAMPLING_STRATEGY = args.sampling

	MODEL_SAVE_PATH = f'{args.base}_{SAMPLING_STRATEGY}_neg{NUM_NEGATIVES}_bs{BATCH_SIZE}_epochs{NUM_EPOCHS}'
	os.makedirs(f'model/{MODEL_SAVE_PATH}', exist_ok=True)
	print(f'Model: {MODEL_SAVE_PATH}')

	data_path = './data/english.txt'
	similarities_path = 'data/similarities.pkl'

	if args.base == 'bert':
		PRETRAINED_MODEL_NAME = 'bert-base-uncased'
	elif args.base == 'distilbert':
		PRETRAINED_MODEL_NAME = 'distilbert-base-uncased'

	print('Loading dataset')
	train_former, train_latter, train_quote, valid_former, valid_latter, valid_quote, test_former, test_latter, test_quote, y_train, y_valid, y_test, all_quotes = load_data(data_path)
	all_quotes_dict = {quote:idx for idx, quote in enumerate(all_quotes)}
	print('tran  valid  test:', len(train_former), len(valid_former), len(test_former))
	print('all quotes: ', len(all_quotes))
	print('train quote:', len(list(set(train_quote))))
	print('valid quote:', len(list(set(valid_quote))))
	print('test quote:', len(list(set(test_quote))))

	# Get the Tokenizer used for pretraining model
	tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
	PRETRAINED_MASK_TOKEN = tokenizer.mask_token
	PRETRAINED_MASK_TOKEN_ID = tokenizer.convert_tokens_to_ids(PRETRAINED_MASK_TOKEN)

	print('Loading train and valid data')
	train_input_ids, train_token_type_ids, train_attention_masks, train_mask_ids = make_context_tensors(train_former, train_latter)
	valid_input_ids, valid_token_type_ids, valid_attention_masks, valid_mask_ids = make_context_tensors(valid_former, valid_latter)
	print(f'train bert input: {train_input_ids.shape} {train_token_type_ids.shape} {train_attention_masks.shape} {train_mask_ids.shape}')
	print(f'valid bert input: {valid_input_ids.shape} {valid_token_type_ids.shape} {valid_attention_masks.shape} {valid_mask_ids.shape}')

	print('Loading train and valid dataloader')
	train_dataset = Dataset(input_ids=train_input_ids,
							token_type_ids=train_token_type_ids,
							attention_masks=train_attention_masks,
							mask_ids=train_mask_ids,
							quote=train_quote)
	valid_dataset = Dataset(input_ids=valid_input_ids,
							token_type_ids=valid_token_type_ids,
							attention_masks=valid_attention_masks,
							mask_ids=valid_mask_ids,
							quote=valid_quote)


	if args.phase == 'train1':
		# Loading quote similarities for weak supervision
		with open(similarities_path, 'rb') as f:
			dat = pickle.load(f)
		assert all_quotes == dat['quotes']
		all_quote_similarities = dat['similarities']

		# Converting cosine similarities to probabilities
		all_quote_similarities = F.softmax(torch.tensor(all_quote_similarities / SIMILARITIES_TEMPERATURE), dim=1).numpy()

		train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
		valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

		print('Loading model')
		contex_model = Context_Encoder()
		quote_model = Quote_Encoder(base=args.base)
		model = QuotRecNet1(contex_model, quote_model)
		model.to(device)

		training(model=model,
				epoch=NUM_EPOCHS,
				train=train_loader,
				valid=valid_loader,
				device=device)

	elif args.phase == 'train2':
		quote_embeddings = generate_quote_tensors(all_quotes)
		print(f'quote tensor: {quote_embeddings.shape}')

		print('Loading model')
		model = QuotRecNet2()
		model.to(device)

		train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=2)
		valid_loader = DataLoader(dataset=valid_dataset, batch_size=32, shuffle=True, num_workers=2)

		print('Start training')
		training_mask(model=model,
					epoch=NUM_EPOCHS,
					train=train_loader,
					valid=valid_loader,
					quote_tensor=quote_embeddings,
					device=device)

	elif args.phase == 'test':
		quote_embeddings = generate_quote_tensors(all_quotes)
		print(f'quote tensor: {quote_embeddings.shape}')

		print('Loading test tensor')
		test_input_ids, test_token_type_ids, test_attention_masks, test_mask_ids = make_context_tensors(test_former, test_latter)
		test_dataset = Dataset(input_ids=test_input_ids,
							token_type_ids=test_token_type_ids,
							attention_masks=test_attention_masks,
							mask_ids=test_mask_ids,
							quote=test_quote)

		test_loader = DataLoader(dataset=test_dataset, batch_size=32, num_workers=2)

		print('Loading model')
		model = torch.load(f'./model/{MODEL_SAVE_PATH}/model_english.model')
		model.to(device)
		test(model=model,
			test_loader=test_loader,
			quote_tensor=quote_embeddings,
			device=device)
