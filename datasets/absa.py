import pickle
import numpy as np
import utils
from torch.utils.data import Dataset


class ABSADataset(Dataset):
    def __init__(self, fname, tokenizer):
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()
        fin = open(fname+'.graph', 'rb')
        idx2graph = pickle.load(fin)
        fin.close()

        all_data = []
        for i in range(0, len(lines), 3):
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].lower().strip()
            polarity = lines[i + 2].strip()

            text_indices = tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)
            context_indices = tokenizer.text_to_sequence(text_left + " " + text_right)
            left_indices = tokenizer.text_to_sequence(text_left)
            left_with_aspect_indices = tokenizer.text_to_sequence(text_left + " " + aspect)
            right_indices = tokenizer.text_to_sequence(text_right, reverse=True)
            right_with_aspect_indices = tokenizer.text_to_sequence(aspect + " " + text_right, reverse=True)
            aspect_indices = tokenizer.text_to_sequence(aspect)
            left_len = np.sum(left_indices != 0)
            aspect_len = np.sum(aspect_indices != 0)
            aspect_boundary = np.asarray([left_len, left_len + aspect_len - 1], dtype=np.int64)
            polarity = int(polarity) + 1

            text_len = np.sum(text_indices != 0)
            concat_bert_indices = tokenizer.text_to_sequence('[CLS] ' + text_left + " " + aspect + " " + text_right + ' [SEP] ' + aspect + " [SEP]")
            concat_segments_indices = [0] * (text_len + 2) + [1] * (aspect_len + 1)
            concat_segments_indices = utils.pad_and_truncate(concat_segments_indices, tokenizer.max_seq_len)

            text_bert_indices = tokenizer.text_to_sequence("[CLS] " + text_left + " " + aspect + " " + text_right + " [SEP]")
            aspect_bert_indices = tokenizer.text_to_sequence("[CLS] " + aspect + " [SEP]")

            dependency_graph = np.pad(idx2graph[i], \
                ((0,tokenizer.max_seq_len-idx2graph[i].shape[0]),(0,tokenizer.max_seq_len-idx2graph[i].shape[0])), 'constant')

            data = {
                'concat_bert_indices': concat_bert_indices,
                'concat_segments_indices': concat_segments_indices,
                'text_bert_indices': text_bert_indices,
                'aspect_bert_indices': aspect_bert_indices,
                'text_indices': text_indices,
                'context_indices': context_indices,
                'left_indices': left_indices,
                'left_with_aspect_indices': left_with_aspect_indices,
                'right_indices': right_indices,
                'right_with_aspect_indices': right_with_aspect_indices,
                'aspect_indices': aspect_indices,
                'aspect_boundary': aspect_boundary,
                'dependency_graph': dependency_graph,
                'polarity': polarity,
            }

            all_data.append(data)
        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
