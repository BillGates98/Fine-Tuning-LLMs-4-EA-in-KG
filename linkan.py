
import argparse
import time
from compute_files import ComputeFile
import pandas as pd
import tiktoken
from sklearn.decomposition import PCA
import numpy as np
from kan_learning import KanLearning


class LinKan:

    def __init__(self, test='test', train='train'):
        self.test = test
        self.train = train

    def normalize(self, x):
        x = np.array(x)
        value = (x - min(x)) / (max(x) - min(x))
        return value

    def tokenizer(self, text):
        encoding = tiktoken.get_encoding("cl100k_base")
        # encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        encoded = encoding.encode(text)

        return encoded

    def read_csv(self, path='', max_length=-1):
        data = pd.read_csv(path)
        data['input'] = data['input'].apply(self.tokenizer)
        print('Path :', path, 'Length : ', len(data['input']))
        if max_length < 0:
            max_length = max(data['input'].apply(len))
        data['input'] = data['input'].apply(
            lambda x: x + [0] * (max_length - len(x)))
        data['input'] = data['input'].apply(self.normalize)
        indexes = [i for i, v in enumerate(
            data['input']) if len(v) > max_length]
        data = data.drop(index=indexes)
        tmp = [v for v in data['input']]
        tmp = np.array(tmp)
        pca = PCA(n_components=20)
        pca.fit(tmp)
        tmp_out = pca.transform(tmp)
        data['input'] = list(tmp_out)
        return data, max_length

    def run(self):
        train, ml = self.read_csv(path=self.train)
        test, _ = self.read_csv(path=self.test, max_length=ml)
        train_input = np.array(train['input'].to_list())
        train_label = np.array(train['label'].to_list())

        test_input = np.array(test['input'].to_list())
        test_label = np.array(test['label'].to_list())
        output = KanLearning(input_size=train_input.shape[1], output_size=1, train_input=train_input,
                             train_label=train_label, test_input=test_input, test_label=test_label).run()
        return output


if __name__ == "__main__":
    def detect_file(path='', type=''):
        files = ComputeFile(input_path=path).build_list_files()
        for v in files:
            if type in v:
                return v
        return None

    def arg_manager():
        parser = argparse.ArgumentParser()
        parser.add_argument("--input_path", type=str, default="./outputs/")
        parser.add_argument("--suffix", type=str, default="restaurant")
        return parser.parse_args()

    start = time.time()
    args = arg_manager()
    test = detect_file(path=args.input_path+args.suffix, type='test')
    train = detect_file(path=args.input_path+args.suffix, type='train')

    print('Dataset : ', args.suffix)

    LinKan(test=test, train=train).run()
    print('Running Time : ', (time.time() - start), ' seconds ')
    print('\n \n')
