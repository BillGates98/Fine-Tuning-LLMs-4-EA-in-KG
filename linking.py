from rdflib import Graph
from compute_files import ComputeFile
import argparse
from tqdm import tqdm
import time
import numpy as np
import validators
import pandas as pd
from sampling import Sampling


class Linking:
    def __init__(self, source='', target='', truth='', suffix='', random_size=0.3):
        self.source = source
        self.target = target
        self.limit = -1
        self.truth = truth
        self.suffix = suffix
        self.truth_subjects = {}
        self.random_size = random_size

    def load_graph(self, file=''):
        graph = Graph()
        print(file)
        graph.parse(file)
        return graph

    def merge_two_dicts(self, x, y):
        z = x.copy()
        z.update(y)
        return z

    def extract_subjects(self, file=None, expected=[]):
        graph = self.load_graph(file=file)
        output = {}
        for s, p, o in graph:
            if s in expected:
                if not s in output:
                    output[s] = []
                output[s].append((p, o))
        return output

    def extract_ground_truths(self, file=None):
        sources = []
        targets = []
        graph = self.load_graph(file=file)
        for s, _, o in graph:
            # if self.limit > 0:
            #     self.limit -= 1
            # else:
            #     break
            sources.append(s)
            targets.append(o)
        return sources, targets

    def string_chain(self, entity=[]):
        output = []
        for p, o in entity:
            if not validators.url(str(o)):
                value = str(o)
                words = value.split(' ')
                is_bad = False
                for word in words:
                    if len(word) > 24:
                        is_bad = True
                if not is_bad:
                    output.append(value)
        return output

    def save_results(self, data={}):
        df = pd.DataFrame.from_dict(data)
        df.to_csv("./outputs/" + self.suffix + "_data.csv")
        return None

    def orchestrator(self):
        source_subjects, target_subjects = self.extract_ground_truths(
            file=self.truth)
        source_entities = self.extract_subjects(
            file=self.source, expected=source_subjects)
        target_entities = self.extract_subjects(
            file=self.target, expected=target_subjects)
        entities = self.merge_two_dicts(source_entities, target_entities)

        Sampling(suffix=self.suffix, source_subjects=source_subjects, target_subjects=target_subjects,
                 entities=entities, random_size=self.random_size).run()

        # self.save_results(data=entities)
        print('End with success !')
        return None

    def run(self):
        return self.orchestrator()


if __name__ == "__main__":
    def detect_file(path='', type=''):
        files = ComputeFile(input_path=path).build_list_files()
        for v in files:
            if type in v:
                return v
        return None

    def arg_manager():
        parser = argparse.ArgumentParser()
        parser.add_argument("--input_path", type=str, default="./inputs/")
        parser.add_argument("--suffix", type=str, default="doremus")
        parser.add_argument("--random_size", type=int, default=0.3)
        return parser.parse_args()

    start = time.time()
    args = arg_manager()
    source = detect_file(path=args.input_path+args.suffix, type='source')
    target = detect_file(path=args.input_path+args.suffix, type='target')
    truth = detect_file(path=args.input_path+args.suffix, type='same_as')

    print('Dataset : ', args.suffix)

    Linking(source=source, target=target,
            truth=truth, suffix=args.suffix, random_size=args.random_size).run()
    print('Running Time : ', (time.time() - start), ' seconds ')
