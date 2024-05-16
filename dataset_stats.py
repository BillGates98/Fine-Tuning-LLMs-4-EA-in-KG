import time
from compute_files import ComputeFile
import argparse
from rdflib import Graph


class DatasetStat:

    def __init__(self, source='', target='', truth=''):
        self.source = source
        self.target = target
        self.truth = truth

    def load_rdf(self, file=''):
        g = Graph()
        g.parse(file)
        return g

    def statistics(self, graph=None):
        output = {'subject': [], 'predicate': []}
        facts = 0
        for s, p, o in graph:
            if s not in output['subject']:
                output['subject'].append(s)
            if p not in output['predicate']:
                output['predicate'].append(p)
            facts += 1
        for key in output:
            output[key] = len(output[key])
        print('#> ', output, ' Fact : ', facts)
        return output

    def run(self):
        print(' Statistic source : ', self.source)
        self.statistics(graph=self.load_rdf(file=self.source))
        print(' Statistic target : ', self.target)
        self.statistics(graph=self.load_rdf(file=self.target))
        print(' Statistic truth: ', self.truth)
        self.statistics(graph=self.load_rdf(file=self.truth))
        print('DatasetStat init called')


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
        parser.add_argument("--suffix", type=str, default="person")
        return parser.parse_args()

    start = time.time()
    args = arg_manager()
    source = detect_file(path=args.input_path+args.suffix, type='source')
    target = detect_file(path=args.input_path+args.suffix, type='target')
    truth = detect_file(path=args.input_path+args.suffix, type='same_as')

    print('Dataset : ', args.suffix)

    DatasetStat(source=source, target=target, truth=truth).run()
    print('Running Time : ', (time.time() - start), ' seconds ')
