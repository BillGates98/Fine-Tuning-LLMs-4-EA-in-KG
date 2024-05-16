from rdflib import Graph
import networkx as nx
from compute_files import ComputeFile
import argparse
import time
import numpy as np
import math

from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, AutoModelForSequenceClassification, DataCollatorForLanguageModeling


class Main:
    def __init__(self, source='', target='', top=5):
        self.source = source
        self.target = target
        self.nx_graph = nx.Graph()
        self.source_subjects = {}
        self.target_subjects = {}
        self.neighbors = {}
        self.top = top
        self.sim_ceil = 0.5

    def load_graph(self, file=''):
        graph = Graph()
        graph.parse(file)
        return graph

    def add_facts(self, graph=None):
        for s, p, o in graph:
            self.nx_graph.add_edge(s, o)
        return self.nx_graph

    def extract_subjects(self, graph=None):
        output = {}
        for s, p, o in graph:
            if not s in output:
                output[s] = []
            output[s].append((p, o))
        return output

    def compute_degrees(self, graph=None, knowledge=[]):
        output = []
        degrees = []
        for node, deg in graph.degree():
            if node in knowledge:
                output.append((node, deg))
                degrees.append(deg)
        return output, degrees

    def compute_neighbors(self, graph=None, source_nodes=[], target_nodes=[], top=5, ceil=0.5):
        output = []
        # print('length of source nodes : ', len(source_nodes))
        for source_node in source_nodes:
            similarities = nx.jaccard_coefficient(
                graph, [(source_node, target_node) for target_node in target_nodes])
            sorted_similarities_array = sorted(
                similarities, key=lambda tup: tup[2], reverse=True)

            selected_candidates = [
                (u, v, s) for u, v, s in sorted_similarities_array if s >= ceil]
            if len(selected_candidates) > 0:
                output.append((source_node, selected_candidates[:top]))
        return output

    def node_candidates(self, data=[], interval=()):
        output = []
        mean, std = interval
        for node, degree in data:
            if degree >= mean-math.sqrt(std) and degree >= mean+math.sqrt(std):
                output.append(node)
        return output

    def orchestrator(self):
        graph1 = self.load_graph(file=self.source)
        self.source_subjects = self.extract_subjects(graph=graph1)
        graph2 = self.load_graph(file=self.target)
        self.target_subjects = self.extract_subjects(graph=graph2)
        graph = graph1 + graph2
        connected_graph = self.add_facts(graph=graph)

        # compute the degree of the neighbors
        source_node_degrees, s_degrees = self.compute_degrees(
            graph=connected_graph, knowledge=list(self.source_subjects.keys()))
        target_node_degrees, t_degrees = self.compute_degrees(
            graph=connected_graph, knowledge=list(self.target_subjects.keys()))

        # node filtering by mean degree
        mean_source_degree, std_source_degree = np.mean(
            s_degrees), np.std(s_degrees)
        mean_target_degree, std_target_degree = np.mean(
            t_degrees), np.std(t_degrees)
        print(mean_source_degree, std_source_degree)
        print(mean_target_degree, std_target_degree)
        # Reduce nodes in set
        print(len(source_node_degrees))
        print(len(target_node_degrees))
        candidate_subject_source = self.node_candidates(
            data=source_node_degrees, interval=(mean_source_degree, std_source_degree))
        candidate_subject_target = self.node_candidates(
            data=target_node_degrees, interval=(mean_target_degree, std_source_degree))
        print(len(candidate_subject_source))
        print(len(candidate_subject_target))

        #
        # compute neighbors of nodes
        neighbors = self.compute_neighbors(
            graph=connected_graph, source_nodes=candidate_subject_source,
            target_nodes=candidate_subject_target, top=self.top, ceil=self.sim_ceil)

        print("selected candidates : ", len(neighbors))
        # print("neighbors : ", neighbors)
        # apply the recommendations algorithm
        recommender = RecommenderSystem(source_subjects=self.source_subjects,
                                        target_subjects=self.target_subjects, candidates=neighbors).run()
        return None

    def run(self):
        result = self.orchestrator()
        return None


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
        return parser.parse_args()

    start = time.time()
    args = arg_manager()
    source = detect_file(path=args.input_path+args.suffix, type='source')
    target = detect_file(path=args.input_path+args.suffix, type='target')

    print('Dataset : ', args.suffix)

    Main(source=source, target=target).run()
    print('Running Time : ', (time.time() - start), ' seconds ')
