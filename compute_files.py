import os
from datetime import datetime
from rdflib import Graph


class ComputeFile:

    def __init__(self, input_path='', output_path=''):
        self.input_path = input_path
        self.output_path = output_path
        self.input_files = []
        self.output_files = []
        self.extensions = ['.ttl', '.nt', '.rdf', '.owl', '.csv', '.json']

    def build_graph(self, input_file=''):
        graph = Graph()
        graph.parse(input_file, format=get_format(value=input_file))
        return graph

    def accept_extension(self, file=''):
        for ext in self.extensions:
            if file.endswith(ext):
                return True
        return False

    def build_list_files(self):
        """
            building the list of input and output files
        """
        output = []
        for current_path, folders, files in os.walk(self.input_path):
            for file in files:
                if self.accept_extension(file=file):
                    tmp_current_path = os.path.join(current_path, file)
                    output.append(tmp_current_path)
        return output
