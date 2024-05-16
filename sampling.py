import time
import random
import validators
import os
import pandas as pd


class Sampling:

    def __init__(self, suffix='', source_subjects=[], target_subjects=[], entities={}, random_size=0.2):
        self.suffix = suffix
        self.output_path = './outputs/' + self.suffix + '/'
        if os.path.exists(self.output_path) is False:
            os.makedirs(self.output_path)
        self.source_subjects = source_subjects
        self.target_subjects = target_subjects
        self.entities = entities
        self.random_size = random_size
        self.labels = ['different', 'same']
        self._labels = {
            'different': 0,
            'same': 1
        }
        self.data_template = """{} : {} \n {} : {} \n"""
        self.content_template = """{} and {} are {}"""

    def refresh_object(self, value=''):
        if not validators.url(value):
            return "'" + str(value) + "'"
        else:
            return "<" + str(value) + ">"

    def from_entity_to_string(self, entity=[]):
        output = []
        for p, o in entity:
            output.append(self.refresh_object(value=p) +
                          ' ' + self.refresh_object(value=o))
        output = '; '.join(output)
        return output + ' . '

    def save_to_json(self, file_name='', data=[]):
        with open(file_name, 'w') as f:
            f.write(str(data))
        return True

    def save_to_csv(self, file_name='', data=None):
        df = pd.DataFrame(data)
        df.to_csv(file_name, index=False)
        return True

    def save_to_examples(self, label='', data=[]):
        file_name = self.output_path + label + '.csv'
        output = {
            'input': [],
            'label': [],
            'label_text': [],
            'content': []
        }
        i = 0
        for s, s_e, t, t_e, l in data:
            _s = self.refresh_object(value=str(s))
            _t = self.refresh_object(value=str(t))
            _se = _s + ' ' + self.from_entity_to_string(entity=s_e)
            _te = _t + ' ' + self.from_entity_to_string(entity=t_e)
            input = self.data_template.format(_s, _se, _t, _te)
            output['input'].append(input)
            output['label'].append(l)
            output['label_text'].append(self.labels[l])
            output['content'].append(
                self.content_template.format(_s, _t, self.labels[l]))
        self.save_to_csv(file_name=file_name, data=output)
        return True

    def learning(self):
        positive_training = []
        negative_training = []
        portion = int(len(self.source_subjects) * self.random_size)
        source_subjects = self.source_subjects[portion:]
        target_subjects = self.target_subjects[portion:]
        for s, t in zip(source_subjects, target_subjects):
            if s in self.entities and t in self.entities:
                positive_training.append(
                    (s, self.entities[s], t, self.entities[t], 1))
        for s in source_subjects:
            t = random.choice(target_subjects)
            while source_subjects.index(s) == target_subjects.index(t):
                t = random.choice(target_subjects)
            if s in self.entities and t in self.entities:
                negative_training.append(
                    (s, self.entities[s], t, self.entities[t], 0))
        self.save_to_examples(
            label='train', data=positive_training+negative_training)
        return True

    def testing(self):
        output = []
        portion = int(len(self.source_subjects) * self.random_size)
        source_subjects = self.source_subjects[0:portion]
        target_subjects = self.target_subjects[0:portion]
        for s, t in zip(source_subjects, target_subjects):
            if s in self.entities and t in self.entities:
                output.append((s, self.entities[s], t, self.entities[t], 1))
        for s in source_subjects:
            t = random.choice(target_subjects)
            while source_subjects.index(s) == target_subjects.index(t):
                t = random.choice(target_subjects)
            if s in self.entities and t in self.entities:
                output.append(
                    (s, self.entities[s], t, self.entities[t], 0))
        self.save_to_examples(label='test', data=output)
        return True

    def run(self):
        self.testing()
        self.learning()
        print('Sampling')
