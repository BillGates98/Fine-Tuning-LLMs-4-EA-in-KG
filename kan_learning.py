from kan import KAN
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


class KanLearning:

    def __init__(self, input_size=-1, output_size=-1, train_input=None, train_label=None, test_input=None, test_label=None):
        self.device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.dataset = {}
        self.dataset['train_input'] = torch.from_numpy(
            train_input).type(torch.float32)
        self.dataset['test_input'] = torch.from_numpy(
            test_input).type(torch.float32)
        self.dataset['train_label'] = torch.from_numpy(
            train_label[:, None]).type(torch.float32)
        self.dataset['test_label'] = torch.from_numpy(
            test_label[:, None]).type(torch.float32)
        self.input_size = input_size
        self.output_size = output_size

    def data_to_device(self):
        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
            self.dataset['train_input'], self.dataset['train_label']), batch_size=16, shuffle=False)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
            self.dataset['test_input'], self.dataset['test_label']), batch_size=16, shuffle=False)

        train_inputs = torch.empty(0, self.input_size, device=self.device)
        train_labels = torch.empty(0, dtype=torch.long, device=self.device)
        test_inputs = torch.empty(0, self.input_size, device=self.device)
        test_labels = torch.empty(0, dtype=torch.long, device=self.device)

        for data, labels in train_loader:
            train_inputs = torch.cat(
                (train_inputs, data.to(self.device)), dim=0)
            train_labels = torch.cat(
                (train_labels, labels.to(self.device)), dim=0)

        for data, labels in test_loader:
            test_inputs = torch.cat((test_inputs, data.to(self.device)), dim=0)
            test_labels = torch.cat(
                (test_labels, labels.to(self.device)), dim=0)

        _dataset = {}
        _dataset['train_input'] = train_inputs
        _dataset['test_input'] = test_inputs
        _dataset['train_label'] = train_labels
        _dataset['test_label'] = test_labels

        return _dataset

    def train_process(self, dataset=None):
        # 20, 4
        model = KAN(width=[self.input_size, 2*self.input_size + 1, self.output_size],
                    grid=5, k=self.input_size+1, device=self.device)  # k=3
        # model(dataset['train_input'])
        # model.plot(beta=100, scale=1, in_vars=['in'+str(i+1)
        #            for i in range(self.input_size)], out_vars=['out'])

        def train_acc():
            return torch.mean((torch.round(model(dataset['train_input'])) == dataset['train_label']).float())

        def test_acc():
            output = torch.mean(
                (torch.round(model(dataset['test_input'])) == dataset['test_label']).float())
            predictions = model(
                dataset['test_input']).view(-1).detach().cpu().numpy()
            predictions = [1 if v >= 0.5 else 0 for v in predictions]
            precision, recall, f_score, _ = precision_recall_fscore_support(
                dataset['test_label'].view(-1).cpu().numpy(), predictions, average='micro')
            test_accuracy = output.item()
            metrics = f"Precision: {precision}\nRecall: {recall}\nF1: {f_score}\nTest Accuracy: {test_accuracy} \n \n"
            print(metrics)
            return output

        results = model.train(dataset, opt="LBFGS", steps=20, metrics=(
            train_acc, test_acc), device=self.device)

        for i in range(len(results['train_acc'])):
            print(
                f" ITER : {i+1} >> Train acc: {results['train_acc'][i]} # Test acc: {results['test_acc'][i]} ")

        print("====================================")
        # model.plot(beta=100, scale=1, in_vars=[
        #            'in'+str(i+1) for i in range(self.input_size)], out_vars=['out'])
        return model

    def test_process(self, model=None, data=None):
        # print(data['test_label'].view(-1).cpu().numpy())
        predictions = model(data['test_input']).view(-1).detach().cpu().numpy()
        predictions = [1 if v >= 0.5 else 0 for v in predictions]
        precision, recall, fscore, _ = precision_recall_fscore_support(
            data['test_label'].view(-1).cpu().numpy(), predictions, average='macro')
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1': fscore
        }
        return metrics

    def run(self):
        # self.plot_points()
        print("====================================")
        dataset = self.data_to_device()
        print("Train data shape: {}".format(dataset['train_input'].shape))
        print("Train target shape: {}".format(dataset['train_label'].shape))
        print("Test data shape: {}".format(dataset['test_input'].shape))
        print("Test target shape: {}".format(dataset['test_label'].shape))
        print("====================================")
        model = self.train_process(dataset=dataset)
        metrics = self.test_process(model=model, data=dataset)
        return metrics
