import csv
import time
from typing import List, Any, Dict, Union
import matplotlib.pyplot as plt

from torch import no_grad
from torch.optim.adam import Adam
from torch import save, Tensor
from torch import max as torch_max
from torch.nn.modules.loss import MSELoss

from quantorch.src.pre_main import ModelsCreator

class Trainer:
    def __init__(self,
                 models_complete: ModelsCreator,
                 train_loader: Tensor,
                 validation_loader: Tensor,
                 loss_fn: MSELoss,
                 epochs: int,
                 learning_rate: float,
                 local: bool):
        self.models_complete = models_complete
        self.models = models_complete.models
        self.epochs = epochs
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.local = local

        self.train_metrics : List[List[List[Any]]] = []
        self.validation_metrics : List[List[List[Any]]] = []
        self.models_state_dicts : List[List[Dict[str, Tensor]]] = []
        self.validation_accuracies : List[List[Any]]= []
        self.best_models_indices : List[int] = [i
                                            for i in range(len(self.models))]
        self.best_models_state_dicts : List[Dict[str, Tensor]] = [
            self.models[0].state_dict() for _ in range(len(self.models))]
        
        self.epoch_losses : List[List[Tensor]] = [
            [] for _ in range(len(self.models))]
        self.epoch_accuracies : List[List[Tensor]] = [
            [] for _ in range(len(self.models))]    

        (self.path_metrics_csv, self.paths_best_models,
                       self.paths_last_models) = self.create_paths_for_saving()
        self.optimizers = self.create_optimizers()
        self.train()
        self.plot_results()

    def create_paths_for_saving(self) -> tuple:
        # Generate folder name
        folder_name : str
        feature_map_name = self.models_complete.feature_map_name
        feature_map_depth = str(self.models_complete.feature_map_depth)
        ansatz_name = self.models_complete.ansatz_name
        ansatz_depth = str(self.models_complete.ansatz_depth)
        if self.local == True:
            folder_name = r"C:\Users\giuseppe.dambruoso\OneDrive - LUTECH SPA\Desktop\Progetto\Risultati\\" + feature_map_depth + feature_map_name + ansatz_depth + ansatz_name
        else :
            folder_name = "Results/" + feature_map_depth + feature_map_name + ansatz_depth + ansatz_name
        # Generate paths for csv, best models and last models
        path_metrics_csv : str
        paths_best_models : List[str]
        paths_last_models : List[str]
        if self.local == True:
            path_metrics_csv = folder_name + r"\MetricsCSV"
            paths_best_models = [folder_name + r"\BestModel" + str(i) 
                                for i in range(len(self.models))]
            paths_last_models = [folder_name + r"\LastModel" + str(i)
                                for i in range(len(self.models))]
        elif self.local == False :
            path_metrics_csv = folder_name + "/MetricsCSV"
            paths_best_models = [folder_name + "/BestModel" + str(i) 
                                for i in range(len(self.models))]
            paths_last_models = [folder_name + "/LastModel" + str(i)
                                for i in range(len(self.models))]
            print(path_metrics_csv)
        return path_metrics_csv, paths_best_models, paths_last_models
    
    def create_optimizers(self) -> List[Adam]:
        optimizers = []
        for model in self.models:
            optimizer = Adam(model.parameters(), lr=self.learning_rate)
            optimizers.append(optimizer)
        return optimizers
    
    def train_ith_model(self,
                        i: int,
                        inputs: Tensor,
                        labels: Tensor,
                        true_labels: Tensor):
        
        model = self.models[i]
        optimizer = self.optimizers[i]

        start_model_time = time.time() # Start recording time
        
        optimizer.zero_grad() # Clear previous gradients
        model_output = model(inputs) # Output from the given model

        # Compute loss and accuracy
        model_loss = self.loss_fn(model_output, labels.float())
        _, predicted_labels = torch_max(model_output, 1)
        correct_predictions = (predicted_labels == true_labels).sum()   
        model_accuracy = correct_predictions / inputs.size(0)

        # Perform backpropagation
        model_loss.backward()
        optimizer.step()

        # Compute the time spent
        end_model_time = time.time()
        model_time = end_model_time - start_model_time

        # Add metrics to lists
        self.epoch_losses[i].append(model_loss)
        self.epoch_accuracies[i].append(model_accuracy)
        
        # Print time, loss and accuracy
        print('{:<8}MODEL {:d}    time: {:.2f}s, loss: {:.4f}, accuracy: {:.2f}%'.format(
            "", i, model_time, model_loss, model_accuracy*100))

    def train_one_batch(self,
                        batch_index: int,
                        inputs: Tensor,
                        labels: Tensor) -> tuple:
        # Print batch index
        print('{:<8}batch index: {:d}'.format("", batch_index+1))
        

        start_batch_time = time.time()

        # Compute true labels
        _, true_labels = torch_max(labels, 1)

        for i in range(len(self.models)): # For every model
            self.train_ith_model(i, inputs, labels, true_labels)

        # Compute batch time
        end_batch_time = time.time()
        batch_time = end_batch_time - start_batch_time
        print('{:<8}batch time: {:.2f}s'.format("", batch_time))
        print()
        
        return (3, 3)

    def compute_avg_epoch_metrics(self) -> List:
        # Compute average loss and accuracy of a given epoch for each model
        avg_epoch_metrics : List[List[Any]] = [
            [] for _ in range(len(self.models))]
        for i in range(len(self.models)):
            model_avg_loss = sum(self.epoch_losses[i]) / len(
                self.epoch_losses[i])
            model_avg_accuracy = sum(self.epoch_accuracies[i]) / len(
                self.epoch_accuracies[i])
            avg_epoch_metrics[i].append(model_avg_loss)
            avg_epoch_metrics[i].append(model_avg_accuracy)
        
        print('{:<8}Average results'.format(""))
        for i in range(len(avg_epoch_metrics)):
            print('{:<8}MODEL {:d}    loss: {:.4f}, accuracy: {:.2f}%'.format(
                "", i, avg_epoch_metrics[i][0], avg_epoch_metrics[i][1]*100))
        
        self.epoch_losses = [[] for _ in range(len(self.models))]
        self.epoch_accuracies = [[] for _ in range(len(self.models))]  
        return avg_epoch_metrics   

    def train_one_epoch(self) -> List:
        """Performs one training epoch."""   
        start_epoch_time = time.time()

        # Set the models to training mode
        for model in self.models:
            model.train()

        for batch_index, (inputs, labels) in enumerate(self.train_loader):
            (a, b) = self.train_one_batch(batch_index, inputs, labels)

        # Compute average loss and average accuracy for each model
        avg_epoch_metrics = self.compute_avg_epoch_metrics()
                    
        # Compute epoch time
        end_epoch_time = time.time()
        epoch_time = (end_epoch_time - start_epoch_time) / 60

        # Print results
        print('{:<8}Epoch train time: {:.2f}min'.format("", epoch_time))

        # Record results
        self.train_metrics.append(avg_epoch_metrics)
        self.models_state_dicts.append([model.state_dict() for
                                            model in self.models])
        return avg_epoch_metrics

    def evaluate_one_epoch(self) -> List:
        """Performs evaluation of the model."""
        # Initialize losses and correct predictions
        total_losses = [0. for _ in range(len(self.models))]
        correct_predictions = [0. for _ in range(len(self.models))]
        start_time = time.time()

        # Set the models to evaluation mode
        for model in self.models:
            model.eval()

        with no_grad():
            for batch_index, (inputs, labels) in enumerate(
                self.validation_loader):
                print('{:<8}batch index: {:d}'.format("", batch_index+1))
                start_batch_time = time.time()
                _, true_labels = torch_max(labels, 1)
                for i in range(len(self.models)):
                    model = self.models[i]
                    output = model(inputs)
                    loss = self.loss_fn(output.float(), labels.float()).item()
                    total_losses[i] += loss
                    _, predicted_labels = torch_max(output, 1)
                    correct_predictions[i] += (predicted_labels == 
                                               true_labels).sum().item()
                end_batch_time = time.time()
                batch_time = end_batch_time - start_batch_time
                print('{:<8}batch time: {:.2f}s'.format("", batch_time))
                print()

        # Compute loss and accuracy on the validation set
        results = []
        for i in range(len(total_losses)):
            avg_loss = total_losses[i] / len(self.validation_loader)
            accuracy = correct_predictions[i] / (len(self.validation_loader) * 
                                                inputs.shape[0])
            results.append([avg_loss, accuracy])
        
        # Compute validation time
        end_time = time.time()
        validation_time = (end_time - start_time)/60

        # Print results
        print('{:<8}Epoch validation time: {:.2f}min'.format("",
                                                             validation_time))
        for i in range(len(results)):
            print('{:<8}MODEL {:d}    loss: {:.4f}, accuracy: {:.2f}%'.format(
                "", i, results[i][0],
                results[i][1]*100))
            
        # Record results
        self.validation_metrics.append(results)
        return results
    
    def save_epoch_models(self, results_one_val: List):
        # Get the best models
        self.validation_accuracies.append([k[1] for k in results_one_val])
        model_validation_accuracies = list(map(list, 
                                    zip(*self.validation_accuracies)))
        for i in range(len(self.models)):
            # Get the best accuracy for a given model
            best_accuracy = max(model_validation_accuracies[i])
            # Get the epoch corresponding to the best accuracy
            best_model_idx = model_validation_accuracies[i].index(
                best_accuracy)
            # Record the model state dict of that epoch
            self.best_models_state_dicts[i] = self.models_state_dicts[
                best_model_idx][i]
            # Create a list with best model indices
            self.best_models_indices[i] = best_model_idx

        for i in range(len(self.models)):
            # Save the last models
            last_model = self.models[i]
            path_last_model = self.paths_last_models[i]
            save(last_model.state_dict(), path_last_model)

            # Save the best models
            path_best_model = self.paths_best_models[i]
            save(self.best_models_state_dicts[i], path_best_model)

    def train(self):
        open(self.path_metrics_csv, 'w').close()
        with open(self.path_metrics_csv, mode='a', newline='') as file:
            writer = csv.writer(file)
            for epoch_number in range(self.epochs):
                print('EPOCH {}'.format(epoch_number+1))
                # Perform one training epoch
                print('{:<4}TRAIN'.format(""))
                results_one_train = self.train_one_epoch()
                    
                print()

                # Perform validation
                print('{:<4}VALIDATION'.format(""))
                results_one_val = self.evaluate_one_epoch()
                    
                print()
                print()

                self.save_epoch_models(results_one_val)
            
                 # Update csv file
                row = sum(self.train_metrics[-1] + self.validation_metrics[-1],
                           [])
                writer.writerow(row)
  
    def plot_results(self):
        # plt.figure(figsize=(10, 12))  # Adjusting figure size for four subplots

        # # Plotting the first subplot (train losses)
        # plt.subplot(4, 1, 1)
        # for i in range(len(self.models)):
        #     plt.plot([t[i][0].detach().numpy() for t in self.train_metrics],
        #             label='Train Loss with model {:d}'.format(i))
        # plt.xlabel('Epoch')
        # plt.ylabel('Loss')
        # plt.title('Train Loss')
        # plt.legend()
        # plt.grid(True)

        # # Plotting the second subplot (validation losses)
        # plt.subplot(4, 1, 2)
        # for i in range(len(self.models)):
        #     plt.plot([t[i][0] for t in self.validation_metrics],
        #             label='Validation Loss with model {:d}'.format(i))
        # plt.xlabel('Epoch')
        # plt.ylabel('Loss')
        # plt.title('Validation Loss')
        # plt.legend()
        # plt.grid(True)

        # # Plotting the third subplot (train accuracies)
        # plt.subplot(4, 1, 3)
        # for i in range(len(self.models)):
        #     plt.plot([t[i][1].detach().numpy() for t in self.train_metrics],
        #             label='Train Accuracy with model {:d}'.format(i))
        # plt.xlabel('Epoch')
        # plt.ylabel('Accuracy')
        # plt.title('Train Accuracy')
        # plt.legend()
        # plt.grid(True)

        # # Plotting the fourth subplot (validation accuracies)
        # plt.subplot(4, 1, 4)
        # for i in range(len(self.models)):
        #     plt.plot([t[i][1] for t in self.validation_metrics],
        #             label='Validation Accuracy with model {:d}'.format(i))
        # plt.xlabel('Epoch')
        # plt.ylabel('Accuracy')
        # plt.title('Validation Accuracy')
        # plt.legend()
        # plt.grid(True)

        # plt.tight_layout()
        # plt.show()

        print('best models obtained at epochs {}, respectively'.format(
            self.best_models_indices))