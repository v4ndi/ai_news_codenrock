import torch 

class Args():
    def __init__(self):
        self.max_length = 128
        self.lr = 1e-5
        self.num_epochs = 5
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 8
        self.model_name = "ai-forever/ruRoberta-large"
        self.random_state = 101
        self.num_classes = 29