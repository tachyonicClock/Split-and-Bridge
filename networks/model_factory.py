import torch
class ModelFactory():
    def __init__(self):
        pass

    @staticmethod
    def get_model(dataset, num_class):
        
        if dataset in ['CIFAR100', 'CIFAR10', 'FMNIST', 'CORE50', "LCORE50"]:
            
            import networks.MyNetwork_split as res
            return res.network('CIFAR',32,num_class)

        if dataset == 'TinyImagenet':

            import networks.MyNetwork_split as res
            return res.network('TinyImagenet', 64, num_class)

