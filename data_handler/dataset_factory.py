import data_handler.dataset as data


class DatasetFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_dataset(name):
        if name == 'CIFAR100':
            return data.CIFAR100()
        elif name == "TinyImagenet":
            return data.TinyImagenet()
        elif name == "CIFAR10":
            return data.CIFAR10()
        elif name == "FMNIST":
            return data.FMNIST()
        elif name == "CORE50":
            return data.CORe50_NC(mini=True)
