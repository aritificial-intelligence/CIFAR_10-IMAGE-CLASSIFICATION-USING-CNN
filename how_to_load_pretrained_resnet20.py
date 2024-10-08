from resnet20_cifar import resnet20


# This resnet20 model is trained with Normalization augmentation for training data:
# transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
# Therefore, you need to test this model also using the same Normalization for your testing data
# As a result, you need to add the above transforms.Normalize(xxxx) in your testing data loader.


model = resnet20()
model_path = "./path_to_your_model/resnet20_cifar10_pretrained.pt"
model.load_state_dict(torch.load(model_path))

