from mobilenetv2 import MobileNetV2
import torch


net = MobileNetV2(num_classes=7)

my_dict = net.state_dict()

model_dict = torch.load("mobilenetv2_1.0-0c6065bc.pth", map_location = torch.device('cpu'))


my_keys = list(my_dict.keys())
model_keys =  list(model_dict.keys())

assert( len(my_keys) == len(model_keys))


for i in range(1, len(my_keys) - 2):

    if model_keys[i] == my_keys[i]:
        my_dict[model_keys[i]] = model_dict[model_keys[i]]



net.load_state_dict(my_dict)


torch.save(net, "my_weights.pth")