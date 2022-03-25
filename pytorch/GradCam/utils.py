def read_class_names(class_file_name):
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


def read_imagenet_names(class_file_name):
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.split(" ", 1)[-1].split(",")[0].strip('\n')
    return names


def get_all_layers(model):
    all_layers = []
    all_layers_names = []

    def get_layer(model):
        layers = model.named_modules()
        for item in layers:
            name = item[0]
            module = item[1]
            sublayers = list(module.named_children())
            num = len(sublayers)
            if num != 0:
                continue
            else:
                all_layers.append(module)
                all_layers_names.append(name)

    get_layer(model)
    all_layers = dict(zip(all_layers_names, all_layers))
    return all_layers
