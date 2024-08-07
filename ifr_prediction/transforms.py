import torchvision.transforms as transforms

def get_crnn_transform(img_x,img_y):    
    transform = transforms.Compose([
        transforms.Resize([img_x, img_y]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform

def get_crnn_augmentation_tranform(img_x,img_y):
    base_transform = get_crnn_transform(img_x, img_y)
    return base_transform