from model import greedy_decode, index_to_label, label_to_index
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
import torch

from model import TransformerMNIST 
from mnist_generator import TiledMNISTDataset 
import os

os.makedirs("predictions", exist_ok=True)

VOCAB_SIZE = 13
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


test_dataset = TiledMNISTDataset(split="test")
checkpoint_path = "model/eternal-grass-2/transformer_epoch7.pth"
model = TransformerMNIST(vocab_size=VOCAB_SIZE).to(device) 
model.load_state_dict(torch.load(checkpoint_path))
model.eval()

for i in range(20):
    test_img, _, test_target = test_dataset[i]
    pred_tokens = greedy_decode(model, test_img.to(device))
    pred_labels = [index_to_label[t] for t in pred_tokens]
    target_labels = [index_to_label[t.item()] for t in test_target if t.item() != label_to_index["<pad>"]]

    plt.imshow(to_pil_image(test_img), cmap="gray")
    plt.axis("off")
    plt.title(f"Pred: {' '.join(pred_labels)}\nTrue: {' '.join(target_labels)}")
    plt.savefig(f"predictions/sample_{i}.png")
    plt.show()
