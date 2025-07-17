from utils.preprocess import load_images
from models.decision_tree import train_tree

images, labels = load_images('data/', size=(64, 64))
model = train_tree(images, labels)
