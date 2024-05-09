import torch
import resnet10

MODEL_PATH = "../ckpt/checkpoint_400.pkl"



if __name__ == "__main__":
    backbone = resnet10.ResNet10()
    state_dict = torch.load(MODEL_PATH)["model"]
    backbone.load_state_dict(state_dict)