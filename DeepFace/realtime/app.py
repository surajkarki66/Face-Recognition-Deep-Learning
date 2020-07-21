from main import verification
from model import build_model
from utils import capture, get_embedding

if __name__ == "__main__":
    model = build_model()
    #capture()
    employees = get_embedding(model)
    verification(deepface=model, employees=employees)

