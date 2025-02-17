from pris7 import train
from pris7 import test

print("Train module:", train)
print("Test module:", test)

def cv(task_name):
    if task_name == "Train Mnist Dataset":
        train.train_mnist()
    elif task_name == "Test Mnist Dataset":
        test.test_mnist()
    else:
        raise ValueError(f"Unknown task: {task_name}")




