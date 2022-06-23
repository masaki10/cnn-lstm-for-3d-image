from cnn_lstm import CNNLSTM
import config as cf

def train():
    cnn_lstm = CNNLSTM()
    cnn_lstm.print_summary()
    cnn_lstm.train()

def predict():
    cnn_lstm = CNNLSTM()
    cnn_lstm.print_summary()
    cnn_lstm.predict()

if __name__ == "__main__":
    print("hello world")
    train()
    # predict()