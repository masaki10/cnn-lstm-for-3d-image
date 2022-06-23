import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--input_shape")
parser.add_argument("--epoch", default=500, type=int)
parser.add_argument("--batch", default=2, type=int)
parser.add_argument("--frame", default=3, type=int)
parser.add_argument("--train_data_csv_path")
parser.add_argument("--val_data_csv_path")
parser.add_argument("--test_data_csv_path")
parser.add_argument("--log_path", default="./checkpoint/log.csv")
parser.add_argument("--early_stop_monitor", default="val_loss")
parser.add_argument("--early_stop_patience", default=6, type=int)
parser.add_argument("--checkpoint_dir", default="./checkpoint/")
parser.add_argument("--checkpoint_monitor", default="val_loss")
parser.add_argument("--checkpoint_period", default=1, type=int)
parser.add_argument("--cnn_model")
parser.add_argument("--loss_function", default="mean_absolute_error")
parser.add_argument("--conv_lstm_filter_num", default=128, type=int)
parser.add_argument("--conv_lstm_filter_size", default=3, type=int)
parser.add_argument("--learning_rate", type=float)
parser.add_argument("--from_checkpoint", default=False, type=bool)
parser.add_argument("--from_checkpoint_path")
parser.add_argument("--model_path_for_predict")
parser.add_argument("--csv_path_for_predict", default="./checkpoint/predict.csv")



args = parser.parse_args()

print(args.batch)
print(args.epoch)
print(type(args.epoch))
print(args.checkpoint_period)
print(args.cnn_model)
print(type(args.cnn_model))
print(args.from_checkpoint)
print(type(args.from_checkpoint))
if args.from_checkpoint:
    print("true!")

print(type(True))