import trainmodel
import yaml
import argparse
def start_train(output_path,data_path,params_file):
    path_model_1 = output_path + '/model_mel_notmel.pkl'
    path_model_2 = output_path + '/model_mel_nv.pkl'
    with open(params_file, 'r') as f:
        params = yaml.safe_load(f)

    trainmodel.TrainModel(params['model_mel_notmel']["label_map"],params['model_mel_notmel']["arch"], path_model_1,params['model_mel_notmel']['train']["epochs"], params['model_mel_notmel']['train']["batch_size"],data_path).train_model()
    trainmodel.TrainModel(params['model_mel_nv']["label_map"],params['model_mel_nv']["arch"], path_model_2, params['model_mel_nv']['train']["epochs"], params['model_mel_nv']['train']["batch_size"],data_path).train_model()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an image classification model with Hugging Face Transformers")
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data directory')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the trained model')
    parser.add_argument('--params_file', type=str, required=True, help='Path to the parameters YAML file')
    args = parser.parse_args()
    start_train(args.output_path, args.data_path, args.params_file)