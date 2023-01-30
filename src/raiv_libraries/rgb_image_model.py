from ray.tune.integration.pytorch_lightning import TuneReportCallback
from raiv_libraries.rgb_cnn import RgbCnn
from raiv_libraries.image_data_module import ImageDataModule, RgbSubset
import torchvision.datasets as datasets
from ray import air, tune
import argparse
import time


# Train a CNN for RGB images.
# Use a ImageDataModule to load the images located in the specified train and val folders
# The resulting model will be stored in a file which name looks like this : model-epoch=01-val_loss=0.62.ckpt
# and which is located in '<ckpt_folder>/model/<model name>' like 'model/resnet50'
# To view the logs : tensorboard --logdir=runs

def train_cnn_tune(config, num_epochs=10):
    print('train_mnist_tune')
    # Build the model
    model_name = 'resnet18'
    model = RgbCnn(config, backbone=model_name, courbe_folder=None)
    # Build the DataModule
    dataset = datasets.ImageFolder(args.images_folder)
    data_module = ImageDataModule(dataset, RgbSubset, dataset_size=args.dataset_size, batch_size=config["batch_size"])
    # Build the trainer
    trainer = model.build_trainer(data_module=data_module, model_name=model_name, ckpt_dir=args.ckpt_folder, num_epochs=num_epochs, suffix=args.suffix_name, dataset_size=args.dataset_size)
    # Now, we can train the model ################################################
    trainer.fit(model=model, datamodule=data_module)

def tune_cnn(num_samples=10, num_epochs=10, gpus_per_trial=0):
    # config = {
    #     "layer_1_size": tune.grid_search([ 128, 256, 512]),
    #     "layer_2_size": tune.grid_search([16, 32, 64]),
    #     #"learning_rate": tune.loguniform(1e-4, 1e-1),
    #     "learning_rate": tune.grid_search([1e-4, 1e-3, 1e-2, 1e-1]),
    #     "batch_size": tune.grid_search([8, 16, 32, 64, 128]),
    # }
    config = {
        "layer_1_size": tune.choice([ 128, 256, 512]),
        "layer_2_size": tune.choice([16, 32, 64]),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        #"learning_rate": tune.choice([1e-3, 1e-2, 1e-1]),
        "batch_size": tune.choice([4, 8, 16, 32]),
    }
    trainable = tune.with_parameters(
        train_cnn_tune, num_epochs=num_epochs
    )
    tuner = tune.Tuner(
        tune.with_resources(trainable, resources={"cpu": 4, "gpu": gpus_per_trial}),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            num_samples=num_samples,
        ),
        run_config=air.RunConfig(
            #log_to_file="std_combined.log",
            local_dir="./ray_tune_results",
            name="tune_rgb_image_model",
        ),
        param_space=config,
    )
    results = tuner.fit()

    print("Best hyperparameters found were: ", results.get_best_result().config)


# --- MAIN ----
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train a Cnn with images from specified images folder. View results with : tensorboard --logdir=runs')
    parser.add_argument('images_folder', type=str, help='images folder with fail and success sub-folders')
    parser.add_argument('ckpt_folder', type=str, help='folder path where to stock the model.CKPT file generated')
    parser.add_argument('-c', '--courbe_path', default=None, type=str, help='Optionnal path folder .txt where the informations of the model will be stocked for courbes_CNN.py')
    parser.add_argument('-s', '--suffix_name', default='', type=str, help='Optionnal suffix to add to the model name')
    parser.add_argument('-e', '--epochs', default=15, type=int, help='Optionnal number of epochs')
    parser.add_argument('-d', '--dataset_size', default=None, type=int, help='Optionnal number of images for the dataset size')
    parser.add_argument('--tune', default=False, action='store_true', help='Tune the hyperparameters')
    parser.add_argument('--no-tune', dest='tune', action='store_false')
    args = parser.parse_args()

    if args.tune:
        print('Hyperparameter tuning.')
        tune_cnn(num_samples=40, num_epochs=args.epochs, gpus_per_trial=0.25)
    else:
        # config = {
        #     "layer_1_size": 256,
        #     "layer_2_size": 16,
        #     "learning_rate": 0.01,
        #     "batch_size": 8
        # }
        config = {  # Config avec acc=1.84375
            "layer_1_size": 128,
            "layer_2_size": 32,
            "learning_rate": 0.00211123,
            "batch_size": 4
        }
        # Build the model
        model_name = 'resnet18'
        model = RgbCnn(config, backbone=model_name, courbe_folder=args.courbe_path)
        # Build the DataModule
        dataset = datasets.ImageFolder(args.images_folder)
        data_module = ImageDataModule(dataset, RgbSubset, dataset_size=args.dataset_size, batch_size=config["batch_size"])
        # Build the trainer
        trainer = model.build_trainer(data_module=data_module, model_name=model_name, ckpt_dir=args.ckpt_folder, num_epochs=args.epochs, suffix=args.suffix_name, dataset_size=args.dataset_size)
        # Now, we can train the model ################################################
        start_fit = time.time()
        trainer.fit(model=model, datamodule=data_module)
        print(f"Training duration = {time.time() - start_fit:.2f} seconds")
        # Test  ################################################
        trainer.test(ckpt_path='best', datamodule=data_module)
        print('End of model training')