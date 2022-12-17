# Train a double CNN for RGB and depth images.
# Use a ImageDataModule to load the images located in the specified train and val folders
# The resulting model will be stored in a file which name looks like this : model-epoch=01-val_loss=0.62.ckpt
# and which is located in '<ckpt_folder>/model/<model name>' like 'model/resnet50'
# To view the logs : tensorboard --logdir=runs

# --- MAIN ----
if __name__ == '__main__':
    from raiv_libraries.rgb_and_depth_cnn import RgbAndDepthCnn
    from raiv_libraries.image_data_module import ImageDataModule, RgbAndDepthSubset
    from raiv_libraries.rgb_and_depth_image_dataset import RgbAndDepthImageDataset
    from raiv_libraries.image_model import ImageModel
    import argparse
    import time

    parser = argparse.ArgumentParser(description='Train a Cnn with RGB and depth images from specified images folder. View results with : tensorboard --logdir=runs')
    parser.add_argument('images_rgb_folder', type=str, help='RGB images folder with fail and success sub-folders')
    parser.add_argument('images_depth_folder', type=str, help='Depth images folder with fail and success sub-folders')
    parser.add_argument('ckpt_folder', type=str, help='folder path where to stock the model.CKPT file generated')
    parser.add_argument('-c', '--courbe_path', default=None, type=str, help='Optionnal path folder .txt where the informations of the model will be stocked for courbes_CNN.py')
    parser.add_argument('-s', '--suffix_name', default='', type=str, help='Optionnal suffix to add to the model name')
    parser.add_argument('-e', '--epochs', default=15, type=int, help='Optionnal number of epochs')
    parser.add_argument('-d', '--dataset_size', default=None, type=int, help='Optionnal number of images for the dataset size')
    args = parser.parse_args()

    # Build the model
    model_name = 'resnet18'
    model = RgbAndDepthCnn(backbone=model_name, courbe_folder=args.courbe_path)
    # Build the dataset and the DataModule
    dataset = RgbAndDepthImageDataset(args.images_rgb_folder, args.images_depth_folder)
    data_module = ImageDataModule(dataset, RgbAndDepthSubset, dataset_size=args.dataset_size)
    # Now, we can build he ImageModel
    rgb_and_depth_image_model = ImageModel(model=model, data_module=data_module, model_name=model_name, ckpt_dir=args.ckpt_folder, num_epochs=args.epochs, suffix=args.suffix_name)
    start = time.time()
    rgb_and_depth_image_model.call_trainer()  # Train model
    end = time.time()
    print(f"Elapsed time = {end-start:.2f} seconds")
    print('End of model training')