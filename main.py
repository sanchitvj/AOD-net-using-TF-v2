

from path_dataloader import data_path, dataloader
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.optimizers import Adam
from aod_net import dehaze_net
from train import train_model


epochs = 10
batch_size = 64

train_data, val_data = data_path(orig_img_path = '/content/clear_images', hazy_img_path = '/content/haze')
train, val = dataloader(train_data, val_data, batch_size)

optimizer = Adam(learning_rate = 1e-4)
net = dehaze_net()

train_loss_tracker = tf.keras.metrics.MeanSquaredError(name = "train loss")
val_loss_tracker = tf.keras.metrics.MeanSquaredError(name = "val loss")


train_model(epochs, train, val, net, train_loss_tracker, val_loss_tracker, optimizer)
