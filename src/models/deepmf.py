import tensorflow as tf
import keras
from keras import Model
from keras.layers import Input, Embedding, Flatten, Dot
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from keras.metrics import MeanSquaredError

from config import Config
from utils.data_loader import DataLoader

class DeepMF(Model):
    
    def __init__(self, latent_dim=5, **kwargs):
        super(DeepMF, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.num_users = Config.Vars.NUM_USERS
        self.num_items = Config.Vars.NUM_ITEMS
        
        self.user_embedding = Embedding(self.num_users + 1, self.latent_dim, name='user_embedding')
        self.user_flatten = Flatten(name='user_flatten')
        self.item_embedding = Embedding(self.num_items + 1, self.latent_dim, name='item_embedding')
        self.item_flatten = Flatten(name='item_flatten')
        self.dot = Dot(axes=1, name='dot')
    
    def call(self, inputs):
        user_input, item_input = inputs[0], inputs[1]
        
        user_embedded = self.user_embedding(user_input)
        user_flattened = self.user_flatten(user_embedded)
        item_embedded = self.item_embedding(item_input)
        item_flattened = self.item_flatten(item_embedded)
        
        rating_vec = self.dot([user_flattened, item_flattened])
        return rating_vec
    
    def summary(self):
        x = Input(shape=(2,))
        return Model(inputs=x, outputs=self.call(x)).summary()
    
    def build_graph(self):
        x = Input(shape=(2,))
        return Model(inputs=x, outputs=self.call(x))

if __name__ == '__main__':
    
    deepmf = DeepMF()
    
    deepmf.compile(
        optimizer=Adam(),
        loss=MeanSquaredError(),
        metrics=[keras.metrics.MeanSquaredError()]
    )
    
    x_train, x_test, y_train, y_test = DataLoader().load_numpy()
    
    deepmf.fit(
    [x_train[:, 0], x_train[:, 1]],
    y_train,
    validation_data=([x_test[:, 0], x_test[:, 1]], y_test),
    epochs=10,
    batch_size=32,
    validation_freq=3
    )
    
    deepmf.summary()