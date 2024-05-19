import tensorflow as tf
import keras
from keras import Model
from keras import Model, layers, optimizers, losses, metrics

from config import Config
from utils.data_loader import DataLoader
from models.layers.split import Split
from models.layers.normalization import MinMaxNormalization

class DeepMF(Model):
    
    def __init__(
        self,
        latent_dim:int = 5, 
        num_users:int = Config.Vars.NUM_USERS,
        num_items:int = Config.Vars.NUM_ITEMS,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.num_users = num_users
        self.num_items = num_items
        self.split = Split(name='split')
        self.rating = layers.Identity(name='rating')
        self.user_embedding = layers.Embedding(self.num_users + 1, self.latent_dim, name='user_embedding')
        self.user_flatten = layers.Flatten(name='user_flatten')
        self.item_embedding = layers.Embedding(self.num_items + 1, self.latent_dim, name='item_embedding')
        self.item_flatten = layers.Flatten(name='item_flatten')
        self.dot = layers.Dot(axes=1, name='dot')
        self.concatenate1 = layers.Concatenate(axis=1, name='concatenate1')
        self.norm = MinMaxNormalization(min_val=0, max_val=5, name='normalize')
        self.reshape = layers.Reshape(target_shape=(-1,), name='reshape')
        self.concatenate2 = layers.Concatenate(axis=1, name='concatenate2')
    
    def call(self, inputs):
        user_id, item_id, ratings = self.split(inputs)
        user_emb = self.user_embedding(user_id)
        user_emb = self.user_flatten(user_emb)
        item_emb = self.item_embedding(item_id)
        item_emb = self.item_flatten(item_emb)
        rating_vec = self.dot([user_emb, item_emb])
        embeddings = self.concatenate1([user_emb, item_emb])
        ratings = self.rating(ratings)
        norm_ratings = self.norm(ratings)
        norm_ratings = self.reshape(norm_ratings)
        embedded_data = self.concatenate2([embeddings, norm_ratings])
        return rating_vec, embedded_data
    
    def predict(self, x):
        _, embedded_data = self.call(x)
        return embedded_data
    
    def train_step(self, data):
        inputs, output = data
        
        with tf.GradientTape() as tape:
            rating_vec, _ = self.call(inputs)
            loss = self.compute_loss(y=output, y_pred=rating_vec)
            
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        return {
            'mse_loss': loss,
            **self.compute_metrics(x=inputs[0], y=output, y_pred=rating_vec)
        }
    
    def summary(self):
        model = self.build_graph()
        return model.summary()
    
    def build_graph(self):
        x = (layers.Input(shape=(3,)))
        return Model(inputs=x, outputs=self.call(x))

if __name__ == '__main__':
    
    x_train, x_test, y_train, y_test = DataLoader().load_numpy()
    
    deepmf = DeepMF()
    deepmf.compile(
        optimizer=optimizers.Adam(),
        loss=losses.MeanSquaredError(),
        metrics=[metrics.R2Score()]
    )
    
    deepmf.fit(
        [x_train[:, 0], x_train[:, 1], x_train[:, 2]],
        y_train,
        validation_data=([x_test[:, 0], x_test[:, 1], x_test[:, 2]], y_test),
        epochs=10,
        batch_size=32,
        validation_freq=3
    )
    deepmf.summary()
    keras.utils.plot_model(
        deepmf.build_graph(),
        show_shapes=True,
        dpi=70,
        to_file='src/models/images/deepmf.png'
    )