from django.apps import AppConfig
import tensorflow as tf

class PredictConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'predict'

    mpath = "models"

    model = tf.keras.models.load_model(mpath)
