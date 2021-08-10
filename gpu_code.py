from tensorflow.python.client import device_lib
import tensorflow as tf

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return local_device_protos

if __name__ == '__main__':
    #print(get_available_gpus())
    print(tf.__version__)
    print(tf.config.list_physical_devices())