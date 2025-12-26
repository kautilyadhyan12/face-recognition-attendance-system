
import tensorflow as tf
import os

def setup_gpu():
    # Clear any previous TensorFlow session
    tf.keras.backend.clear_session()
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Set memory growth to prevent allocation of all GPU memory
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
           
            tf.config.experimental.set_memory_growth(gpus[0], True)
            
           
           
            
            print(f" GPU configured: {gpus[0].name}")
            return True
        except RuntimeError as e:
            print(f" GPU configuration failed: {e}")
            return False
    else:
        print(" No GPU found, using CPU")
        return False

# Force CPU if GPU causes issues
def force_cpu():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    print("  Forcing CPU mode")
