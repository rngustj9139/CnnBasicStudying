import tensorflow as tf
import tensorflow_datasets as tfds

def load_processing_mnist(train_ratio, train_batch_size, test_batch_size):
    (train_validation_ds, test_ds), ds_info = tfds.load(name="mnist", shuffle_files=True, as_supervised=True, split=['train', 'test'], with_info=True)

    print("---------")
    print("ds info\n", ds_info)
    print("---------")

    n_train_validation = ds_info.splits['train'].num_examples
    n_train = int(n_train_validation * train_ratio)
    n_validation = n_train_validation - n_train

    train_ds = train_validation_ds.take(n_train)
    remaining_ds = train_validation_ds.skip(n_train)
    validation_ds = remaining_ds.take(n_validation)

    def normalization(images, labels):
        images = tf.cast(images, tf.float32) / 255.
        return [images, labels]

    train_ds = train_ds.shuffle(1000).map(normalization).batch(train_batch_size)
    validation_ds = validation_ds.map(normalization).batch(test_batch_size)
    test_ds = test_ds.map(normalization).batch(test_batch_size)

    return train_ds, validation_ds, test_ds

def load_processing_cifar10(train_ratio, train_batch_size, test_batch_size):
    (train_validation_ds, test_ds), ds_info = tfds.load(name="cifar10", shuffle_files=True, as_supervised=True, split=['train', 'test'], with_info=True)

    n_train_validation = ds_info.splits['train'].num_examples
    n_train = int(n_train_validation * train_ratio)
    n_validation = n_train_validation - n_train

    train_ds = train_validation_ds.take(n_train)
    remaining_ds = train_validation_ds.skip(n_train)
    validation_ds = remaining_ds.take(n_validation)

    def normalization(images, labels):
        images = tf.cast(images, tf.float32) / 255.
        return [images, labels]

    train_ds = train_ds.shuffle(1000).map(normalization).batch(train_batch_size)
    validation_ds = validation_ds.map(normalization).batch(test_batch_size)
    test_ds = test_ds.map(normalization).batch(test_batch_size)

    return train_ds, validation_ds, test_ds
