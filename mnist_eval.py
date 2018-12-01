import time
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference
import mnist_train

# 每10秒加载一次模型
EVAL_INTERVAL_SECS = 10

def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [mnist.validation.num_examples, mnist_inference.IMAGE_SIZE,
                                        mnist_inference.IMAGE_SIZE, mnist_inference.NUM_CHANNELS],
                           name='x-input')
        y_ = tf.placeholder(tf.float32,[None, mnist_inference.OUT_NODE], name='y-input')

        xs = mnist.validation.images
        reshaped_xs = np.reshape(xs, (mnist.validation.num_examples, mnist_inference.IMAGE_SIZE,
                                      mnist_inference.IMAGE_SIZE, mnist_inference.NUM_CHANNELS))
        validate_feed = {x:reshaped_xs, y_:mnist.validation.labels}
        # 测试的时候不用计算正则化损失函数
        y = mnist_inference.inference(x, 0, None)

        correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    print(ckpt.model_checkpoint_path)
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('\\')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                    print("After %s training step(s), validation "
                          "accuracy = %g" %(global_step, accuracy_score))
                else:
                    print ('No checkpoint file found')
                    return
            time.sleep(EVAL_INTERVAL_SECS)
def main(argv=None):
    mnist = input_data.read_data_sets(r"mnist_data", one_hot=True)
    evaluate(mnist)
if __name__ == '__main__':
    tf.app.run()

