#coding:utf-8
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import summary_ops_v2
import os


class policy_value_network(object):
    def __init__(self, learning_rate_fn, res_block_nums = 7):
        #         self.ckpt = os.path.join(os.getcwd(), 'models/best_model.ckpt-13999')    # TODO
        self.save_dir = "./models"
        self.is_logging = True

        if tf.io.gfile.exists(self.save_dir):
            #             print('Removing existing model dir: {}'.format(MODEL_DIR))
            #             tf.io.gfile.rmtree(MODEL_DIR)
            pass
        else:
            tf.io.gfile.makedirs(self.save_dir)

        train_dir = os.path.join(self.save_dir, 'summaries', 'train')
        test_dir = os.path.join(self.save_dir, 'summaries', 'eval')

        self.train_summary_writer = summary_ops_v2.create_file_writer(train_dir, flush_millis=10000)
        self.test_summary_writer = summary_ops_v2.create_file_writer(test_dir, flush_millis=10000, name='test')

        # Variables
        self.filters_size = 128    # or 256
        self.prob_size = 2086
        self.digest = None

        self.inputs_ = tf.keras.layers.Input([9, 10, 14], dtype='float32', name='inputs')  # TODO C plain x 2
        self.c_l2 = 0.0001
        self.momentum = 0.9
        self.global_norm = 100

        regularizer = tf.keras.regularizers.l2(self.c_l2)

        self.layer = tf.keras.layers.Conv2D(kernel_size=3, filters=self.filters_size, padding='same', kernel_regularizer=regularizer)(self.inputs_)
        self.layer = tf.keras.layers.BatchNormalization(epsilon=1e-5)(self.layer)
        self.layer = tf.keras.layers.ReLU()(self.layer)

        # residual_block
        with tf.name_scope("residual_block"):
            for _ in range(res_block_nums):
                self.layer = self.residual_block(self.layer)

        # policy_head
        with tf.name_scope("policy_head"):
            self.policy_head = tf.keras.layers.Conv2D(filters=2, kernel_size=1, padding='same', kernel_regularizer=regularizer)(self.layer)
            self.policy_head = tf.keras.layers.BatchNormalization(epsilon=1e-5)(self.policy_head)
            self.policy_head = tf.keras.layers.ReLU()(self.policy_head)

            self.policy_head = tf.keras.layers.Reshape([9 * 10 * 2])(self.policy_head)
            self.policy_head = tf.keras.layers.Dense(self.prob_size, kernel_regularizer=regularizer, name="policy_head_output")(self.policy_head)

        # value_head
        with tf.name_scope("value_head"):
            self.value_head = tf.keras.layers.Conv2D(filters=1, kernel_size=1, padding='same', kernel_regularizer=regularizer)(self.layer)
            self.value_head = tf.keras.layers.BatchNormalization(epsilon=1e-5)(
                self.value_head)
            self.value_head = tf.keras.layers.ReLU()(self.value_head)

            self.value_head = tf.keras.layers.Reshape([9 * 10 * 1])(self.value_head)
            self.value_head = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=regularizer)(self.value_head)
            self.value_head = tf.keras.layers.Dense(1, activation='tanh', kernel_regularizer=regularizer, name="value_head_output")(self.value_head)

        self.model = tf.keras.Model(
            inputs=[self.inputs_],
            outputs=[self.policy_head, self.value_head])

        self.model.summary()

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        #         optimizer = tf.train.AdamOptimizer(self.learning_rate)

        # 优化损失
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate_fn, momentum=self.momentum)

        self.model.compile(optimizer=self.optimizer,
                           loss={'policy_head_output': tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                                 'value_head_output': tf.keras.losses.MeanSquaredError()},
                           loss_weights={'policy_head_output': 1.0, 'value_head_output': 1.0},
                           metrics={'policy_head_output': 'accuracy'})


        self.checkpoint_dir = os.path.join(self.save_dir, 'checkpoints')
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, 'ckpt')
        self.checkpoint = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)

        # Restore variables on creation if a checkpoint exists.
        self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))

    def residual_block(self, in_layer):
        orig = in_layer
        regularizer = tf.keras.regularizers.l2(self.c_l2)
        layer = tf.keras.layers.Conv2D(kernel_size=3, filters=self.filters_size, padding='same', kernel_regularizer=regularizer)(in_layer)
        layer = tf.keras.layers.BatchNormalization(epsilon=1e-5)(layer)
        layer = tf.keras.layers.ReLU()(layer)

        layer = tf.keras.layers.Conv2D(kernel_size=3, filters=self.filters_size, padding='same', kernel_regularizer=regularizer)(layer)
        layer = tf.keras.layers.BatchNormalization(epsilon=1e-5)(layer)
        add_layer = tf.keras.layers.add([orig, layer])
        out = tf.keras.layers.ReLU()(add_layer)

        return out

    def save(self, in_global_step):
        self.checkpoint.save(self.checkpoint_prefix)

    def train_step(self, positions, pi, z, learning_rate=0):
        with self.train_summary_writer.as_default():
            history = self.model.fit(positions, {'policy_head_output': pi, 'value_head_output': z}, batch_size=len(positions), verbose=0)
            loss = history.history['loss'][0]
            accuracy = history.history['policy_head_output_accuracy'][0]
            self.global_step.assign_add(1)
            tf.summary.scalar('loss', loss, step=self.global_step)
            tf.summary.scalar('accuracy', accuracy, step=self.global_step)
            return accuracy, loss, self.global_step

    #@profile
    def forward(self, positions):

        positions=np.array(positions)
        if len(positions.shape) == 3:
            sp = positions.shape
            positions=np.reshape(positions, [1, sp[0], sp[1], sp[2]])
        action_probs, value = self.model(positions, training=False)

        return action_probs, value
