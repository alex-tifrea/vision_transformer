# Copyright 2021 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from absl import app
from absl import flags
from absl import logging
from clu import platform
import jax
from ml_collections import config_flags
import tensorflow as tf

from vit_jax import inference_time
from vit_jax import train
from vit_jax import utils
from vit_jax.utils import retry

import mlflow
import os

os.environ["MLFLOW_TRACKING_USERNAME"] = "exp-01.mlflow-yang.tifreaa"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "parola"
remote_server_uri = "https://exp-01.mlflow-yang.inf.ethz.ch"
mlflow.set_tracking_uri(remote_server_uri)


FLAGS = flags.FLAGS

_WORKDIR = flags.DEFINE_string('workdir', None,
                               'Directory to store logs and model data.')
config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the training hyperparameter configuration.',
    lock_config=True)
flags.mark_flags_as_required(['config', 'workdir'])
# Flags --jax_backend_target and --jax_xla_backend are available through JAX.


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  utils.add_gfile_logger(_WORKDIR.value)

  # Hide any GPUs form TensorFlow. Otherwise TF might reserve memory and make
  # it unavailable to JAX.
  tf.config.experimental.set_visible_devices([], 'GPU')

  jax.config.update('jax_log_compiles', True)

  logging.info('JAX process: %d / %d', jax.process_index(), jax.process_count())
  logging.info('JAX local devices: %r', jax.local_devices())
  jax_xla_backend = ('None' if FLAGS.jax_xla_backend is None else
                     FLAGS.jax_xla_backend)
  logging.info('Using JAX XLA backend %s', jax_xla_backend)

  logging.info('Config: %s', FLAGS.config)

  retry(lambda: mlflow.set_experiment("vit_finetuning"))
  curr_run = retry(lambda: mlflow.start_run(run_name=f"{FLAGS.config.dataset}_{FLAGS.config.total_steps}"))

  mlflow_params = utils.flatten_config_dict(FLAGS.config, sep=".")
  mlflow_params["run_id"] = curr_run.info.run_id
  mlflow_params["workdir"] = _WORKDIR.value
  retry(lambda: mlflow.log_params(mlflow_params))

  # Add a note so that we can tell which task is which JAX host.
  # (Depending on the platform task 0 is not guaranteed to be host 0)
  platform.work_unit().set_task_status(f'process_index: {jax.process_index()}, '
                                       f'process_count: {jax.process_count()}')
  platform.work_unit().create_artifact(platform.ArtifactType.DIRECTORY,
                                       _WORKDIR.value, 'workdir')

  if FLAGS.config.trainer == 'train':
    train.train_and_evaluate(FLAGS.config, _WORKDIR.value)
  elif FLAGS.config.trainer == 'inference_time':
    inference_time.inference_time(FLAGS.config, _WORKDIR.value)
  else:
    raise app.UsageError(f'Unknown trainer: {FLAGS.config.trainer}')

if __name__ == '__main__':
  # Provide access to --jax_backend_target and --jax_xla_backend flags.
  jax.config.config_with_absl()
  app.run(main)
