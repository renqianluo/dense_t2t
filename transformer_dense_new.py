"""transformer (attention).

encoder: [Self-Attention, Feed-forward] x n
decoder: [Self-Attention, Source-Target-Attention, Feed-forward] x n
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

# Dependency imports

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model
from tensor2tensor.models import transformer

import tensorflow as tf

__author__ = 'renqianluo'


@registry.register_model
class TransformerDense(t2t_model.T2TModel):
  """Attention net.  See file docstring."""

  def model_fn_body(self, features):
    # Remove dropout if not training
    hparams = copy.copy(self._hparams)
    targets = features["targets"]
    inputs = features.get("inputs")
    target_space = features.get("target_space_id")

    inputs = common_layers.flatten4d3d(inputs)
    targets = common_layers.flatten4d3d(targets)

    (encoder_input, encoder_attention_bias, _) = transformer_prepare_encoder(
        inputs, target_space, hparams)
    (decoder_input, decoder_self_attention_bias) = transformer_prepare_decoder(
        targets, hparams)

    encoder_input = tf.nn.dropout(encoder_input, 1.0 - hparams.layer_prepostprocess_dropout)
    decoder_input = tf.nn.dropout(decoder_input, 1.0 - hparams.layer_prepostprocess_dropout)
    encoder_output_list = transformer_encoder(encoder_input, encoder_attention_bias, hparams)

    decoder_output = transformer_decoder(decoder_input, 
      encoder_output_list, decoder_self_attention_bias, 
      encoder_attention_bias, hparams)
    decoder_output = tf.expand_dims(decoder_output, 2)

    return decoder_output


def transformer_prepare_encoder(inputs, target_space, hparams):
  """Prepare one shard of the model for the encoder.

  Args:
    inputs: a Tensor.
    target_space: a Tensor.
    hparams: run hyperparameters

  Returns:
    encoder_input: a Tensor, bottom of encoder stack
    encoder_self_attention_bias: a Tensor, containing large negative values
      to implement masked attention and possibly baises for diagonal
      alignments
    encoder_padding: a Tensor
  """
  # Flatten inputs.
  ishape_static = inputs.shape.as_list()
  encoder_input = inputs
  encoder_padding = common_attention.embedding_to_padding(encoder_input)
  ignore_padding = common_attention.attention_bias_ignore_padding(encoder_padding)
  encoder_self_attention_bias = ignore_padding
  encoder_decoder_attention_bias = ignore_padding
  if hparams.proximity_bias:
    encoder_self_attention_bias += common_attention.attention_bias_proximal(
        tf.shape(inputs)[1])
  # Append target_space_id embedding to inputs.
  emb_target_space = common_layers.embedding(
    target_space, 32, ishape_static[-1], name="target_space_embedding")
  emb_target_space = tf.reshape(emb_target_space, [1, 1, -1])
  encoder_input += emb_target_space
  if hparams.pos == "timing":
    encoder_input = common_attention.add_timing_signal_1d(encoder_input)
  return (encoder_input, encoder_self_attention_bias, encoder_decoder_attention_bias)


def transformer_prepare_decoder(targets, hparams):
  """Prepare one shard of the model for the decoder.

  Args:
    targets: a Tensor.
    hparams: run hyperparameters

  Returns:
    decoder_input: a Tensor, bottom of decoder stack
    decoder_self_attention_bias: a Tensor, containing large negative values
    to implement masked attention and possibly baises for diagonal alignments
  """
  decoder_self_attention_bias = (
    common_attention.attention_bias_lower_triangle(tf.shape(targets)[1]))
  if hparams.proximity_bias:
    decoder_self_attention_bias += common_attention.attention_bias_proximal(
      tf.shape(targets)[1])
  decoder_input = common_layers.shift_left_3d(targets)
  if hparams.pos == "timing":
    decoder_input = common_attention.add_timing_signal_1d(decoder_input)
  return (decoder_input, decoder_self_attention_bias)

def transformer_encoder(encoder_input,
                        encoder_self_attention_bias,
                        hparams,
                        name="encoder"):
  """A stack of transformer layers.

  Args:
    encoder_input: a Tensor
    residual_fn: a function from (layer_input, layer_output) -> combined_output
    encoder_self_attention_bias: bias Tensor for self-attention
       (see common_attention.attention_bias())
    hparams: hyperparameters for model
    name: a string

  Returns:
    encoder_output_list: a list of Tensors
  """
  encoder_output_list = []
  layer_input = encoder_input
  with tf.variable_scope(name):
    pad_remover = None
    if hparams.use_pad_remover:
        pad_remover = expert_utils.PadRemover(common_attention.attention_bias_to_padding(encoder_self_attention_bias))
    n_layers = hparams.num_encoder_layers or hparams.num_hidden_layers
    for layer in xrange(n_layers):
      with tf.variable_scope("layer_%d" % layer):
        if layer == 0:
          x = layer_input
        else:
          x = linear_projection(layer_input, hparams.hidden_size)
        with tf.variable_scope("self_attention"):
          y = common_attention.multihead_attention(
              common_layers.layer_preprocess(x, hparams),
              None, 
              encoder_self_attention_bias,
              hparams.attention_key_channels or hparams.hidden_size, #toatl_key_depth
              hparams.attention_value_channels or hparams.hidden_size, #total_value_dept
              hparams.hidden_size, #output_depth
              hparams.num_heads,
              hparams.attention_dropout)
          x = common_layers.layer_postprocess(x, y, hparams)
        with tf.variable_scope("ffn"):
          y = transformer_ffn_layer(common_layers.layer_preprocess(x, hparams), hparams, pad_remover)
          x = common_layers.layer_postprocess(x, y, hparams)
        encoder_output_list.append(x)
        if layer != n_layers - 1:
            layer_input = tf.concat([layer_input, x], axis=-1)
  return encoder_output_list


def transformer_decoder(decoder_input,
                        encoder_output_list,
                        decoder_self_attention_bias,
                        encoder_decoder_attention_bias,
                        hparams,
                        name="decoder"):
  """A stack of transformer layers.

  Args:
    decoder_input: a Tensor
    encoder_output_list: a lsit of Tensors
    decoder_self_attention_bias: bias Tensor for self-attention
      (see common_attention.attention_bias())
    encoder_decoder_attention_bias: bias Tensor for encoder-decoder attention
      (see common_attention.attention_bias())
    hparams: hyperparameters for model
    name: a string

  Returns:
    y: a Tensors
  """
  layer_input = decoder_input
  with tf.variable_scope(name):
    n_layers = hparams.num_decoder_layers or hparams.num_hidden_layers
    for layer in xrange(n_layers):
      with tf.variable_scope("layer_%d" % layer):
        if layer == 0:
          x = layer_input
        else:
          x = linear_projection(layer_input, hparams.hidden_size)
        with tf.variable_scope("self_attention"):
          y = common_attention.multihead_attention(
              common_layers.layer_preprocess(x, hparams),
              None,
              decoder_self_attention_bias,
              hparams.attention_key_channels or hparams.hidden_size, #total_key_depth
              hparams.attention_value_channels or hparams.hidden_size, #total_value_depth
              hparams.hidden_size,
              hparams.num_heads,
              hparams.attention_dropout)
          x = common_layers.layer_postprocess(x, y, hparams)
        with tf.variable_scope("encdec_attention"):
          for enc_layer in xrange(hparams.num_hidden_layers):
            with tf.variable_scope("enc_%d" % enc_layer):
              yi = common_attention.multihead_attention(
                  common_layers.layer_preprocess(x, hparams),
                  encoder_output_list[enc_layer],
                  encoder_decoder_attention_bias,
                  hparams.attention_key_channels or hparams.hidden_size, #total_key_depth
                  hparams.attention_value_channels or hparams.hidden_size, #total_value_depth
                  hparams.hidden_size,
                  hparams.num_heads,
                  hparams.attention_dropout)
              yi = common_layers.layer_postprocess(x, yi, hparams)
              if enc_layer == 0:
                y = yi
              else:
                y = tf.concat([y, yi], axis=-1)
          x = linear_projection(y, hparams.hidden_size)
        with tf.variable_scope("ffn"):
          y = transformer_ffn_layer(common_layers.layer_preprocess(x, hparams), hparams)
          x = common_layers.layer_postprocess(x, y, hparams)
        layer_input = tf.concat([layer_input, x], axis=-1)
          
    # Due to the high dimension of the output, we use a linear_projection here
    with tf.variable_scope("linear_transformation"):
      y = linear_projection(layer_input, hparams.hidden_size)
  return y


def transformer_ffn_layer(x, hparams):
  """Feed-forward layer in the transformer.

  Args:
    x: a Tensor of shape [batch_size, length, hparams.hidden_size]
    hparams: hyperparmeters for model
    pad_remover: an expert_utils.PadRemover object tracking the padding
      positions. If provided, when using convolutional settings, the padding
      is removed before applying the convolution, and restored afterward. This
      can give a significant speedup.
  Returns:
    a Tensor of shape [batch_size, length, hparams.hidden_size]
  """

  if hparams.ffn_layer == "conv_hidden_relu":
    if pad_remover:
      original_shape = tf.shape(x)
      # Collapse `x` across examples, and remove padding positions.
      x = tf.reshape(x, tf.concat([[-1], tf.shape(x)[2:]], axis=0))
      x = tf.expand_dims(pad_remover.remove(x), axis=0)
    conv_output = common_layers.conv_hidden_relu(
        x,
        hparams.filter_size,
        hparams.hidden_size,
        dropout=hparams.relu_dropout)
    if pad_remover:
      # Restore `conv_output` to the original shape of `x`, including padding.
      conv_output = tf.reshape(
      pad_remover.restore(tf.squeeze(conv_output, axis=0)), original_shape)
    return conv_output
  elif hparams.ffn_layer == "parameter_attention":
    return common_attention.parameter_attention(
        x,
        hparams.parameter_attention_key_channels or hparams.hidden_size,
        hparams.parameter_attention_value_channels or hparams.hidden_size,
        hparams.hidden_size,
        hparams.filter_size,
        hparams.num_heads,
        hparams.attention_dropout)
  elif hparams.ffn_layer == "conv_hidden_relu_with_sepconv":
    return common_layers.conv_hidden_relu(
        x,
        hparams.filter_size,
        hparams.hidden_size,
        kernel_size=(3, 1),
        second_kernel_size=(31, 1),
        padding="LEFT",
        dropout=hparams.relu_dropout)
  else:
    assert hparams.ffn_layer == "none"
    return x

def linear_projection(inputs,
                     hidden_size,
                     kernel_size=(1, 1),
                     **kwargs):
  """Hidden layer of linear projection."""
  return common_layers.conv1d(inputs, hidden_size, 1)
  '''name = kwargs.pop("name") if "name" in kwargs else None
  with tf.variable_scope(name, "linear_projection", [inputs]):
    if inputs.get_shape().ndims == 3:
      is_3d = True
      inputs = tf.expand_dims(inputs, 2)
    else:
      is_3d = False
    ret = common_layers.conv(
        inputs,
        hidden_size,
        kernel_size,
        **kwargs)
    if is_3d:
      ret = tf.squeeze(ret, 2)
    return ret'''

@registry.register_hparams
def transformer_dense():
  hparams = transformer.transformer_base()
  hparams.layer_preprocess_sequence = "none"
  hparams.layer_postprocess_sequence = "dn"
  return hparams
