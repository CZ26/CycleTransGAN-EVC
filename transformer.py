import tensorflow as tf
import numpy as np
import six
import math

def attention_layer(from_tensor,
                    to_tensor,
                    attention_mask=None,
                    num_attention_heads=1,
                    size_per_head=1024,
                    query_act=None,
                    key_act=None,
                    value_act=None,
                    attention_probs_dropout_prob=0.0,
                    initializer_range=0.02,
                    do_return_2d_tensor=False,
                    batch_size=None,
                    from_seq_length=None,
                    to_seq_length=None):
    
    def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                           seq_length, width):
        output_tensor = tf.reshape(
            input_tensor, [batch_size, seq_length, num_attention_heads, width])

        output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
        return output_tensor
    
    from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
    to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])
    
    if len(from_shape) != len(to_shape):
        raise ValueError("The rank of `from_tensor` must match the rank of `to_tensor`.")
        
    if len(from_shape) == 3:
        batch_size = from_shape[0]
        from_seq_length = from_shape[1]
        to_seq_length = to_shape[1]
    elif len(from_shape) == 2:
        if (batch_size is None or from_seq_length is None or to_seq_length is None):
            raise ValueError(
              "When passing in rank 2 tensors to attention_layer, the values "
              "for `batch_size`, `from_seq_length`, and `to_seq_length` "
              "must all be specified.")
            
            
    from_tensor_2d = reshape_to_matrix(from_tensor)
    to_tensor_2d = reshape_to_matrix(to_tensor)
    
    query_layer = tf.layers.dense(
      from_tensor_2d,
      num_attention_heads * size_per_head,
      activation=query_act,
      name="query",
      kernel_initializer=create_initializer(initializer_range))
    
    
    key_layer = tf.layers.dense(
      to_tensor_2d,
      num_attention_heads * size_per_head,
      activation=key_act,
      name="key",
      kernel_initializer=create_initializer(initializer_range))

    value_layer = tf.layers.dense(
      to_tensor_2d,
      num_attention_heads * size_per_head,
      activation=value_act,
      name="value",
      kernel_initializer=create_initializer(initializer_range))

    query_layer = transpose_for_scores(query_layer, batch_size,
                                     num_attention_heads, from_seq_length,
                                     size_per_head)

    key_layer = transpose_for_scores(key_layer, batch_size, num_attention_heads,
                                   to_seq_length, size_per_head)
    
    
    attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
    attention_scores = tf.multiply(attention_scores,
                                 1.0 / math.sqrt(float(size_per_head)))

    if attention_mask is not None:
        attention_mask = tf.expand_dims(attention_mask, axis=[1])
        adder = (1.0 - tf.cast(attention_mask, 'float32')) * -10000.0
        attention_scores += adder
        
    attention_probs = tf.nn.softmax(attention_scores)
    attention_probs = dropout(attention_probs, attention_probs_dropout_prob)
    
    value_layer = tf.reshape(
      value_layer,
      [batch_size, to_seq_length, num_attention_heads, size_per_head])
    
    value_layer = tf.transpose(value_layer, [0, 2, 1, 3])
    context_layer = tf.matmul(attention_probs, value_layer)
    context_layer = tf.transpose(context_layer, [0, 2, 1, 3])
    
    if do_return_2d_tensor:
        context_layer = tf.reshape(
            context_layer,
            [batch_size * from_seq_length, num_attention_heads * size_per_head])
    else:
        context_layer = tf.reshape(
            context_layer,
            [batch_size, from_seq_length, num_attention_heads * size_per_head])

    return context_layer


def gelu(x):
    cdf = 0.5 * (1.0 + tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf


def transformer(input_tensor,
                attention_mask=None,
                hidden_size=1024,
                num_hidden_layers=6,
                num_attention_heads=4,
                intermediate_size=2048,
                intermediate_act_fn=gelu,
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
                initializer_range=0.02,
                do_return_all_layers=False,
                use_position_emb=True,
                max_position_embeddings=512,
                name=None):
    
    if hidden_size % num_attention_heads != 0:
        raise ValueError(
            "The hidden size (%d) is not a multiple of the number of attention "
            "heads (%d)" % (hidden_size, num_attention_heads))

    attention_head_size = int(hidden_size / num_attention_heads)
    input_shape = get_shape_list(input_tensor, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    input_width = input_shape[2]

    if input_width != hidden_size:
        raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
                         (input_width, hidden_size))
        
    prev_output = reshape_to_matrix(input_tensor)
    
    assert_op = tf.assert_less_equal(seq_length, max_position_embeddings)
    if use_position_emb:
        with tf.control_dependencies([assert_op]):
            full_position_embeddings = tf.get_variable(name=name+'position_embedding', 
                                                       shape=[max_position_embeddings, input_width], 
                                                       initializer=create_initializer(initializer_range))
            position_embeddings = tf.slice(full_position_embeddings, [0, 0], [seq_length, -1])

            num_dims = len(prev_output.shape.as_list())
            position_broadcast_shape = []
            for _ in range(num_dims - 2):
                position_broadcast_shape.append(1)
            position_broadcast_shape.extend([seq_length, input_width])
            position_embeddings = tf.reshape(position_embeddings, position_broadcast_shape)
            prev_output += position_embeddings
            
    prev_output = layer_norm_and_dropout(prev_output, 0.2)
    
    all_layer_outputs = []
    
    for layer_idx in range(num_hidden_layers):
        with tf.variable_scope(name+"layer_%d" % layer_idx):
            layer_input = prev_output
            with tf.variable_scope("attention"):
                attention_heads = []
                with tf.variable_scope("self"):
                    attention_head = attention_layer(from_tensor=layer_input,
                                                      to_tensor=layer_input,
                                                      attention_mask=attention_mask,
                                                      num_attention_heads=num_attention_heads,
                                                      size_per_head=attention_head_size,
                                                      attention_probs_dropout_prob=attention_probs_dropout_prob,
                                                      initializer_range=initializer_range,
                                                      do_return_2d_tensor=True,
                                                      batch_size=batch_size,
                                                      from_seq_length=seq_length,
                                                      to_seq_length=seq_length)
                    attention_heads.append(attention_head)

                if len(attention_heads) == 1:
                    attention_output = attention_heads[0]
                else:
                    attention_output = tf.concat(attention_heads, axis=-1)

                with tf.variable_scope("output"):
                    attention_output = tf.layers.dense(
                          attention_output,
                          hidden_size,
                          kernel_initializer=create_initializer(initializer_range))
                    attention_output = dropout(attention_output, hidden_dropout_prob)
                    attention_output = layer_norm(attention_output + layer_input)
            
            with tf.variable_scope("intermediate"):
                intermediate_output = tf.layers.dense(
                                                attention_output,
                                                intermediate_size,
                                                activation=intermediate_act_fn,
                                                kernel_initializer=create_initializer(initializer_range))

            with tf.variable_scope("output"):
                layer_output = tf.layers.dense(
                                intermediate_output,
                                hidden_size,
                                kernel_initializer=create_initializer(initializer_range))
                layer_output = dropout(layer_output, hidden_dropout_prob)
                layer_output = layer_norm(layer_output + attention_output)
                prev_output = layer_output
                all_layer_outputs.append(layer_output)
            
    if do_return_all_layers:
        final_outputs = []
        for layer_output in all_layer_outputs:
            final_output = reshape_from_matrix(layer_output, input_shape)
            final_outputs.append(final_output)
            
        return final_outputs
    else:
        final_output = reshape_from_matrix(prev_output, input_shape)
        return final_output
        
        
def dropout(input_tensor, dropout_prob):

    if dropout_prob is None or dropout_prob == 0.0:
        return input_tensor

    output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
    return output


def layer_norm(input_tensor, name=None):
    return tf.contrib.layers.layer_norm(
      inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)


def layer_norm_and_dropout(input_tensor, dropout_prob, name=None):
    
    output_tensor = layer_norm(input_tensor, name)
    output_tensor = dropout(output_tensor, dropout_prob)
    return output_tensor


def create_attention_mask_from_input_mask(from_tensor, to_mask):
    from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]

    to_shape = get_shape_list(to_mask, expected_rank=2)
    to_seq_length = to_shape[1]

    to_mask = tf.cast(tf.reshape(to_mask, [batch_size, 1, to_seq_length]), tf.float32)

    broadcast_ones = tf.ones(shape=[batch_size, from_seq_length, 1], dtype=tf.float32)

    mask = broadcast_ones * to_mask

    return mask


def create_initializer(initializer_range=0.02):
    return tf.truncated_normal_initializer(stddev=initializer_range)


def get_shape_list(tensor, expected_rank=None, name=None):
    if name is None:
        name = tensor.name

    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    
    return shape


def reshape_to_matrix(input_tensor):
  
    ndims = input_tensor.shape.ndims
    if ndims < 2:
        raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
                     (input_tensor.shape))
    if ndims == 2:
        return input_tensor

    width = input_tensor.shape[-1]
    output_tensor = tf.reshape(input_tensor, [-1, width])
    return output_tensor


def reshape_from_matrix(output_tensor, orig_shape_list):
    
    if len(orig_shape_list) == 2:
        return output_tensor

    output_shape = get_shape_list(output_tensor)
    orig_dims = orig_shape_list[0:-1]
    width = output_shape[-1]

    return tf.reshape(output_tensor, orig_dims + [width])


def assert_rank(tensor, expected_rank, name=None):
    if name is None:
        name = tensor.name

    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True

    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        scope_name = tf.get_variable_scope().name
        raise ValueError(
            "For the tensor `%s` in scope `%s`, the actual rank "
            "`%d` (shape = %s) is not equal to the expected rank `%s`" %
            (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))