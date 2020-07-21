import warnings
warnings.filterwarnings('ignore')

import io
import random
import numpy as np
import mxnet as mx
import gluonnlp as nlp
from gluonnlp.calibration import BertLayerCollector
# this notebook assumes that all required scripts are already
# downloaded from the corresponding tutorial webpage on http://gluon-nlp.mxnet.io
from bert import data

nlp.utils.check_version('0.8.1')

np.random.seed(100)
random.seed(100)
mx.random.seed(10000)
# change `ctx` to `mx.cpu()` if no GPU is available.
ctx = mx.cpu(0)


bert_base, vocabulary = nlp.model.get_model('bert_12_768_12',
                                             dataset_name='book_corpus_wiki_en_uncased',
                                             pretrained=True, ctx=ctx, use_pooler=True,
                                             use_decoder=False, use_classifier=False)
print(bert_base)


bert_classifier = nlp.model.BERTClassifier(bert_base, num_classes=2, dropout=0.1)
# only need to initialize the classifier layer.
bert_classifier.classifier.initialize(init=mx.init.Normal(0.02), ctx=ctx)
bert_classifier.hybridize(static_alloc=True)

# softmax cross entropy loss for classification
loss_function = mx.gluon.loss.SoftmaxCELoss()
loss_function.hybridize(static_alloc=True)

metric = mx.metric.Accuracy()


tsv_file = io.open('dev.tsv', encoding='utf-8')
for i in range(5):
    print(tsv_file.readline())


# Skip the first line, which is the schema
num_discard_samples = 1
# Split fields by tabs
field_separator = nlp.data.Splitter('\t')
# Fields to select from the file
field_indices = [3, 4, 0]
data_train_raw = nlp.data.TSVDataset(filename='dev.tsv',
                                 field_separator=field_separator,
                                 num_discard_samples=num_discard_samples,
                                 field_indices=field_indices)
sample_id = 0
# Sentence A
print(data_train_raw[sample_id][0])
# Sentence B
print(data_train_raw[sample_id][1])
# 1 means equivalent, 0 means not equivalent
print(data_train_raw[sample_id][2])


# Use the vocabulary from pre-trained model for tokenization
bert_tokenizer = nlp.data.BERTTokenizer(vocabulary, lower=True)

# The maximum length of an input sequence
max_len = 128

# The labels for the two classes [(0 = not similar) or  (1 = similar)]
all_labels = ["0", "1"]

# whether to transform the data as sentence pairs.
# for single sentence classification, set pair=False
# for regression task, set class_labels=None
# for inference without label available, set has_label=False
pair = True
transform = data.transform.BERTDatasetTransform(bert_tokenizer, max_len,
                                                class_labels=all_labels,
                                                has_label=True,
                                                pad=True,
                                                pair=pair)
data_train = data_train_raw.transform(transform)

print('vocabulary used for tokenization = \n%s'%vocabulary)
print('%s token id = %s'%(vocabulary.padding_token, vocabulary[vocabulary.padding_token]))
print('%s token id = %s'%(vocabulary.cls_token, vocabulary[vocabulary.cls_token]))
print('%s token id = %s'%(vocabulary.sep_token, vocabulary[vocabulary.sep_token]))
print('token ids = \n%s'%data_train[sample_id][0])
print('segment ids = \n%s'%data_train[sample_id][1])
print('valid length = \n%s'%data_train[sample_id][2])
print('label = \n%s'%data_train[sample_id][3])


# The hyperparameters
batch_size = 32
lr = 5e-6

# The FixedBucketSampler and the DataLoader for making the mini-batches
train_sampler = nlp.data.FixedBucketSampler(lengths=[int(item[2]) for item in data_train],
                                            batch_size=batch_size,
                                            shuffle=True)
bert_dataloader = mx.gluon.data.DataLoader(data_train, batch_sampler=train_sampler)

trainer = mx.gluon.Trainer(bert_classifier.collect_params(), 'adam',
                           {'learning_rate': lr, 'epsilon': 1e-9})

# Collect all differentiable parameters
# `grad_req == 'null'` indicates no gradients are calculated (e.g. constant parameters)
# The gradients for these params are clipped later
params = [p for p in bert_classifier.collect_params().values() if p.grad_req != 'null']
grad_clip = 1

# Training the model with only three epochs
log_interval = 4
num_epochs = 3
for epoch_id in range(num_epochs):
    metric.reset()
    step_loss = 0
    for batch_id, (token_ids, segment_ids, valid_length, label) in enumerate(bert_dataloader):
        with mx.autograd.record():

            # Load the data to the GPU
            token_ids = token_ids.as_in_context(ctx)
            valid_length = valid_length.as_in_context(ctx)
            segment_ids = segment_ids.as_in_context(ctx)
            label = label.as_in_context(ctx)

            # Forward computation
            out = bert_classifier(token_ids, segment_ids, valid_length.astype('float32'))
            ls = loss_function(out, label).mean()

        # And backwards computation
        ls.backward()

        # Gradient clipping
        trainer.allreduce_grads()
        nlp.utils.clip_grad_global_norm(params, 1)
        trainer.update(1)

        step_loss += ls.asscalar()
        metric.update([label], [out])

        # Printing vital information
        if (batch_id + 1) % (log_interval) == 0:
            print('[Epoch {} Batch {}/{}] loss={:.4f}, lr={:.7f}, acc={:.3f}'
                         .format(epoch_id, batch_id + 1, len(bert_dataloader),
                                 step_loss / log_interval,
                                 trainer.learning_rate, metric.get()[1]))
            step_loss = 0


# The hyperparameters
dev_batch_size = 32
num_calib_batches = 5
quantized_dtype = 'auto'
calib_mode = 'customize'

# sampler for evaluation
pad_val = vocabulary[vocabulary.padding_token]
batchify_fn = nlp.data.batchify.Tuple(
    nlp.data.batchify.Pad(axis=0, pad_val=pad_val),  # input
    nlp.data.batchify.Pad(axis=0, pad_val=0),  # segment
    nlp.data.batchify.Stack(),  # length
    nlp.data.batchify.Stack('int32'))  # label
dev_dataloader = mx.gluon.data.DataLoader(data_train, batch_size=dev_batch_size, num_workers=4,
                                           shuffle=False, batchify_fn=batchify_fn)

# Calibration function
gran = 'tensor-wise'
def calibration(net, dev_data, num_calib_batches, quantized_dtype, calib_mode):
    """calibration function on the dev dataset."""
    print('Now we are doing calibration on dev with cpu.')
    collector = BertLayerCollector(clip_min=-50, clip_max=10, logger=None)
    num_calib_examples = dev_batch_size * num_calib_batches
    quantized_net = mx.contrib.quantization.quantize_net_v2(net, quantized_dtype=quantized_dtype,
                                                            exclude_layers=[],
                                                            quantize_mode='smart',
                                                            quantize_granularity=gran,
                                                            calib_data=dev_data,
                                                            calib_mode=calib_mode,
                                                            num_calib_examples=num_calib_examples,
                                                            ctx=mx.cpu(),
                                                            LayerOutputCollector=collector,
                                                            logger=None)
    print('Calibration done with success.')
    return quantized_net

# will remove until mxnet 1.7 release.
try:
    quantized_net = calibration(bert_classifier,
                                dev_dataloader,
                                num_calib_batches,
                                quantized_dtype,
                                calib_mode)
except AttributeError:
    nlp.utils.version.check_version('1.7.0', warning_only=True, library=mx)
    warnings.warn('INT8 Quantization for BERT need mxnet-mkl >= 1.6.0b20200115')


prefix = '../model/model_bert_squad_' + gran + '_quantized'

def deployment(net, prefix, dataloader):
    net.export(prefix, epoch=0)
    print('Saving quantized model at ', prefix)
    print('load symbol file directly as SymbolBlock for model deployment.')
    static_net = mx.gluon.SymbolBlock.imports('{}-symbol.json'.format(prefix),
                                    ['data0', 'data1', 'data2'],
                                    '{}-0000.params'.format(prefix))
    static_net.hybridize(static_alloc=True, static_shape=True)
    for batch_id, (token_ids, segment_ids, valid_length, label) in enumerate(dev_dataloader):
            token_ids = token_ids.as_in_context(mx.cpu())
            valid_length = valid_length.as_in_context(mx.cpu())
            segment_ids = segment_ids.as_in_context(mx.cpu())
            label = label.as_in_context(mx.cpu())
            out = static_net(token_ids, segment_ids, valid_length.astype('float32'))
            metric.update([label], [out])

            # Printing vital information
            if (batch_id + 1) % (log_interval) == 0:
                print('[Batch {}/{}], acc={:.3f}'
                            .format(batch_id + 1, len(bert_dataloader),
                                    metric.get()[1]))
    return metric

# will remove until mxnet 1.7 release.
try:
    eval_metric = deployment(quantized_net, prefix, dev_dataloader)
except NameError:
    nlp.utils.version.check_version('1.7.0', warning_only=True, library=mx)
    warnings.warn('INT8 Quantization for BERT need mxnet-mkl >= 1.6.0b20200115')
