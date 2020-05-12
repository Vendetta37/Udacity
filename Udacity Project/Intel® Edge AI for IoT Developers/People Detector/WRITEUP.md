## Explaining Custom Layers

The process behind converting custom layers involves...

To actually add custom layers, there are a few differences depending on the original model framework. In both TensorFlow and Caffe, the first option is to register the custom layers as extensions to the Model Optimizer.

For Caffe, the second option is to register the layers as Custom, then use Caffe to calculate the output shape of the layer. You’ll need Caffe on your system to do this option.

For TensorFlow, its second option is to actually replace the unsupported subgraph with a different subgraph. The final TensorFlow option is to actually offload the computation of the subgraph back to TensorFlow during inference.

Some of the potential reasons for handling custom layers are...

For example, you could potentially use TensorFlow to load and process the inputs and outputs for a specific layer you built in that framework, if it isn’t supported with the Model Optimizer. Also, there are also unsupported layers for certain hardware, that you may run into when working with the Inference Engine.

## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were...We can use time.time() to calculate the inference time, use the file size to compare model size, and use accuracy metric to compare Performance.

The difference between model accuracy pre- and post-conversion was...post-conversion usually has lower accuracy because cloud service has higher computing power while edge device can only use FP16 or even INT8.

The size of the model pre- and post-conversion was...The storage of edge device is limited so the model size is also smaller than those on the cloud.

The inference time of the model pre- and post-conversion was...Models on edge devices are much faster because of lower precisions and smaller model size.

## Assess Model Use Cases

Some of the potential use cases of the people counter app are...Classroom and store.

Each of these use cases would be useful because...Count students and anti-theft.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows...Lighting and focal length can greatly affect the model accuracy which may be a critical area for an end user. Image size affect how does the end user store the data.
