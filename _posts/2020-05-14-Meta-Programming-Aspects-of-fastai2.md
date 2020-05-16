---
title: Metaprogramming aspects of fastai2
description: fastai-v2 has "transformed" the way we write Python for Deep Learning. A library called 'fastcore' is where all the magic happens and is framework agnostic, so most of the ideas I'm about to discuss could be easily integrated in your favourite python library.
toc: true
badges: false
comments: true
image: images/logo.png
author: Kshitij Patil
categories: [fastai2, fastcore, metaprogramming]
layout: post
---

# `encode` for model, `decode` for yourself

Conventionally, in PyTorch, we write most of the data pre-processing logic in a single block of code, IE, inside `__getitem__` of PyTorch `Dataset`. Even if we try to create separate methods for different parts of the Pipeline, we end up creating highly coupled logic, which works best for your current project, but you often hesitate to reuse this code for your next project. 

```python
class WhaleDataset(Dataset):
    def __init__(self, datafolder, datatype='train', df=None, transform =
                 transforms.Compose([transforms.ToTensor()]), y=None):
        self.datafolder = datafolder
        self.datatype = datatype
        self.y = y
        if self.datatype == 'train':
            self.df = df.values
        self.image_files_list = [s for s in os.listdir(datafolder)]
        self.transform = transform


    def __len__(self):
        return len(self.image_files_list)
    
    def __getitem__(self, idx):
        if self.datatype == 'train':
            img_name = os.path.join(self.datafolder, self.df[idx][0])
            label = self.y[idx]
            
        elif self.datatype == 'test':
            img_name = os.path.join(self.datafolder, self.image_files_list[idx])
            label = np.zeros((5005,))

        image = Image.open(img_name).convert('RGB')
        image = self.transform(image)
        if self.datatype == 'train':
            return image, label
        elif self.datatype == 'test':
            # so that the images will be in a correct order
            return image, label, self.image_files_list[idx]
```

The problem with this approach is a lack of modularity and the ability to test individual block of code. Certain problems that might be left unnoticed with this design are (considering image classification problem):

1. You displayed a single image from given filenames, are you sure the same image is not being repeated in a batch?
2. You got the numeric labels from given text labels, but are you sure they're mapped properly, do you know their reverse mapping, as in, which no. represents what?
3. have you tried several permutations of your given augmentations? What if your subject is getting cropped out of the image and you left wondering why my model isn't converging 🤔
4. Are you sure the split logic isn't mixing the train and validation set?

We need a better way to organize the `Transforms` and fastai has introduced a novel way for this. A `Transform` class from fastcore (library from fastai) has a special behavior due to its `encodes`, `decodes`, and `setup` methods.

<img src="{{site.baseurl}}/images/meta-fastai/fastai-encode-decode.png" style="zoom:200%;">

As name suggests, `encodes` hold primary logic of transforming the input, which the next `transform` will grab as input, process it and pass-on to the next. Similarly,  `decodes` has the logic of undoing the effect of current transform. An ideal use-case for `decodes` is to manage the inconsistencies between different libraries.  

> For instance, in object detection, suppose your math library uses a matrix notations for coordinates, so model will be trained to predict the bounding in that form, but your visualization library uses the exact opposite, so you want to undo the preprocessing step while showing the results and `decodes` is the place to keep that logic.

Key features of`Transform` :

1. **type-dispatch**: yes ! you can define type-specific methods in Python :astonished: what do you think `TensorImage` is in the above image.? Here is the definition

```python
class TensorImage(TensorImageBase): pass
```

`TensorImageBase` is just a wrapper around a PyTorch `Tensor` holding image channels and `TensorImage` is just an empty class.  We call it semantic type, as this type has some special treatment in the Pipeline. But the advantage is, now you can define a `Transform` (augmentation / preprocessing) for `TensorImage`, test that in isolation and just freeze that block of code in your library, no need to write that again in `__getitem__`

2. **Handling tuples**: you just pass in tuple of whatever, `Transform` will act on only types it's allowed to (all instances of that type). 

3. **Reversibility**: As discussed earlier, you can define how to undo the operation performed on a `Tensor`

4. **Ordering**: you can define some order (int) for a `Transform` and they'll be applied in the ascending order of that. Suppose, you want some augmentation to be applied before Normalization, just specify the order of your `Transform` lower than `Normalize` and it'll be applied in desired order.

   You can learn more about `Transform` [here](http://fastcore.fast.ai/transform#The-main-Transform-features:). Let's look at an example:

```python
class Normalize(Transform):
    "Normalize/denorm batch of `TensorImage`"
    order=99
    ...
    def setups(self, dl:DataLoader):
        if self.mean is None or self.std is None:
            x,*_ = dl.one_batch()
            self.mean,self.std = x.mean(self.axes, keepdim=True),x.std(self.axes, keepdim=True)+1e-7

    def encodes(self, x:TensorImage): return (x-self.mean) / self.std
    def decodes(self, x:TensorImage):
        f = to_cpu if x.device.type=='cpu' else noop
        return (x*f(self.std) + f(self.mean))
    ...
```
`Normalize` is one of the transforms offered by fastai. The `setups` method is used to perform one-time calculations such as mean and standard deviation in this case. `encodes` is just a replacement for `__call__`  while `decodes` is denormalizing the Tensor. Notice the type-annotation `TensorImage` which has a lot more meaning than just a typing hint offered by Python 3.7. `Transform` also retains the types, IE, an input `TensorImage` will be returned as `TensorImage` only and all this is handled by a meta-class for `Transform`

# Method Overloading and Type Dispatch

```python
class IntToFloatTensor(Transform):
    "Transform image to float tensor, optionally dividing by 255 (e.g. for images)."
    order = 10 #Need to run after PIL transforms on the GPU
    def __init__(self, div=255., div_mask=1): store_attr(self, 'div,div_mask')
    def encodes(self, o:TensorImage): return o.float().div_(self.div)
    def encodes(self, o:TensorMask ): return o.long() // self.div_mask
    def decodes(self, o:TensorImage): return ((o.clamp(0., 1.) * self.div).long()) if self.div else o
```

`IntToFloatTensor` has a separate behavior for `TensorImage` and `TensorMask` and this is achieved by dynamic type-dispatch. So if the type is `TensorImage`, it'll be divided by 255 and returned as a float Tensor, whereas Masks won't be scaled and returned as long Tensors. Now, say, you want to return a double() Tensor instead of float and add some $\epsilon$=1e-7 to it, you can simply do that using `@IntToFloatTensor` decorator.

```python
@IntToFloatTensor
def encodes(self, x: TensorImage): return x.double().div_(self.div) + 1e-7
```

which means, you don't need to write a class extending `Transform` if it's some variation of existing one in the library, just write type-annotated `encodes` with a decorator of that method.

> You don't necessarily need extend the `Transform` class to make type-dispatch work. There's a decorator for that as well

```python
@typedispatch
def show_results(x:TensorImage, y:(TensorMask, TensorPoint, TensorBBox), ...):

@typedispatch
def show_results(x:TensorImage, y:TensorCategory, ...):

@typedispatch
def show_results(x:TensorImage, y:TensorImage, ...):
```

fastai has utility functions to show the training results. Now, each combination has a different way of showing the results. For example, `TensorImage` and `TensorCategory` will simply show the image with target vs predicted label as title. A `TensorImage` with any localization type will actually show both ground-truth and predicted images.

<img src="{{site.baseurl}}/images/meta-fastai/seg-show-results.png">

In this case, a ground-truth segmentation mask (left) and predicted (right) mask is shown as result. Look how everything is coming together and those semantic types are the actors in this plot :sunglasses:

# `@patch` and `@patch-property`

There are several practices used to extend the functionality of existing class in the library. Swift or Kotlin does this by writing extension functions. "Monkey-patching" is one of those aspects used by several python libraries and fastai provides a decorator for that. A cool example of this:

```python
@patch
def pca(x:Tensor, k=2):
    "Compute PCA of `x` with `k` dimensions."
    x = x-torch.mean(x,0)
    U,S,V = torch.svd(x.t())
    return torch.mm(x,U[:,:k])
```

We're monkey-patching a method to compute PCA of given `Tensor` which allows us to call this method as if it was part of PyTorch tensors; so you can simply call `x.pca()`  and it'll work like a charm. Another great example of this, if you're coming from `numpy` background, you might be used to the `shape` property of arrays. This is not available in `PyTorch`, but not anymore, just patch it to the `Tensor` class and you're good to go:

```python
@patch_property
def shape(x: Image.Image): return x.size[1],x.size[0]
```

Another example that I couldn't resist to mention, you've a `ls()` method for Path class, thanks to `@patch` :wink:

# You can `@delegate` the rest

Say you are writing a wrapper for library function to add some tweaks and you're only concerned with some parameters, rest of them will be passed on to the original method, how can you make sure that user should be able to work with every-single-parameter of original method. You need to write all those parameters in your method definition, don't you?. Let's work with `matplotlib` for this:

`matplotlib.pyplot.plot` has several customization parameters for graph, but we don't really know all of them. As per [matplotlib documentation](https://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot), these are the properties of `Line2D` class, however, as matplotlib uses **kwargs to handle them, it remains mystery what all we can modify:

<img src="{{site.baseurl}}/images/meta-fastai/plt-plot.png">

This is the type-hint you get in the IDE for `plt.plot()` method and clearly most of the arguments are disguised in the **kwargs. I wrote a simple wrapper function for this which delegates those kwargs to Line2D.

```python
@delegates(matplotlib.pyplot.Line2D)
def mat_plot(x, **kwargs):
  return plt.plot(x,**kwargs)
```

And.... the result is:

<img src="{{site.baseurl}}/images/meta-fastai/mat_plot.png">

:frowning: we have these many arguments to play with!! 

But `@delegates` isn't just used to make those type-hints appear for you, you can have your own parameters along with those **kwargs.

> in simple terms, your method doesn't actually accept the kwargs, but the only parameters that are available in the method you're delegating to. 

For example, in fastai, there's a method called `save_model` which has the logic to save the PyTorch model to a given path. Now `Learner` has the "model", "optimizer state" and some parameters which could be used to create the "destination path" for `save_model`, so `Learner` will build the desired path and hand over its available parameters to `save_model` to perform actual save:

```python
class Learner():
    ...
    
    @delegates(save_model)
    def save(self, file, **kwargs):
        file = join_path_file(file, self.path/self.model_dir, ext='.pth')
        save_model(file, self.model, getattr(self,'opt',None), **kwargs)
```

`@delegates` could also be used with classes. With no target specified, it'll delegate the parameters from you constructor to the base class's one.

# Who likes boilerplate?

Some decorators that makes you write even lesser code with Python :bangbang: 

{% include info.html text="Now we're manipulating the signature of class/method" %}

Do you want to control what could be passed in those `**kwargs`? This is not for your convenience :stuck_out_tongue:actually, end users would always like to know what they could pass in those `**kwargs` instead of guessing randomly. So it's a deal benefiting both the parties.

## `@use_kwargs` and `@use_kwargs_dict`

Making your method super flexible has no harm and who knows it may cover the use-case that you didn't even think of. Also, cutting down the parameter list because it's getting too long is not a great excuse either. So, you have a method with most of the kwargs being `None` and tired of listing all of them? or you've already listed them somewhere else? then `@use_kwargs` is made for you. Borrowing this snippet from [pytorch-metric-learning](https://github.com/KevinMusgrave/pytorch-metric-learning/blob/93e6421addc822d7565c64d7ff166d46be757acd/src/pytorch_metric_learning/trainers/base_trainer.py#L9) (modified a bit for brevity)

```python
class Trainer:
  def __init__(self,models,
               loss_funcs,
               mining_funcs,
               iterations_per_epoch=None,
               data_device=None,
               loss_weights=None,
               sampler=None,
               collate_fn=None,
               lr_schedulers=None,
               gradient_clippers=None,
               data_and_label_getter=None,
               dataset_labels=None,
               end_of_iteration_hook=None,
               end_of_epoch_hook=None
      ):
    ...
```

of course there's no harm in writing this way, but doesn't it look too bulky? what if it grows even further? by the time you reach to actual logic of this method, the parameter list will already scare you off.  Also, say you've written this list somewhere else, why do you want to re-write here as well? `@use_kwargs` will make things a little cute for you :wink:

```python
class Trainer:
  _none_attrs="""iterations_per_epoch data_device loss_wights sampler 
                collate_fn lr_schedulers gradient_clippers data_and_labels_getter 
                dataset_labels end_of_iteration_hook end_of_epoch_hook""".split()

  @use_kwargs(_none_attrs)
  def __init__(self,models,loss_funcs,mining_funcs,**kwargs):
  ...
```

But what if they're not `None` and have their own default values :thinking: ? Borrowing this snippet from [keras](https://github.com/keras-team/keras/blob/1cf5218edb23e575a827ca4d849f1d52d21b4bb0/keras/preprocessing/image.py#L238) 

```python
class ImageDataGenerator(image.ImageDataGenerator):

    def __init__(self,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 zca_epsilon=1e-6,
                 rotation_range=0,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 brightness_range=None,
                 shear_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 preprocessing_function=None,
                 data_format='channels_last',
                 validation_split=0.0,
                 interpolation_order=1,
                 dtype='float32'):
       ...
        super(ImageDataGenerator, self).__init__(
            featurewise_center=featurewise_center,
            samplewise_center=samplewise_center,
            featurewise_std_normalization=featurewise_std_normalization,
            samplewise_std_normalization=samplewise_std_normalization,
            zca_whitening=zca_whitening,
            zca_epsilon=zca_epsilon,
            rotation_range=rotation_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            brightness_range=brightness_range,
            shear_range=shear_range,
            zoom_range=zoom_range,
            channel_shift_range=channel_shift_range,
            fill_mode=fill_mode,
            cval=cval,
            horizontal_flip=horizontal_flip,
            vertical_flip=vertical_flip,
            rescale=rescale,
            preprocessing_function=preprocessing_function,
            data_format=data_format,
            validation_split=validation_split,
            interpolation_order=interpolation_order,
            dtype=dtype)
```

:frowning: :cold_sweat: I'm not at all saying this bad, in fact, this follows the PEP8 coding standards; but the first question came into my mind was, what is `image.ImageDataGenerator` ? then I got to know, it came from another library called `keras_preprocessing`. So the next obvious question was, does that library has the exact same constructor, and indeed it [has](https://github.com/keras-team/keras-preprocessing/blob/371ca04391566d00d4fea4b347612b1efc146997/keras_preprocessing/image/image_data_generator.py#L254). Now, isn't it redundant code? and what if, you've some factory methods in the class that uses partial or exact same set of parameters? Can we do any better?

```python
class ImageDataGenerator2(IGenerator):
  _datagen_kwargs = dict(featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 zca_epsilon=1e-6,
                 rotation_range=0,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 brightness_range=None,
                 shear_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 preprocessing_function=None,
                 data_format='channels_last',
                 validation_split=0.0,
                 interpolation_order=1,
                 dtype='float32')
  
  @use_kwargs_dict(**_datagen_kwargs)
  def __init__(self,**kwargs):
        super(ImageDataGenerator, self).__init__(**kwargs)
        
  ...        
        
  _auglist = """brightness_range samplewise_center 
                samplewise_std_normalization zca_epsilon 
                zca_whitening""".split()
  @classmethod
  @use_kwargs_dict(**dict((k,_datagen_kwargs[k]) for k in _auglist))
  def augment(**kwargs):
        ...
```

At first, you might think what's the difference :thinking: we 're just writing it in a dictionary instead of method definition. But I see three advantages here:

1.  The `__init__` method is less scary and succinct.
2.  We could reuse those parameters. As in the example, I've created a static method `augment` which is using partial set of parameters from the `_datagen_dict`.
3. It became a central place for default arguments. Suppose, in future, we decided to change the `data_format=channel_first`; this will require you to change it from all the mentions of the same to make it consistent. Whereas, in this case, changing it in our default dictionary will be reflected in all the other places, making it less error prone.

{% include alert.html text="Note that, using **kwargs won't conceal the parameter list and you'll get all the parameters in type-completion  "%}

<img src="{{site.baseurl}}/images/meta-fastai/aug-hint.png">

## `@funcs_kwargs`

Now that we're able handle kwargs in a better way, why not make the constructor accept methods as a parameter. Let's define the default behavior for them, but let's also allow the users to change it as well and that too, **without inheritance**.  One reason to avoid inheritance is, doing so might break the dependency chain. We were not concerned about the types so far (as python is dynamic language) but now that I've shown you the perks of type-dispatch, we need to worry about it a little.

Again, we won't be polluting the argument list of `__init__`, instead we've a contrived way for doing so :wink:

```python
@funcs_kwargs
class DataBlock():
    "Generic container to quickly build `Datasets` and `DataLoaders`"
    get_x=get_items=splitter=get_y=None
    _methods = 'get_items splitter get_y get_x'.split()
    def __init__(self, blocks=None, dl_type=None, getters=None, n_inp=None, item_tfms=None, batch_tfms=None, **kwargs):
        ...
```

By adding `@funcs_kwargs` decorator to the class definition, you've enabled the users to modify any method you included in `_methods` list. For example, 

```python
dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),
                 get_items=get_image_files,
                 splitter=RandomSplitter(),
                 get_y=parent_label,
                 item_tfms=item_tfms,
                 batch_tfms=batch_tfms)
```

`parent_label` is a function (simply returns the parent folder name) which is assigned to `get_y`. `DataBlock` makes `get_y` a no operation function unless you provide your implementation for that. (`@funcs_kwargs` are not shown in the auto-completion)

# `@docs` in the source code :fire:

Writing documentation in the source code is absolutely rewarding, mainly, for two reasons.

1. It makes your source code readable
2. Documentation generator libraries such as [Sphinx](https://www.sphinx-doc.org/en/master/), [MkDocs](https://www.mkdocs.org/), [nbdev](https://nbdev.fast.ai/) will generate the html docs for you

But writing it within method/class definition itself will make it so not readable. Just look at this [Conv1D](https://github.com/keras-team/keras/blob/7a39b6c62d43c25472b2c2476bd2a8983ae4f682/keras/layers/convolutional.py#L234) class :no_mouth:,​ instead, grouping it at one place seems a better idea.

```python
@docs
class DataLoaders(GetAttr):
	 _default='train'
     _xtra=""   
	...
	 _docs=dict(__getitem__="Retrieve `DataLoader` at `i` (`0` is training, `1` is validation)", train="Training `DataLoader`",
	 		  valid="Validation `DataLoader`",
	 		  train_ds="Training `Dataset`",
	 		  valid_ds="Validation `Dataset`",
	 		  to="Use `device`",
	 		  cuda="Use the gpu if available",
	 		  cpu="Use the cpu",
	 		  new_empty="Create a new empty version of `self` with the same transforms",
	 		  from_dblock="Create a dataloaders from a given `dblock`")
```

# Get it easily (`GetAttr`)

A wrapper class sometimes make things redundant. `GettAttr` is meant to simplify this if you use the composed object frequently. Just specify that object as `_default` and no more redundancy.

```python
dls = dblock.dataloaders(...)

dls.train.n	
dls.train.vocab 
dls.train.tfms
dls.train.show_batch()

vs

dls.n	
dls.vocab 
dls.tfms
dls.show_batch()
```

But what if wrapper class also has a method/property with that name :thinking: ? then be specific about which properties you'd like to expose from the composed object.

```python
class OptimWrapper(_BaseOptimizer, GetAttr):
    _xtra=['zero_grad', 'step', 'st ate_dict', 'load_state_dict']
    _default='opt'
```

This wrapper is used to wrap PyTorch Optimizers and to add some utility functions associated with it. So the original optimizer is stored as `opt` in this class and we don't really want to access all attributes like `adam.opt.step`, `adam.opt.state_dict`. Thus, class inherits `GettAttr` alongside `_BaseOptimizer` (multiple inheritance is allowed in Python). The `_xtra` parameter will make sure only those attributes will be passed on to the `self`.



These are the perquisites of dynamic Python and `fastcore` shows how malleable the language is. As I said in the beginning, you can start using all of these in your project as `fastcore` is independent of any Deep Learning framework. Hope you enjoyed the article, this is my first attempt to write such a thorough post, so your suggestions are most welcome!

# References

- [fastcore](http://fastcore.fast.ai/index.html) 
- [fastai2]( https://github.com/fastai/fastai2)
- [fastai  paper](https://arxiv.org/abs/2002.04688) 

