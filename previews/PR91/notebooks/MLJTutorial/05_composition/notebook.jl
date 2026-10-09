# # Tutorial 5. Advanced Model Composition

# !!! warning
#
#     This is an advanced MLJ feature. For extensive documentation and further examples,
#     see [the manual](https://juliaai.github.io/MLJ.jl/dev/learning_networks/).

# > **Goals:**
# > 1. Learn how to build a prototypes of a composite model, called *learning networks*
# > 2. Learn how to "export" a learning network as a new stand-alone model type

# To run the code in this tutorial in a live Julia session, first follow the instructions
# given [here](@ref instructions).

# Pipelines are great for composing models in an unbranching sequence. Another built-in
# type of model composition is a model *stack*; see
# [here](https://juliaai.github.io/MLJ.jl/dev/model_stacking/#Model-Stacking) for
# details. For other more complicated model compositions you'll want to use MLJ's generic
# model composition syntax.

# There are two main steps:

# - **Prototype** the composite model by building a *learning
#   network*, which can be tested on some (dummy) data as you build
#   it.

# - **Export** the learning network as a new stand-alone model type.

# Like pipeline models, instances of the exported model type behave
# like any other model (and are not bound to any data, until you wrap
# them in a machine).


# ### Building a pipeline using the generic composition syntax

using MLJ
MLJ.color_off() #hide
LogisticClassifier = @load LogisticClassifier pkg=MLJLinearModels

# To warm up, we'll build a learning network to replace this basic pipeline model:

pipe = Standardizer |> LogisticClassifier(lambda=0.001)
nothing #hide

# Here's some dummy data we'll be using to test our learning network:

X, y = make_blobs(5, 3)
pretty(X)

# **Step 0** - Proceed as if you were combining the models "by hand", using all the data
# available for training, transformation and prediction:

standardizer = Standardizer();
linear = LogisticClassifier(lambda=0.001);

mach1 = machine(standardizer, X);
fit!(mach1);
Xstand = transform(mach1, X);

mach2 = machine(linear, Xstand, y);
fit!(mach2);
yhat = predict(mach2, Xstand)

# **Step 1** - Edit your code as follows:

# - wrap the data in `Source` nodes

# - delete the `fit!` calls

X = source(X)  # or X = source() if not testing
y = source(y)  # or y = source()

standardizer = Standardizer();
linear = LogisticClassifier(lambda=0.001);

mach1 = machine(standardizer, X);
Xstand = transform(mach1, X);

mach2 = machine(linear, Xstand, y);
yhat = predict(mach2, Xstand)

# Now `X`, `y`, `Xstand` and `yhat` are *nodes* ("variables" or
# "dynamic data") instead of data. All training, predicting and
# transforming is now executed lazily, whenever we `fit!` one of these
# nodes. We *call* a node to retrieve the data it represents in the
# original manual workflow.

fit!(Xstand)
Xstand() |> pretty

#-

fit!(yhat); # training is smart and so `Standardizer` is not retrained

#-

yhat()

# The node `yhat` is the "descendant" (in an associated DAG we have
# defined) of a unique source node:

origins(yhat)

#-

# The data at the source node is replaced by `Xnew` to obtain a
# new prediction when we call `yhat` like this:

Xnew, _ = make_blobs(2, 3);
yhat(Xnew)


# **Step 2** - Export the learning network as a new stand-alone model type

# We start by defining a new model type for our composite. We subtype
# `ProbabilisticNetworkComposite` because our composite is to be a probabilistic
# predictor. If it were a deterministic predictor, we would use
# `DeterministicNetworkComposite` instead. There is also a `UnsupervisedNetworkComposite`
# for transformers.

mutable struct YourPipe <: ProbabilisticNetworkComposite
    standardizer
    classifier
end

# Next, we make our learning network above generic by substituting each model instance
# with the corresponding symbol representing a property (field) of the new model struct:

mach1 = machine(:standardizer, X);
Xstand = transform(mach1, X);

mach2 = machine(:classifier, Xstand, y);
yhat = predict(mach2, Xstand)

# Incidentally, this network can be used as before except we must provide an instance of
# `YourPipe` in our `fit!` calls, to indicate which models replace the symbols:

your_pipe = YourPipe(standardizer, linear)
fit!(yhat, composite=your_pipe);

# In this case `:standardizer` is being substituted by `standardizer` and `:classifier` by
# `linear` in training.

# The final step is to wrap our learning network code in a method called `prefit`
# dispatched on `YourPipe`. This method returns a "learning network interface" which is a
# named tuple telling the method which node of the network returns predictions for the
# composite model.

import MLJ.MLJBase
function MLJBase.prefit(composite::YourPipe, verbosity, X, y)
    ## the learning network from above:
    X = source(X)
    y = source(y)
    mach1 = machine(:standardizer, X);
    Xstand = transform(mach1, X);
    mach2 = machine(:classifier, Xstand, y);
    yhat = predict(mach2, Xstand)

    verbosity > 0 && @info "I'm a noisy fellow!"

    ## return "learning network interface":
    return (; predict=yhat)
end

# Instantiating and training on some new data:

pipe = YourPipe(standardizer, linear)
X, y = @load_iris;   # built-in data set
mach = machine(pipe, X, y)
fit!(mach);

# The learned parameters and report (where non-empty) for each component model are
# accessible:

fitted_params(mach).classifier.coefs

#-

report(mach).standardizer

# Component models can be swapped out for new ones:

pipe.classifier = ConstantClassifier()
fit!(mach)
fitted_params(mach).classifier.target_distribution


# ### A composite model to average two regressor predictors

# Next, we define a composite model that:

# - standardizes the input data
# - learns and applies a Box-Cox transformation to the target variable
# - averages the predictions of two supervised learning models - a ridge regressor and a
#   random forest regressor - using a simple average
# - applies the *inverse* Box-Cox transformation to this blended prediction

# We'll start with a learning network, with source nodes bound to some dummy test data:

RandomForestRegressor = @load RandomForestRegressor pkg=DecisionTree
RidgeRegressor = @load RidgeRegressor pkg=MLJLinearModels

# **Input layer with dummy data:**

X, y = make_regression()
y = abs.(y)
X = source(X)
y = source(y)

# **First layer and target transformation:**

standardizer = Standardizer()
mach1 = machine(standardizer, X)
W = MLJ.transform(mach1, X)

box_model = UnivariateBoxCoxTransformer()
mach2 = machine(box_model, y)
z = MLJ.transform(mach2, y)

# **Second layer:**

regressor1 = RidgeRegressor(lambda=0.1)
mach3 = machine(regressor1, W, z)

regressor2 = RandomForestRegressor(n_trees=50)
mach4 = machine(regressor2, W, z)

zhat = 0.5*predict(mach3, W) + 0.5*predict(mach4, W)

# **Output:**

yhat = inverse_transform(mach2, zhat)

# Let's test this learning network (always a good idea!):

fit!(yhat)
yhat(rows=1:3)

# Now for the new model type:

mutable struct CompositeModel <: DeterministicNetworkComposite
    standardizer
    box_cox
    regressor1
    regressor2
end

# And the `prefit` function wrapping our learning network code, with model substitutions

function MLJBase.prefit(composite::CompositeModel, verbosity, X, y)
    X = source(X)
    y = source(y)

    ## First layer and target transformation:
    mach1 = machine(:standardizer, X)
    W = MLJ.transform(mach1, X)
    mach2 = machine(:box_cox, y)
    z = MLJ.transform(mach2, y)

    ## Second layer:
    mach3 = machine(:regressor1, W, z)
    mach4 = machine(:regressor2, W, z)
    zhat = 0.5*predict(mach3, W) + 0.5*predict(mach4, W)

    ## Output:
    yhat = inverse_transform(mach2, zhat)

    return (; predict=yhat)
end

# We instantiate the new model type and try it out on some new data:

composite = CompositeModel(standardizer, box_model, regressor1, regressor2)
X, y = @load_boston
evaluate(composite, X, y; resampling=CV(nfolds=6, shuffle=true), measures=[rms, mae])


# ### Tutorial 5 Resources
#
# - From the MLJ manual:
#    - [Learning Networks](https://juliaai.github.io/MLJ.jl/stable/composing_models/#Learning-Networks-1)
# - From Data Science Tutorials:
#     - [Model ensembles via learning networks](https://juliaai.github.io/DataScienceTutorials.jl/advanced/ensembles-3/)
#     - [Model stacking via learning networks](https://juliaai.github.io/DataScienceTutorials.jl/advanced/stacking/)
