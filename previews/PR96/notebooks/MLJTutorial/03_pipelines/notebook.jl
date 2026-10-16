# # Tutorial 3. Transformers and Pipelines

# > **Goals:** Learn how to:
# > 1. Apply common data preprocessing tasks using an MLJ **transformer**.
# > 2. Combine a sequence of transformers with a supervised model in a single standalone model called a **pipelines**
# > 3. Wrap a supervised model in transformations/inverse transformations of the target variable

# To run the code in this tutorial in a live Julia session, first follow the instructions
# given [here](@ref instructions).


# ### Transformers

# Unsupervised models, which receive no target `y` during training, always have a
# `transform` operation. They sometimes also support an `inverse_transform` operation,
# with obvious meaning, and sometimes support a `predict` operation (see the clustering
# example discussed
# [here](https://juliaai.github.io/MLJ.jl/dev/transformers/#Transformers-that-also-predict-1)).
# Otherwise, they are handled much like supervised models.

# Here's a simple standardization example:

using MLJ

x = rand(100);
@show mean(x) std(x);

#-

model = Standardizer() # a built-in model
mach = machine(model, x)
fit!(mach)
xhat = transform(mach, x);
@show mean(xhat) std(xhat);

# This particular model has an `inverse_transform`:

inverse_transform(mach, xhat) ≈ x


# ### Re-encoding the King County House data as continuous

# For further illustrations of transformers, let's re-encode *all* of the King County
# House input features (see Exercise 3) into a set of `Continuous` features. We do this
# with the `ContinuousEncoder` model, which, by default, will:

# - one-hot encode all `Multiclass` features
# - coerce all `OrderedFactor` features to `Continuous` ones
# - coerce all `Count` features to `Continuous` ones (there aren't any)
# - drop any remaining non-Continuous features (none of these either)

# First, we reload the data and fix the scitypes (Exercise 3):

import Downloads, CSV
import DataFrames
url = "https://raw.githubusercontent.com/ablaom/"*
    "MachineLearningInJulia2020/for-MLJ-version-0.16/"*
    "data/house.csv"
csv_file = Downloads.download(url)
house = CSV.read(csv_file, DataFrames.DataFrame)
coerce!(house, autotype(house))
coerce!(house, Count => Continuous, :zipcode => Multiclass)
schema(house)

#-

y, X = unpack(house, ==(:price), rng=123);

# Instantiate the unsupervised model (transformer):

encoder = ContinuousEncoder() # a built-in model; no need to @load it

# Bind the model to the data and fit!

mach = machine(encoder, X) |> fit!;

# Transform and inspect the result:

Xcont = transform(mach, X);
schema(Xcont)


# ### More transformers

# Here's how to list all of MLJ's unsupervised models:

models(m->!m.is_supervised)

# Some commonly used ones are built-in (do not require `@load`ing):

# model type                  | does what?
# ----------------------------|----------------------------------------------
# ContinuousEncoder | transform input table to a table of `Continuous` features (see above)
# FeatureSelector | retain or dump selected features
# FillImputer | impute missing values
# OneHotEncoder | one-hot encoder `Multiclass` (and optionally `OrderedFactor`) features
# Standardizer | standardize (whiten) a vector or all `Continuous` features of a table
# UnivariateBoxCoxTransformer | apply a learned Box-Cox transformation to a vector
# UnivariateDiscretizer | discretize a `Continuous` vector, and hence render its elscitypw `OrderedFactor`


# In addition to "dynamic" transformers (ones that learn something
# from the data and must be `fit!`) users can wrap ordinary functions
# as transformers, and such *static* transformers can depend on
# parameters, like the dynamic ones. See
# [here](https://juliaai.github.io/MLJ.jl/dev/transformers/#Static-transformers-1)
# for how to define your own static transformers.


# ### Pipelines

length(schema(Xcont).names)

# Let's suppose that additionally we'd like to reduce the dimension of
# our data.  A model that will do this is `PCA` from
# `MultivariateStats.jl`:

PCA = @load PCA
reducer = PCA()

# Now, rather simply repeating the work-flow above, applying the new
# transformation to `Xcont`, we can combine both the encoding and the
# dimension-reducing models into a single model, known as a
# *pipeline*. While MLJ offers a powerful interface for composing
# models in a variety of ways, we'll stick to these simplest class of
# composite models for now. The simplest way to construct a pipeline
# is using the Julia's `|>` syntax:

pipe = encoder |> reducer

# Notice that the model `pipe` has other models as hyperparameters
# (with names automatically generated based on the mode type
# name). The hyperparameters of the component models are are now
# *nested*, but we can still access them:

@show pipe.pca.variance_ratio

#-

pipe.pca.variance_ratio = 0.9995

# The pipeline model behaves like any other transformer:

mach = machine(pipe, X)
fit!(mach)
Xsmall = transform(mach, X)
schema(Xsmall)

# Want to combine this pre-processing with ridge regression?

RidgeRegressor = @load RidgeRegressor pkg=MLJLinearModels
rgs = RidgeRegressor()
pipe2 = pipe |> rgs

# Now our pipeline is a supervised model, instead of a transformer,
# whose performance we can evaluate:

mach = machine(pipe2, X, y)
evaluate!(mach, measure=mae, resampling=Holdout()) # `CV(nfolds=6)` is `resampling` default


# ### Training of composite models is "smart"

# Now notice what happens if we train on all the data, then change a
# regressor hyperparameter and retrain:

fit!(mach);

#-

pipe2.ridge_regressor.lambda = 0.1
fit!(mach);

# Second time only the ridge regressor is retrained!

# Mutate a hyperparameter of the `PCA` model and every model except
# the `ContinuousEncoder` (which comes before it will be retrained):

pipe2.pca.variance_ratio = 0.9999
fit!(mach);


# ### Inspecting composite models

# The dot syntax used above to change the values of *nested*
# hyperparameters is also useful when inspecting the learned
# parameters and report generated when training a composite model:

fitted_params(mach).ridge_regressor

#-

report(mach).pca


# ### Incorporating target transformations

# Next, suppose that instead of using the raw `:price` as the training target, we want to
# use the log-price (a common practice in dealing with house price data). However, suppose
# that we still want to report final *predictions* on the original linear scale (and use
# these for evaluation purposes). Then we wrap our supervised model using
# `TransformedTargetModel`, which has two keyword arguments `transformer` and `inverse`.

rgs_log = TransformedTargetModel(rgs, transformer=y->log.(y), inverse=z->exp.(z))

# And here is a revised pipeline model:

pipe3 = pipe |> rgs_log
mach = machine(pipe3, X, y)
evaluate!(mach, measure=mae)

# MLJ will also allow you to insert *learned* target transformations. For example, we
# might want to apply `Standardizer()` to the target, to standardize it, or
# `UnivariateBoxCoxTransformer()` to make it look Gaussian. Then instead of specifying a
# *function* for `target`, we specify a unsupervised *model* (or model type). One does not
# specify `inverse` because only models implementing `inverse_transform` are allowed.

# Let's see which of these two options results in a better outcome:

box = UnivariateBoxCoxTransformer(n=20)
stand = Standardizer()

rgs_box = TransformedTargetModel(rgs, transformer=box)
pipe4 = pipe |> rgs_box
mach = machine(pipe4, X, y)
evaluate!(mach, measure=mae)

#-

pipe4.transformed_target_model_deterministic.transformer = stand
evaluate!(mach, measure=mae)


# ### Tutorial 3 Resources

# - From the MLJ manual:
#     - [Transformers and other unsupervised models](https://juliaai.github.io/MLJ.jl/dev/transformers/)
#     - [Linear pipelines](https://juliaai.github.io/MLJ.jl/dev/linear_pipelines/#Linear-Pipelines)
# - From Data Science Tutorials:
#     - [Composing models](https://juliaai.github.io/DataScienceTutorials.jl/getting-started/composing-models/)


# ### Tutorial 3 Exercises

# #### Exercise 7

# Consider again the Horse Colic classification problem considered in
# Exercise 6, but with all features, `Finite` and `Infinite`:

import Downloads
import CSV
url = "https://raw.githubusercontent.com/ablaom/"*
    "MachineLearningInJulia2020/"*
    "for-MLJ-version-0.16/data/horse.csv"
csv_file = Downloads.download(url)
horse = CSV.read(csv_file, DataFrames.DataFrame)
coerce!(horse, autotype(horse));
coerce!(horse, Count => Continuous);
coerce!(
    horse,
    :surgery               => Multiclass,
    :age                   => Multiclass,
    :mucous_membranes      => Multiclass,
    :capillary_refill_time => Multiclass,
    :outcome               => Multiclass,
    :cp_data               => Multiclass,
)
schema(X)

# (a) Define a pipeline that:
# - uses `Standardizer` to ensure that features that are already
#   continuous are centered at zero and have unit variance
# - re-encodes the full set of features as `Continuous`, using
#   `ContinuousEncoder`
# - uses the `KMeans` clustering model from `Clustering.jl`
#   to reduce the dimension of the feature space to `k=10`.
# - trains a `EvoTreeClassifier` (a gradient tree boosting
#   algorithm in `EvoTrees.jl`) on the reduced data, using
#   `nrounds=50` and default values for the other
#    hyperparameters

# (b) Evaluate the pipeline on all data, using 6-fold cross-validation
# and `cross_entropy` loss.

# (c) Plot a learning curve which examines the effect on this loss
# as the tree booster parameter `max_depth` varies from 2 to 10.
