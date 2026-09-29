```@meta
EditURL = "notebook.jl"
```

# Tutorial 5. Advanced Model Composition

!!! warning

    This is an advanced MLJ feature. For extensive documentation and further examples,
    see [the manual](https://juliaai.github.io/MLJ.jl/dev/learning_networks/).

> **Goals:**
> 1. Learn how to build a prototypes of a composite model, called *learning networks*
> 2. Learn how to "export" a learning network as a new stand-alone model type

To run the code in this tutorial in a live Julia session, first follow the instructions
given [here](@ref instructions).

Pipelines are great for composing models in an unbranching sequence. Another built-in
type of model composition is a model *stack*; see
[here](https://juliaai.github.io/MLJ.jl/dev/model_stacking/#Model-Stacking) for
details. For other more complicated model compositions you'll want to use MLJ's generic
model composition syntax.

There are two main steps:

- **Prototype** the composite model by building a *learning
  network*, which can be tested on some (dummy) data as you build
  it.

- **Export** the learning network as a new stand-alone model type.

Like pipeline models, instances of the exported model type behave
like any other model (and are not bound to any data, until you wrap
them in a machine).

### Building a pipeline using the generic composition syntax

````@julia
using MLJ
LogisticClassifier = @load LogisticClassifier pkg=MLJLinearModels
````

````
MLJLinearModels.LogisticClassifier
````

To warm up, we'll build a learning network to replace this basic pipeline model:

````@julia
pipe = Standardizer |> LogisticClassifier(lambda=0.001)
````

Here's some dummy data we'll be using to test our learning network:

````@julia
X, y = make_blobs(5, 3)
pretty(X)
````

````
┌────────────┬────────────┬────────────┐
│ x1         │ x2         │ x3         │
│ Float64    │ Float64    │ Float64    │
│ Continuous │ Continuous │ Continuous │
├────────────┼────────────┼────────────┤
│ 14.9681    │ -1.47137   │ 12.1949    │
│ -3.26626   │ -0.710617  │ 7.36507    │
│ -4.17429   │ -0.0882215 │ 9.87907    │
│ 6.00034    │ -2.41859   │ 14.9897    │
│ -4.38537   │ -0.187352  │ 7.22233    │
└────────────┴────────────┴────────────┘

````

**Step 0** - Proceed as if you were combining the models "by hand", using all the data
available for training, transformation and prediction:

````@julia
standardizer = Standardizer();
linear = LogisticClassifier(lambda=0.001);

mach1 = machine(standardizer, X);
fit!(mach1);
Xstand = transform(mach1, X);

mach2 = machine(linear, Xstand, y);
fit!(mach2);
yhat = predict(mach2, Xstand)
````

````
5-element CategoricalDistributions.UnivariateFiniteVector{ScientificTypesBase.Multiclass{3}, Int64, UInt32, Float64}:
 UnivariateFinite{ScientificTypesBase.Multiclass{3}}(1=>0.993, 2=>0.00372, 3=>0.00319)
 UnivariateFinite{ScientificTypesBase.Multiclass{3}}(1=>0.000405, 2=>0.000195, 3=>0.999)
 UnivariateFinite{ScientificTypesBase.Multiclass{3}}(1=>0.000309, 2=>0.000249, 3=>0.999)
 UnivariateFinite{ScientificTypesBase.Multiclass{3}}(1=>0.00752, 2=>0.992, 3=>0.000947)
 UnivariateFinite{ScientificTypesBase.Multiclass{3}}(1=>0.00012, 2=>1.96e-5, 3=>1.0)
````

**Step 1** - Edit your code as follows:

- wrap the data in `Source` nodes

- delete the `fit!` calls

````@julia
X = source(X)  # or X = source() if not testing
y = source(y)  # or y = source()

standardizer = Standardizer();
linear = LogisticClassifier(lambda=0.001);

mach1 = machine(standardizer, X);
Xstand = transform(mach1, X);

mach2 = machine(linear, Xstand, y);
yhat = predict(mach2, Xstand)
````

````
Node @440 → LogisticClassifier(…)
  args:
    1:	Node @909 → Standardizer(…)
  formula:
    predict(
      machine(LogisticClassifier(lambda = 0.001, …), …), 
      transform(
        machine(Standardizer(features = Symbol[], …), …), 
        Source @128,
      ),
    )
````

Now `X`, `y`, `Xstand` and `yhat` are *nodes* ("variables" or
"dynamic data") instead of data. All training, predicting and
transforming is now executed lazily, whenever we `fit!` one of these
nodes. We *call* a node to retrieve the data it represents in the
original manual workflow.

````@julia
fit!(Xstand)
Xstand() |> pretty
````

````
[ Info: Training machine(Standardizer(features = Symbol[], …), …).
┌────────────┬────────────┬────────────┐
│ x1         │ x2         │ x3         │
│ Float64    │ Float64    │ Float64    │
│ Continuous │ Continuous │ Continuous │
├────────────┼────────────┼────────────┤
│ 1.54143    │ -0.508564  │ 0.563239   │
│ -0.597675  │ 0.271238   │ -0.895631  │
│ -0.704197  │ 0.909215   │ -0.136268  │
│ 0.489404   │ -1.47949   │ 1.40741    │
│ -0.728959  │ 0.807603   │ -0.938747  │
└────────────┴────────────┴────────────┘

````

````@julia
fit!(yhat); # training is smart and so `Standardizer` is not retrained
````

````
[ Info: Not retraining machine(Standardizer(features = Symbol[], …), …). Use `force=true` to force.
[ Info: Training machine(LogisticClassifier(lambda = 0.001, …), …).
┌ Info: Solver: MLJLinearModels.LBFGS{Optim.Options{Float64, Nothing}, @NamedTuple{}}
│   optim_options: Optim.Options{Float64, Nothing}
└   lbfgs_options: @NamedTuple{} NamedTuple()

````

````@julia
yhat()
````

````
5-element CategoricalDistributions.UnivariateFiniteVector{ScientificTypesBase.Multiclass{3}, Int64, UInt32, Float64}:
 UnivariateFinite{ScientificTypesBase.Multiclass{3}}(1=>0.993, 2=>0.00372, 3=>0.00319)
 UnivariateFinite{ScientificTypesBase.Multiclass{3}}(1=>0.000405, 2=>0.000195, 3=>0.999)
 UnivariateFinite{ScientificTypesBase.Multiclass{3}}(1=>0.000309, 2=>0.000249, 3=>0.999)
 UnivariateFinite{ScientificTypesBase.Multiclass{3}}(1=>0.00752, 2=>0.992, 3=>0.000947)
 UnivariateFinite{ScientificTypesBase.Multiclass{3}}(1=>0.00012, 2=>1.96e-5, 3=>1.0)
````

The node `yhat` is the "descendant" (in an associated DAG we have
defined) of a unique source node:

````@julia
origins(yhat)
````

````
1-element Vector{MLJBase.Source}:
 Source @128 ⏎ `ScientificTypesBase.Table{AbstractVector{ScientificTypesBase.Continuous}}`
````

The data at the source node is replaced by `Xnew` to obtain a
new prediction when we call `yhat` like this:

````@julia
Xnew, _ = make_blobs(2, 3);
yhat(Xnew)
````

````
2-element CategoricalDistributions.UnivariateFiniteVector{ScientificTypesBase.Multiclass{3}, Int64, UInt32, Float64}:
 UnivariateFinite{ScientificTypesBase.Multiclass{3}}(1=>1.13e-8, 2=>1.0, 3=>6.09e-8)
 UnivariateFinite{ScientificTypesBase.Multiclass{3}}(1=>2.89e-16, 2=>1.0, 3=>4.36e-13)
````

**Step 2** - Export the learning network as a new stand-alone model type

We start by defining a new model type for our composite. We subtype
`ProbabilisticNetworkComposite` because our composite is to be a probabilistic
predictor. If it were a deterministic predictor, we would use
`DeterministicNetworkComposite` instead. There is also a `UnsupervisedNetworkComposite`
for transformers.

````@julia
mutable struct YourPipe <: ProbabilisticNetworkComposite
    standardizer
    classifier
end
````

Next, we make our learning network above generic by substituting each model instance
with the corresponding symbol representing a property (field) of the new model struct:

````@julia
mach1 = machine(:standardizer, X);
Xstand = transform(mach1, X);

mach2 = machine(:classifier, Xstand, y);
yhat = predict(mach2, Xstand)
````

````
Node @530 → :classifier
  args:
    1:	Node @804 → :standardizer
  formula:
    predict(
      machine(:classifier, …), 
      transform(
        machine(:standardizer, …), 
        Source @128,
      ),
    )
````

Incidentally, this network can be used as before except we must provide an instance of
`YourPipe` in our `fit!` calls, to indicate which models replace the symbols:

````@julia
your_pipe = YourPipe(standardizer, linear)
fit!(yhat, composite=your_pipe);
````

````
[ Info: Training machine(:standardizer, …).
[ Info: Training machine(:classifier, …).
┌ Info: Solver: MLJLinearModels.LBFGS{Optim.Options{Float64, Nothing}, @NamedTuple{}}
│   optim_options: Optim.Options{Float64, Nothing}
└   lbfgs_options: @NamedTuple{} NamedTuple()

````

In this case `:standardizer` is being substituted by `standardizer` and `:classifier` by
`linear` in training.

The final step is to wrap our learning network code in a method called `prefit`
dispatched on `YourPipe`. This method returns a "learning network interface" which is a
named tuple telling the method which node of the network returns predictions for the
composite model.

````@julia
import MLJ.MLJBase
function MLJBase.prefit(composite::YourPipe, verbosity, X, y)
    # the learning network from above:
    X = source(X)
    y = source(y)
    mach1 = machine(:standardizer, X);
    Xstand = transform(mach1, X);
    mach2 = machine(:classifier, Xstand, y);
    yhat = predict(mach2, Xstand)

    verbosity > 0 && @info "I'm a noisy fellow!"

    # return "learning network interface":
    return (; predict=yhat)
end
````

Instantiating and training on some new data:

````@julia
pipe = YourPipe(standardizer, linear)
X, y = @load_iris;   # built-in data set
mach = machine(pipe, X, y)
fit!(mach);
````

````
[ Info: Training machine(YourPipe(standardizer = Standardizer(features = Symbol[], …), …), …).
[ Info: I'm a noisy fellow!
[ Info: Training machine(:standardizer, …).
[ Info: Training machine(:classifier, …).
┌ Info: Solver: MLJLinearModels.LBFGS{Optim.Options{Float64, Nothing}, @NamedTuple{}}
│   optim_options: Optim.Options{Float64, Nothing}
└   lbfgs_options: @NamedTuple{} NamedTuple()

````

The learned parameters and report (where non-empty) for each component model are
accessible:

````@julia
fitted_params(mach).classifier.coefs
````

````
4-element Vector{Pair{Symbol, SubArray{Float64, 1, Matrix{Float64}, Tuple{Int64, Base.Slice{Base.OneTo{Int64}}}, true}}}:
 :sepal_length => [-2.2884323728871574, 1.4607388496829223, 0.8276935232042454]
  :sepal_width => [2.5849165395350733, -0.6646830918859379, -1.9202334476491332]
 :petal_length => [-3.5313085896005036, -1.4061808524064117, 4.937489442006905]
  :petal_width => [-3.448007649885748, -1.6977190775004702, 5.145726727386221]
````

````@julia
report(mach).standardizer
````

````
(features_fit = [:sepal_length, :sepal_width, :petal_length, :petal_width],)
````

Component models can be swapped out for new ones:

````@julia
pipe.classifier = ConstantClassifier()
fit!(mach)
fitted_params(mach).classifier.target_distribution
````

````
(CategoricalArrays.CategoricalValue{String, UInt32}[CategoricalValue(CategoricalArrays.CategoricalPool{String, UInt32}(["setosa", "versicolor", "virginica"]), 1), CategoricalValue(CategoricalArrays.CategoricalPool{String, UInt32}(["setosa", "versicolor", "virginica"]), 2), CategoricalValue(CategoricalArrays.CategoricalPool{String, UInt32}(["setosa", "versicolor", "virginica"]), 3)], [0.3333333333333333 0.3333333333333333 0.3333333333333333])
````

### A composite model to average two regressor predictors

Next, we define a composite model that:

- standardizes the input data
- learns and applies a Box-Cox transformation to the target variable
- averages the predictions of two supervised learning models - a ridge regressor and a
  random forest regressor - using a simple average
- applies the *inverse* Box-Cox transformation to this blended prediction

We'll start with a learning network, with source nodes bound to some dummy test data:

````@julia
RandomForestRegressor = @load RandomForestRegressor pkg=DecisionTree
RidgeRegressor = @load RidgeRegressor pkg=MLJLinearModels
````

````
MLJLinearModels.RidgeRegressor
````

**Input layer with dummy data:**

````@julia
X, y = make_regression()
y = abs.(y)
X = source(X)
y = source(y)
````

````
Source @555 ⏎ `AbstractVector{ScientificTypesBase.Continuous}`
````

**First layer and target transformation:**

````@julia
standardizer = Standardizer()
mach1 = machine(standardizer, X)
W = MLJ.transform(mach1, X)

box_model = UnivariateBoxCoxTransformer()
mach2 = machine(box_model, y)
z = MLJ.transform(mach2, y)
````

````
Node @022 → UnivariateBoxCoxTransformer(…)
  args:
    1:	Source @555
  formula:
    transform(
      machine(UnivariateBoxCoxTransformer(n = 171, …), …), 
      Source @555,
    )
````

**Second layer:**

````@julia
regressor1 = RidgeRegressor(lambda=0.1)
mach3 = machine(regressor1, W, z)

regressor2 = RandomForestRegressor(n_trees=50)
mach4 = machine(regressor2, W, z)

zhat = 0.5*predict(mach3, W) + 0.5*predict(mach4, W)
````

````
Node @135
  args:
    1:	Node @083
    2:	Node @797
  formula:
    +(
     var"#*##0#*##1"(
       predict(
         machine(RidgeRegressor(lambda = 0.1, …), …), 
         transform(
           machine(Standardizer(features = Symbol[], …), …), 
           Source @100,
         ),
       ),
     ),
     var"#*##0#*##1"(
       predict(
         machine(RandomForestRegressor(max_depth = -1, …), …), 
         transform(
           machine(Standardizer(features = Symbol[], …), …), 
           Source @100,
         ),
       ),
     ),
    )
````

**Output:**

````@julia
yhat = inverse_transform(mach2, zhat)
````

````
Node @539 → UnivariateBoxCoxTransformer(…)
  args:
    1:	Node @135
  formula:
    inverse_transform(
      machine(UnivariateBoxCoxTransformer(n = 171, …), …), 
      +(
       var"#*##0#*##1"(
         predict(
           machine(RidgeRegressor(lambda = 0.1, …), …), 
           transform(
             machine(Standardizer(features = Symbol[], …), …), 
             Source @100,
           ),
         ),
       ),
       var"#*##0#*##1"(
         predict(
           machine(RandomForestRegressor(max_depth = -1, …), …), 
           transform(
             machine(Standardizer(features = Symbol[], …), …), 
             Source @100,
           ),
         ),
       ),
      ),
    )
````

Let's test this learning network (always a good idea!):

````@julia
fit!(yhat)
yhat(rows=1:3)
````

````
3-element Vector{Float64}:
 4.1049434111127105
 4.674958202525163
 4.112097582032496
````

Now for the new model type:

````@julia
mutable struct CompositeModel <: DeterministicNetworkComposite
    standardizer
    box_cox
    regressor1
    regressor2
end
````

And the `prefit` function wrapping our learning network code, with model substitutions

````@julia
function MLJBase.prefit(composite::CompositeModel, verbosity, X, y)
    X = source(X)
    y = source(y)

    # First layer and target transformation:
    mach1 = machine(:standardizer, X)
    W = MLJ.transform(mach1, X)
    mach2 = machine(:box_cox, y)
    z = MLJ.transform(mach2, y)

    # Second layer:
    mach3 = machine(:regressor1, W, z)
    mach4 = machine(:regressor2, W, z)
    zhat = 0.5*predict(mach3, W) + 0.5*predict(mach4, W)

    # Output:
    yhat = inverse_transform(mach2, zhat)

    return (; predict=yhat)
end
````

We instantiate the new model type and try it out on some new data:

````@julia
composite = CompositeModel(standardizer, box_model, regressor1, regressor2)
X, y = @load_boston
evaluate(composite, X, y; resampling=CV(nfolds=6, shuffle=true), measures=[rms, mae])
````

````
PerformanceEvaluation object with these fields:
  model, tag, measure, operation,
  measurement (per-fold aggregate), uncertainty_radius_95 (1.96*SE),
  per_fold, per_observation,
  fitted_params_per_fold, report_per_fold,
  train_test_rows, resampling, repeats
Tag: CompositeModel-983
Extract:
┌───┬────────────────────────┬───────────┬─────────────┬─────────┐
│   │ measure                │ operation │ measurement │ 1.96*SE │
├───┼────────────────────────┼───────────┼─────────────┼─────────┤
│ A │ RootMeanSquaredError() │ predict   │ 3.87        │ 0.46    │
│ B │ LPLoss(                │ predict   │ 2.454       │ 0.04    │
│   │   p = 1)               │           │             │         │
└───┴────────────────────────┴───────────┴─────────────┴─────────┘
┌───┬──────────────────────────────────────┐
│   │ per_fold                             │
├───┼──────────────────────────────────────┤
│ A │ [4.45, 4.28, 3.41, 3.3, 4.21, 3.37]  │
│ B │ [2.45, 2.46, 2.46, 2.38, 2.53, 2.45] │
└───┴──────────────────────────────────────┘
Apply `describe` to this result for a named tuple summary.
````

### Tutorial 5 Resources

- From the MLJ manual:
   - [Learning Networks](https://juliaai.github.io/MLJ.jl/stable/composing_models/#Learning-Networks-1)
- From Data Science Tutorials:
    - [Model ensembles via learning networks](https://juliaai.github.io/DataScienceTutorials.jl/advanced/ensembles-3/)
    - [Model stacking via learning networks](https://juliaai.github.io/DataScienceTutorials.jl/advanced/stacking/)

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

