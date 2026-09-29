```@meta
EditURL = "notebook.jl"
```

# Lesson 2. Model Composition

Notebook supporting the video series "Using MLJ".

[Slides](slides.pdf) from the video.

To run the code in this tutorial in a live Julia session, first follow the instructions
given [here](@ref instructions).

````@julia
using MLJ
````

Load some model code:

````@julia
RidgeRegressor = @load RidgeRegressor pkg=MLJLinearModels
````

````
MLJLinearModels.RidgeRegressor
````

Load some data and inspect schema:

````@julia
data = load_reduced_ames();
schema(data)
````

````
┌──────────────┬───────────────────┬──────────────────────────────────┐
│ names        │ scitypes          │ types                            │
├──────────────┼───────────────────┼──────────────────────────────────┤
│ target       │ Continuous        │ Float64                          │
│ OverallQual  │ OrderedFactor{10} │ CategoricalValue{Int64, UInt32}  │
│ GrLivArea    │ Continuous        │ Float64                          │
│ Neighborhood │ Multiclass{25}    │ CategoricalValue{String, UInt32} │
│ x1stFlrSF    │ Continuous        │ Float64                          │
│ TotalBsmtSF  │ Continuous        │ Float64                          │
│ BsmtFinSF1   │ Continuous        │ Float64                          │
│ LotArea      │ Continuous        │ Float64                          │
│ GarageCars   │ Count             │ Int64                            │
│ MSSubClass   │ Multiclass{15}    │ CategoricalValue{String, UInt32} │
│ GarageArea   │ Continuous        │ Float64                          │
│ YearRemodAdd │ Count             │ Int64                            │
│ YearBuilt    │ Count             │ Int64                            │
└──────────────┴───────────────────┴──────────────────────────────────┘

````

Horizontally split with observation shuffling:

````@julia
y, X = unpack(data, ==(:target); rng=123);
schema(X)
````

````
┌──────────────┬───────────────────┬──────────────────────────────────┐
│ names        │ scitypes          │ types                            │
├──────────────┼───────────────────┼──────────────────────────────────┤
│ OverallQual  │ OrderedFactor{10} │ CategoricalValue{Int64, UInt32}  │
│ GrLivArea    │ Continuous        │ Float64                          │
│ Neighborhood │ Multiclass{25}    │ CategoricalValue{String, UInt32} │
│ x1stFlrSF    │ Continuous        │ Float64                          │
│ TotalBsmtSF  │ Continuous        │ Float64                          │
│ BsmtFinSF1   │ Continuous        │ Float64                          │
│ LotArea      │ Continuous        │ Float64                          │
│ GarageCars   │ Count             │ Int64                            │
│ MSSubClass   │ Multiclass{15}    │ CategoricalValue{String, UInt32} │
│ GarageArea   │ Continuous        │ Float64                          │
│ YearRemodAdd │ Count             │ Int64                            │
│ YearBuilt    │ Count             │ Int64                            │
└──────────────┴───────────────────┴──────────────────────────────────┘

````

````@julia
first(y, 4)
````

````
4-element Vector{Float64}:
 145000.0
 239799.0
 268000.0
 226000.0
````

Define a pipeline model:

````@julia
pipe = ContinuousEncoder() |> Standardizer() |> RidgeRegressor()
````

````
DeterministicPipeline(
  continuous_encoder = ContinuousEncoder(
        drop_last = false, 
        one_hot_ordered_factors = false), 
  standardizer = Standardizer(
        features = Symbol[], 
        ignore = false, 
        ordered_factor = false, 
        count = false), 
  ridge_regressor = RidgeRegressor(
        lambda = 1.0, 
        fit_intercept = true, 
        penalize_intercept = false, 
        scale_penalty_with_samples = true, 
        solver = nothing), 
  cache = true)
````

Access a nested hyperparameter:

````@julia
pipe.ridge_regressor.fit_intercept
````

````
true
````

Change it's value:

````@julia
pipe.ridge_regressor.fit_intercept = false;
````

Evaluate the pipeline:

````@julia
e1 = evaluate(pipe, X, y; resampling=CV(nfolds=4, rng=123), repeats=2, measure=mav)
````

````
PerformanceEvaluation object with these fields:
  model, tag, measure, operation,
  measurement (per-fold aggregate), uncertainty_radius_95 (1.96*SE),
  per_fold, per_observation,
  fitted_params_per_fold, report_per_fold,
  train_test_rows, resampling, repeats
Tag: DeterministicPipeline-639
Extract:
┌──────────┬───────────┬─────────────┬─────────┐
│ measure  │ operation │ measurement │ 1.96*SE │
├──────────┼───────────┼─────────────┼─────────┤
│ LPLoss(  │ predict   │ 179980.0    │ 630.0   │
│   p = 1) │           │             │         │
└──────────┴───────────┴─────────────┴─────────┘
┌───────────────────────────────────────────────────────────────────────────────
│ per_fold                                                                     ⋯
├───────────────────────────────────────────────────────────────────────────────
│ [180000.0, 180000.0, 181000.0, 179000.0, 179000.0, 180000.0, 180000.0, 18100 ⋯
└───────────────────────────────────────────────────────────────────────────────
                                                                1 column omitted
Apply `describe` to this result for a named tuple summary.
````

Notice the target very large on the current scale:

````@julia
@show mean(y) std(y);
````

````
mean(y) = 180151.2335164835
std(y) = 76696.59253004662

````

So we wrap the pipeline in target normalization:

````@julia
norm_pipe = TransformedTargetModel(pipe, transformer=Standardizer())
````

````
TransformedTargetModelDeterministic(
  model = DeterministicPipeline(
        continuous_encoder = ContinuousEncoder(drop_last = false, …), 
        standardizer = Standardizer(features = Symbol[], …), 
        ridge_regressor = RidgeRegressor(lambda = 1.0, …), 
        cache = true), 
  transformer = Standardizer(
        features = Symbol[], 
        ignore = false, 
        ordered_factor = false, 
        count = false), 
  inverse = nothing, 
  cache = true)
````

Note that target predictions will remain on the original scale. However, as
internally we are using a normalized target, we get different performance:

````@julia
e2 = evaluate(norm_pipe, X, y; resampling=CV(nfolds=4, rng=123), repeats=2, measure=mav)
````

````
PerformanceEvaluation object with these fields:
  model, tag, measure, operation,
  measurement (per-fold aggregate), uncertainty_radius_95 (1.96*SE),
  per_fold, per_observation,
  fitted_params_per_fold, report_per_fold,
  train_test_rows, resampling, repeats
Tag: TransformedTargetModelDeterministic-399
Extract:
┌──────────┬───────────┬─────────────┬─────────┐
│ measure  │ operation │ measurement │ 1.96*SE │
├──────────┼───────────┼─────────────┼─────────┤
│ LPLoss(  │ predict   │ 19780.0     │ 800.0   │
│   p = 1) │           │             │         │
└──────────┴───────────┴─────────────┴─────────┘
┌──────────────────────────────────────────────────────────────────────────┐
│ per_fold                                                                 │
├──────────────────────────────────────────────────────────────────────────┤
│ [18100.0, 19800.0, 21400.0, 19600.0, 20300.0, 19500.0, 18700.0, 20900.0] │
└──────────────────────────────────────────────────────────────────────────┘
Apply `describe` to this result for a named tuple summary.
````

Changing the regularization parameter `lambda` of ridge regressor, we can arrange that
the target transformation gives better performance:

````@julia
pipe_original = deepcopy(pipe)
pipe.ridge_regressor.lambda = 0.45

evaluations = evaluate(
    [
        "default lambda" => pipe_original,
        "new lambda" => pipe,
        "new lambda & normalized target" => norm_pipe],
    X,
    y;
    resampling=CV(nfolds=4, rng=123),
    repeats=2,
    measure=mav,
)
````

````
3-element Vector{PerformanceEvaluation{M, Vector{StatisticalMeasuresBase.RobustMeasure{StatisticalMeasuresBase.FussyMeasure{StatisticalMeasuresBase.RobustMeasure{StatisticalMeasuresBase.Multimeasure{StatisticalMeasuresBase.SupportsMissingsMeasure{StatisticalMeasures.LPLossOnScalars{Int64}}, Nothing, StatisticalMeasuresBase.Mean, typeof(identity)}}, Nothing}}}, Vector{Float64}, Vector{Float64}, Vector{typeof(predict)}, Vector{Vector{Float64}}, Vector{Vector{Vector{Float64}}}, FittedParamsPerFold, ReportPerFold, CV} where {M, FittedParamsPerFold, ReportPerFold}}:
 PerformanceEvaluation("default lambda", 179980.0 ± 630.0)
 PerformanceEvaluation("new lambda", 180100.0 ± 1500.0)
 PerformanceEvaluation("new lambda & normalized target", 18460.0 ± 440.0)
````

Here's a pretty view of these results:

````@julia
describe.(evaluations) |> pretty
````

````
┌────────────────────────────────┬──────────────────────┐
│ tag                            │ LPLoss               │
│ String                         │ Measurement{Float64} │
│ Textual                        │ Continuous           │
├────────────────────────────────┼──────────────────────┤
│ default lambda                 │ 179980.0±630.0       │
│ new lambda                     │ 180100.0±1500.0      │
│ new lambda & normalized target │ 18460.0±440.0        │
└────────────────────────────────┴──────────────────────┘

````

Finding optimal hyper-parameter values is the subject of the next lesson.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

