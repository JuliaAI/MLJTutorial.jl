```@meta
EditURL = "notebook.jl"
```

# Tutorial 3. Transformers and Pipelines

> **Goals:** Learn how to:
> 1. Apply common data preprocessing tasks using an MLJ **transformer**.
> 2. Combine a sequence of transformers with a supervised model in a single standalone model called a **pipelines**
> 3. Wrap a supervised model in transformations/inverse transformations of the target variable

To run the code in this tutorial in a live Julia session, first follow the instructions
given [here](@ref instructions).

### Transformers

Unsupervised models, which receive no target `y` during training, always have a
`transform` operation. They sometimes also support an `inverse_transform` operation,
with obvious meaning, and sometimes support a `predict` operation (see the clustering
example discussed
[here](https://juliaai.github.io/MLJ.jl/dev/transformers/#Transformers-that-also-predict-1)).
Otherwise, they are handled much like supervised models.

Here's a simple standardization example:

````@julia
using MLJ

x = rand(100);
@show mean(x) std(x);
````

````
mean(x) = 0.47696749924138343
std(x) = 0.29728842148044776

````

````@julia
model = Standardizer() # a built-in model
mach = machine(model, x)
fit!(mach)
xhat = transform(mach, x);
@show mean(xhat) std(xhat);
````

````
[ Info: Training machine(Standardizer(features = Symbol[], …), …).
mean(xhat) = 2.042810365310288e-16
std(xhat) = 1.0

````

This particular model has an `inverse_transform`:

````@julia
inverse_transform(mach, xhat) ≈ x
````

````
true
````

### Re-encoding the King County House data as continuous

For further illustrations of transformers, let's re-encode *all* of the King County
House input features (see Exercise 3) into a set of `Continuous` features. We do this
with the `ContinuousEncoder` model, which, by default, will:

- one-hot encode all `Multiclass` features
- coerce all `OrderedFactor` features to `Continuous` ones
- coerce all `Count` features to `Continuous` ones (there aren't any)
- drop any remaining non-Continuous features (none of these either)

First, we reload the data and fix the scitypes (Exercise 3):

````@julia
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
````

````
┌───────────────┬───────────────────┬───────────────────────────────────┐
│ names         │ scitypes          │ types                             │
├───────────────┼───────────────────┼───────────────────────────────────┤
│ price         │ Continuous        │ Float64                           │
│ bedrooms      │ OrderedFactor{13} │ CategoricalValue{Int64, UInt32}   │
│ bathrooms     │ OrderedFactor{30} │ CategoricalValue{Float64, UInt32} │
│ sqft_living   │ Continuous        │ Float64                           │
│ sqft_lot      │ Continuous        │ Float64                           │
│ floors        │ OrderedFactor{6}  │ CategoricalValue{Float64, UInt32} │
│ waterfront    │ OrderedFactor{2}  │ CategoricalValue{Int64, UInt32}   │
│ view          │ OrderedFactor{5}  │ CategoricalValue{Int64, UInt32}   │
│ condition     │ OrderedFactor{5}  │ CategoricalValue{Int64, UInt32}   │
│ grade         │ OrderedFactor{12} │ CategoricalValue{Int64, UInt32}   │
│ sqft_above    │ Continuous        │ Float64                           │
│ sqft_basement │ Continuous        │ Float64                           │
│ yr_built      │ Continuous        │ Float64                           │
│ zipcode       │ Multiclass{70}    │ CategoricalValue{Int64, UInt32}   │
│ lat           │ Continuous        │ Float64                           │
│ long          │ Continuous        │ Float64                           │
│ sqft_living15 │ Continuous        │ Float64                           │
│ sqft_lot15    │ Continuous        │ Float64                           │
│ is_renovated  │ OrderedFactor{2}  │ CategoricalValue{Bool, UInt32}    │
└───────────────┴───────────────────┴───────────────────────────────────┘

````

````@julia
y, X = unpack(house, ==(:price), rng=123);
````

Instantiate the unsupervised model (transformer):

````@julia
encoder = ContinuousEncoder() # a built-in model; no need to @load it
````

````
ContinuousEncoder(
  drop_last = false, 
  one_hot_ordered_factors = false)
````

Bind the model to the data and fit!

````@julia
mach = machine(encoder, X) |> fit!;
````

````
[ Info: Training machine(ContinuousEncoder(drop_last = false, …), …).

````

Transform and inspect the result:

````@julia
Xcont = transform(mach, X);
schema(Xcont)
````

````
┌────────────────┬────────────┬─────────┐
│ names          │ scitypes   │ types   │
├────────────────┼────────────┼─────────┤
│ bedrooms       │ Continuous │ Float64 │
│ bathrooms      │ Continuous │ Float64 │
│ sqft_living    │ Continuous │ Float64 │
│ sqft_lot       │ Continuous │ Float64 │
│ floors         │ Continuous │ Float64 │
│ waterfront     │ Continuous │ Float64 │
│ view           │ Continuous │ Float64 │
│ condition      │ Continuous │ Float64 │
│ grade          │ Continuous │ Float64 │
│ sqft_above     │ Continuous │ Float64 │
│ sqft_basement  │ Continuous │ Float64 │
│ yr_built       │ Continuous │ Float64 │
│ zipcode__98001 │ Continuous │ Float64 │
│ zipcode__98002 │ Continuous │ Float64 │
│ zipcode__98003 │ Continuous │ Float64 │
│ zipcode__98004 │ Continuous │ Float64 │
│ zipcode__98005 │ Continuous │ Float64 │
│ zipcode__98006 │ Continuous │ Float64 │
│ zipcode__98007 │ Continuous │ Float64 │
│ zipcode__98008 │ Continuous │ Float64 │
│ zipcode__98010 │ Continuous │ Float64 │
│ zipcode__98011 │ Continuous │ Float64 │
│ zipcode__98014 │ Continuous │ Float64 │
│ zipcode__98019 │ Continuous │ Float64 │
│ zipcode__98022 │ Continuous │ Float64 │
│ zipcode__98023 │ Continuous │ Float64 │
│ zipcode__98024 │ Continuous │ Float64 │
│ zipcode__98027 │ Continuous │ Float64 │
│ zipcode__98028 │ Continuous │ Float64 │
│ zipcode__98029 │ Continuous │ Float64 │
│ zipcode__98030 │ Continuous │ Float64 │
│ zipcode__98031 │ Continuous │ Float64 │
│ zipcode__98032 │ Continuous │ Float64 │
│ zipcode__98033 │ Continuous │ Float64 │
│ zipcode__98034 │ Continuous │ Float64 │
│ zipcode__98038 │ Continuous │ Float64 │
│ zipcode__98039 │ Continuous │ Float64 │
│ zipcode__98040 │ Continuous │ Float64 │
│ zipcode__98042 │ Continuous │ Float64 │
│ zipcode__98045 │ Continuous │ Float64 │
│ zipcode__98052 │ Continuous │ Float64 │
│ zipcode__98053 │ Continuous │ Float64 │
│ zipcode__98055 │ Continuous │ Float64 │
│ zipcode__98056 │ Continuous │ Float64 │
│ zipcode__98058 │ Continuous │ Float64 │
│ zipcode__98059 │ Continuous │ Float64 │
│ zipcode__98065 │ Continuous │ Float64 │
│ zipcode__98070 │ Continuous │ Float64 │
│ zipcode__98072 │ Continuous │ Float64 │
│ zipcode__98074 │ Continuous │ Float64 │
│ zipcode__98075 │ Continuous │ Float64 │
│ zipcode__98077 │ Continuous │ Float64 │
│ zipcode__98092 │ Continuous │ Float64 │
│ zipcode__98102 │ Continuous │ Float64 │
│ zipcode__98103 │ Continuous │ Float64 │
│ zipcode__98105 │ Continuous │ Float64 │
│ zipcode__98106 │ Continuous │ Float64 │
│ zipcode__98107 │ Continuous │ Float64 │
│ zipcode__98108 │ Continuous │ Float64 │
│ zipcode__98109 │ Continuous │ Float64 │
│ zipcode__98112 │ Continuous │ Float64 │
│ zipcode__98115 │ Continuous │ Float64 │
│ zipcode__98116 │ Continuous │ Float64 │
│ zipcode__98117 │ Continuous │ Float64 │
│ zipcode__98118 │ Continuous │ Float64 │
│ zipcode__98119 │ Continuous │ Float64 │
│ zipcode__98122 │ Continuous │ Float64 │
│ zipcode__98125 │ Continuous │ Float64 │
│ zipcode__98126 │ Continuous │ Float64 │
│ zipcode__98133 │ Continuous │ Float64 │
│ zipcode__98136 │ Continuous │ Float64 │
│ zipcode__98144 │ Continuous │ Float64 │
│ zipcode__98146 │ Continuous │ Float64 │
│ zipcode__98148 │ Continuous │ Float64 │
│ zipcode__98155 │ Continuous │ Float64 │
│ zipcode__98166 │ Continuous │ Float64 │
│ zipcode__98168 │ Continuous │ Float64 │
│ zipcode__98177 │ Continuous │ Float64 │
│ zipcode__98178 │ Continuous │ Float64 │
│ zipcode__98188 │ Continuous │ Float64 │
│ zipcode__98198 │ Continuous │ Float64 │
│ zipcode__98199 │ Continuous │ Float64 │
│ lat            │ Continuous │ Float64 │
│ long           │ Continuous │ Float64 │
│ sqft_living15  │ Continuous │ Float64 │
│ sqft_lot15     │ Continuous │ Float64 │
│ is_renovated   │ Continuous │ Float64 │
└────────────────┴────────────┴─────────┘

````

### More transformers

Here's how to list all of MLJ's unsupervised models:

````@julia
models(m->!m.is_supervised)
````

````
92-element Vector{NamedTuple{(:name, :package_name, :is_supervised, :abstract_type, :constructor, :deep_properties, :docstring, :fit_data_scitype, :human_name, :hyperparameter_ranges, :hyperparameter_types, :hyperparameters, :implemented_methods, :inverse_transform_scitype, :is_pure_julia, :is_wrapper, :iteration_parameter, :load_path, :package_license, :package_url, :package_uuid, :predict_scitype, :prediction_type, :reporting_operations, :reports_feature_importances, :supports_class_weights, :supports_online, :supports_training_losses, :supports_weights, :tags, :target_in_fit, :transform_scitype, :input_scitype, :target_scitype, :output_scitype)}}:
 (name = ABODDetector, package_name = OutlierDetectionNeighbors, ... )
 (name = ABODDetector, package_name = OutlierDetectionPython, ... )
 (name = AffinityPropagation, package_name = Clustering, ... )
 (name = AffinityPropagation, package_name = MLJScikitLearnInterface, ... )
 (name = AgglomerativeClustering, package_name = MLJScikitLearnInterface, ... )
 (name = AutoEncoder, package_name = BetaML, ... )
 (name = BM25Transformer, package_name = MLJText, ... )
 (name = Birch, package_name = MLJScikitLearnInterface, ... )
 (name = BisectingKMeans, package_name = MLJScikitLearnInterface, ... )
 (name = BorderlineSMOTE1, package_name = Imbalance, ... )
 (name = CBLOFDetector, package_name = OutlierDetectionPython, ... )
 (name = CDDetector, package_name = OutlierDetectionPython, ... )
 (name = COFDetector, package_name = OutlierDetectionNeighbors, ... )
 (name = COFDetector, package_name = OutlierDetectionPython, ... )
 (name = COPODDetector, package_name = OutlierDetectionPython, ... )
 (name = CardinalityReducer, package_name = MLJTransforms, ... )
 (name = ClusterUndersampler, package_name = Imbalance, ... )
 (name = ContinuousEncoder, package_name = MLJTransforms, ... )
 (name = ContrastEncoder, package_name = MLJTransforms, ... )
 (name = CountTransformer, package_name = MLJText, ... )
 (name = DBSCAN, package_name = Clustering, ... )
 (name = DBSCAN, package_name = MLJScikitLearnInterface, ... )
 (name = DNNDetector, package_name = OutlierDetectionNeighbors, ... )
 (name = ECODDetector, package_name = OutlierDetectionPython, ... )
 (name = ENNUndersampler, package_name = Imbalance, ... )
 (name = FactorAnalysis, package_name = MultivariateStats, ... )
 (name = FeatureAgglomeration, package_name = MLJScikitLearnInterface, ... )
 (name = FeatureSelector, package_name = FeatureSelection, ... )
 (name = FillImputer, package_name = MLJTransforms, ... )
 (name = FrequencyEncoder, package_name = MLJTransforms, ... )
 (name = GMMDetector, package_name = OutlierDetectionPython, ... )
 (name = GaussianMixtureClusterer, package_name = BetaML, ... )
 (name = GaussianMixtureImputer, package_name = BetaML, ... )
 (name = GeneralImputer, package_name = BetaML, ... )
 (name = HBOSDetector, package_name = OutlierDetectionPython, ... )
 (name = HDBSCAN, package_name = MLJScikitLearnInterface, ... )
 (name = HierarchicalClustering, package_name = Clustering, ... )
 (name = ICA, package_name = MultivariateStats, ... )
 (name = IForestDetector, package_name = OutlierDetectionPython, ... )
 (name = INNEDetector, package_name = OutlierDetectionPython, ... )
 (name = InteractionTransformer, package_name = MLJTransforms, ... )
 (name = KDEDetector, package_name = OutlierDetectionPython, ... )
 (name = KMeans, package_name = Clustering, ... )
 (name = KMeans, package_name = MLJScikitLearnInterface, ... )
 (name = KMeans, package_name = ParallelKMeans, ... )
 (name = KMeansClusterer, package_name = BetaML, ... )
 (name = KMedoids, package_name = Clustering, ... )
 (name = KMedoidsClusterer, package_name = BetaML, ... )
 (name = KNNDetector, package_name = OutlierDetectionNeighbors, ... )
 (name = KNNDetector, package_name = OutlierDetectionPython, ... )
 (name = KernelPCA, package_name = MultivariateStats, ... )
 (name = LMDDDetector, package_name = OutlierDetectionPython, ... )
 (name = LOCIDetector, package_name = OutlierDetectionPython, ... )
 (name = LODADetector, package_name = OutlierDetectionPython, ... )
 (name = LOFDetector, package_name = OutlierDetectionNeighbors, ... )
 (name = LOFDetector, package_name = OutlierDetectionPython, ... )
 (name = MCDDetector, package_name = OutlierDetectionPython, ... )
 (name = MeanShift, package_name = MLJScikitLearnInterface, ... )
 (name = MiniBatchKMeans, package_name = MLJScikitLearnInterface, ... )
 (name = MissingnessEncoder, package_name = MLJTransforms, ... )
 (name = OCSVMDetector, package_name = OutlierDetectionPython, ... )
 (name = OPTICS, package_name = MLJScikitLearnInterface, ... )
 (name = OneClassSVM, package_name = LIBSVM, ... )
 (name = OneHotEncoder, package_name = MLJTransforms, ... )
 (name = OrdinalEncoder, package_name = MLJTransforms, ... )
 (name = PCA, package_name = MultivariateStats, ... )
 (name = PCADetector, package_name = OutlierDetectionPython, ... )
 (name = PPCA, package_name = MultivariateStats, ... )
 (name = RODDetector, package_name = OutlierDetectionPython, ... )
 (name = ROSE, package_name = Imbalance, ... )
 (name = RandomForestImputer, package_name = BetaML, ... )
 (name = RandomOversampler, package_name = Imbalance, ... )
 (name = RandomUndersampler, package_name = Imbalance, ... )
 (name = RandomWalkOversampler, package_name = Imbalance, ... )
 (name = SMOTE, package_name = Imbalance, ... )
 (name = SMOTEN, package_name = Imbalance, ... )
 (name = SMOTENC, package_name = Imbalance, ... )
 (name = SODDetector, package_name = OutlierDetectionPython, ... )
 (name = SOSDetector, package_name = OutlierDetectionPython, ... )
 (name = SelfOrganizingMap, package_name = SelfOrganizingMaps, ... )
 (name = SimpleImputer, package_name = BetaML, ... )
 (name = SpectralClustering, package_name = MLJScikitLearnInterface, ... )
 (name = Standardizer, package_name = MLJTransforms, ... )
 (name = TSVDTransformer, package_name = TSVD, ... )
 (name = TargetEncoder, package_name = MLJTransforms, ... )
 (name = TfidfTransformer, package_name = MLJText, ... )
 (name = TomekUndersampler, package_name = Imbalance, ... )
 (name = UnivariateBoxCoxTransformer, package_name = MLJTransforms, ... )
 (name = UnivariateDiscretizer, package_name = MLJTransforms, ... )
 (name = UnivariateFillImputer, package_name = MLJTransforms, ... )
 (name = UnivariateStandardizer, package_name = MLJTransforms, ... )
 (name = UnivariateTimeTypeToContinuous, package_name = MLJTransforms, ... )
````

Some commonly used ones are built-in (do not require `@load`ing):

model type                  | does what?
----------------------------|----------------------------------------------
ContinuousEncoder | transform input table to a table of `Continuous` features (see above)
FeatureSelector | retain or dump selected features
FillImputer | impute missing values
OneHotEncoder | one-hot encoder `Multiclass` (and optionally `OrderedFactor`) features
Standardizer | standardize (whiten) a vector or all `Continuous` features of a table
UnivariateBoxCoxTransformer | apply a learned Box-Cox transformation to a vector
UnivariateDiscretizer | discretize a `Continuous` vector, and hence render its elscitypw `OrderedFactor`

In addition to "dynamic" transformers (ones that learn something
from the data and must be `fit!`) users can wrap ordinary functions
as transformers, and such *static* transformers can depend on
parameters, like the dynamic ones. See
[here](https://juliaai.github.io/MLJ.jl/dev/transformers/#Static-transformers-1)
for how to define your own static transformers.

### Pipelines

````@julia
length(schema(Xcont).names)
````

````
87
````

Let's suppose that additionally we'd like to reduce the dimension of
our data.  A model that will do this is `PCA` from
`MultivariateStats.jl`:

````@julia
PCA = @load PCA
reducer = PCA()
````

````
PCA(
  maxoutdim = 0, 
  method = :auto, 
  variance_ratio = 0.99, 
  mean = nothing)
````

Now, rather simply repeating the work-flow above, applying the new
transformation to `Xcont`, we can combine both the encoding and the
dimension-reducing models into a single model, known as a
*pipeline*. While MLJ offers a powerful interface for composing
models in a variety of ways, we'll stick to these simplest class of
composite models for now. The simplest way to construct a pipeline
is using the Julia's `|>` syntax:

````@julia
pipe = encoder |> reducer
````

````
UnsupervisedPipeline(
  continuous_encoder = ContinuousEncoder(
        drop_last = false, 
        one_hot_ordered_factors = false), 
  pca = PCA(
        maxoutdim = 0, 
        method = :auto, 
        variance_ratio = 0.99, 
        mean = nothing), 
  cache = true)
````

Notice that the model `pipe` has other models as hyperparameters
(with names automatically generated based on the mode type
name). The hyperparameters of the component models are are now
*nested*, but we can still access them:

````@julia
@show pipe.pca.variance_ratio
````

````
0.99
````

````@julia
pipe.pca.variance_ratio = 0.9995
````

````
0.9995
````

The pipeline model behaves like any other transformer:

````@julia
mach = machine(pipe, X)
fit!(mach)
Xsmall = transform(mach, X)
schema(Xsmall)
````

````
┌───────┬────────────┬─────────┐
│ names │ scitypes   │ types   │
├───────┼────────────┼─────────┤
│ x1    │ Continuous │ Float64 │
│ x2    │ Continuous │ Float64 │
│ x3    │ Continuous │ Float64 │
└───────┴────────────┴─────────┘

````

Want to combine this pre-processing with ridge regression?

````@julia
RidgeRegressor = @load RidgeRegressor pkg=MLJLinearModels
rgs = RidgeRegressor()
pipe2 = pipe |> rgs
````

````
DeterministicPipeline(
  continuous_encoder = ContinuousEncoder(
        drop_last = false, 
        one_hot_ordered_factors = false), 
  pca = PCA(
        maxoutdim = 0, 
        method = :auto, 
        variance_ratio = 0.9995, 
        mean = nothing), 
  ridge_regressor = RidgeRegressor(
        lambda = 1.0, 
        fit_intercept = true, 
        penalize_intercept = false, 
        scale_penalty_with_samples = true, 
        solver = nothing), 
  cache = true)
````

Now our pipeline is a supervised model, instead of a transformer,
whose performance we can evaluate:

````@julia
mach = machine(pipe2, X, y)
evaluate!(mach, measure=mae, resampling=Holdout()) # `CV(nfolds=6)` is `resampling` default
````

````
PerformanceEvaluation object with these fields:
  model, tag, measure, operation,
  measurement (per-fold aggregate), uncertainty_radius_95 (1.96*SE),
  per_fold, per_observation,
  fitted_params_per_fold, report_per_fold,
  train_test_rows, resampling, repeats
Tag: DeterministicPipeline-190
Extract:
┌──────────┬───────────┬─────────────┐
│ measure  │ operation │ measurement │
├──────────┼───────────┼─────────────┤
│ LPLoss(  │ predict   │ 176000.0    │
│   p = 1) │           │             │
└──────────┴───────────┴─────────────┘
Apply `describe` to this result for a named tuple summary.
````

### Training of composite models is "smart"

Now notice what happens if we train on all the data, then change a
regressor hyperparameter and retrain:

````@julia
fit!(mach);
````

````
[ Info: Training machine(DeterministicPipeline(continuous_encoder = ContinuousEncoder(drop_last = false, …), …), …).
[ Info: Training machine(:continuous_encoder, …).
[ Info: Training machine(:pca, …).
[ Info: Training machine(:ridge_regressor, …).
┌ Info: Solver: MLJLinearModels.Analytical
│   iterative: Bool false
└   max_inner: Int64 200

````

````@julia
pipe2.ridge_regressor.lambda = 0.1
fit!(mach);
````

````
[ Info: Updating machine(DeterministicPipeline(continuous_encoder = ContinuousEncoder(drop_last = false, …), …), …) (with warm restart if possible).
[ Info: Not retraining machine(:continuous_encoder, …). Use `force=true` to force.
[ Info: Not retraining machine(:pca, …). Use `force=true` to force.
[ Info: Updating machine(:ridge_regressor, …) (with warm restart if possible).
┌ Info: Solver: MLJLinearModels.Analytical
│   iterative: Bool false
└   max_inner: Int64 200

````

Second time only the ridge regressor is retrained!

Mutate a hyperparameter of the `PCA` model and every model except
the `ContinuousEncoder` (which comes before it will be retrained):

````@julia
pipe2.pca.variance_ratio = 0.9999
fit!(mach);
````

````
[ Info: Updating machine(DeterministicPipeline(continuous_encoder = ContinuousEncoder(drop_last = false, …), …), …) (with warm restart if possible).
[ Info: Not retraining machine(:continuous_encoder, …). Use `force=true` to force.
[ Info: Updating machine(:pca, …) (with warm restart if possible).
[ Info: Training machine(:ridge_regressor, …).
┌ Info: Solver: MLJLinearModels.Analytical
│   iterative: Bool false
└   max_inner: Int64 200

````

### Inspecting composite models

The dot syntax used above to change the values of *nested*
hyperparameters is also useful when inspecting the learned
parameters and report generated when training a composite model:

````@julia
fitted_params(mach).ridge_regressor
````

````
(coefs = [:x1 => -0.7328956348620362, :x2 => -0.16590563197071073, :x3 => 194.59514735965914, :x4 => 102.7129805185524], intercept = 540088.1417665293)
````

````@julia
report(mach).pca
````

````
(indim = 87, outdim = 4, tprincipalvar = 2.463215246230865e9, tresidualvar = 157533.26199674606, tvar = 2.4633727794928617e9, mean = [4.369869985656781, 8.45912182482765, 2079.8997362698374, 15106.967565816869, 1.988617961412113, 1.0075417572757137, 1.2343034284921113, 3.4094295100171195, 6.6569194466293435, 1788.3906907879516, 291.5090454818859, 1971.0051357978994, 0.01674917873502059, 0.009207421459306898, 0.012955165872391617, 0.01466709850552908, 0.007773099523434969, 0.023041687873039375, 0.006523851385740064, 0.013093971221024384, 0.004626844954425577, 0.009022347661129875, 0.005737287743487716, 0.008791005413408597, 0.010826817193355851, 0.02308795632258363, 0.0037477444130847174, 0.01906260121223338, 0.013093971221024384, 0.014852172303706102, 0.011844723083329478, 0.012677555175126082, 0.005783556193031971, 0.019987970203118495, 0.025216305001619397, 0.027298385231110906, 0.0023134224772127887, 0.013047702771480128, 0.025355110350252164, 0.010225327349280526, 0.02655809003840281, 0.018738722065423586, 0.012399944477860548, 0.018784990514967844, 0.021052144542636375, 0.021653634386711702, 0.01434321935871929, 0.005459677046222181, 0.012631286725581826, 0.020404386249016797, 0.016610373386387822, 0.009161153009762642, 0.016240225790033775, 0.0048581872021468565, 0.027853606625641975, 0.010595474945634571, 0.015499930597325684, 0.012307407578772035, 0.008605931615231573, 0.005043261000323879, 0.012446212927404802, 0.026974506084301113, 0.015268588349604404, 0.02558645259797344, 0.02350437236848193, 0.008513394716143062, 0.013417850367834173, 0.018970064313144866, 0.016379031138666542, 0.02285661407486235, 0.012168602230139268, 0.01587007819367973, 0.013325313468745662, 0.002637301624022579, 0.020635728496738073, 0.011752186184240966, 0.012446212927404802, 0.011798454633785222, 0.012122333780595013, 0.006292509138018785, 0.012955165872391617, 0.01466709850552908, 47.560052519317075, -122.21389640494147, 1986.552491556008, 12768.455651691113, 1.9577106371165502], principalvars = [2.1770715510450835e9, 2.8418139726430434e8, 1.6850160830642523e6, 277281.83841332485], loadings = [-0.03094830022236769 -0.0028697042773965405 0.5095749351696565 0.14813207201762205; -0.2858879205025633 -0.05108905045966113 2.268889769352169 0.26924666347207565; -171.31152075330942 -44.87769073823875 875.5659689915609 168.66602408773912; -40575.715872989625 8322.862769828478 -3.2832135973351932 0.12584457836092736; 0.007778995904394227 0.010162646821257629 0.45831749135900773 -0.40525347805466555; -0.002211643733571352 -0.0014772314887836425 0.008030553125354504 0.005029961404333902; -0.06029631790584413 -0.008960249584414274 0.1958156352012937 0.16895626991097693; 0.005151697385623508 -0.003924246356610304 -0.0665597154098043 0.1354264847325144; -0.1437593832705363 -0.035785564907580926 0.9296582832070637 -0.04708045076934479; -163.86359230271847 -42.2761711031475 759.8406475174772 -258.1268732347522; -7.4479284505907195 -2.6015196350913126 115.72532147407891 426.79289732249435; -1.807573225134045 -1.0484988749143294 10.79613973242878 -7.023826170513946; 0.00032880415830771555 0.0012608115742089364 -0.002945476396696076 -0.0036782279694778045; 0.0018081489212890985 0.00041720803785755316 -0.004581868056013803 -0.0029394721466696524; 0.0015021831533924377 0.00031177511754644895 -0.0019320797726573077 -0.0002596561899149827; 0.0005396372791692474 -0.000893346454772592 0.01417719412810004 0.0031706817296192446; -0.0011599327900621245 -0.0011496200901734738 0.004972673322122764 0.002520313937122101; 0.0008852219451363187 -0.0004482791535716682 0.021874476941801627 0.010727396671281166; 0.0007603631489364767 0.00022354851848726864 0.0011083720581940193 -0.00017177945017749513; 0.0016539107318789892 0.0003248505808109719 0.0005427567135431955 0.0038757212549927747; -0.005197066654766852 -0.0028999696129264957 -0.0006610034877139049 -0.0016313631088370703; 0.0009483368558858148 0.0005124522329180149 0.002517643264875871 -0.0004027030686682856; -0.012705747793516893 -0.005556238235984659 -0.0024627691630764673 -0.0017728103263237398; -0.006308120223093767 -0.004629616612497465 9.875656548477721e-5 -0.0033864500235734796; -0.016336236551631574 -0.0028974171250481205 -0.006012357830940101 -0.0033676160018411743; 0.0027366321306056475 0.0004778242487191469 -0.0015501314219459148 0.0001325216045831566; -0.008668652483283187 -0.003256324995816614 -0.0006435711877160101 -0.0011877774016416137; -0.009468251570438523 -0.005500626005433714 0.0075196267669614385 0.0028126131539448814; 0.001140913317722173 0.000572531307418339 0.0010254522492583957 0.001382814504817588; 0.0026777974153313257 0.0015871226964074357 0.005756335524299741 -0.007719851104656195; 0.0012602481447323685 0.0014069712977438811 -0.0006748299461092006 -0.0035038683290240713; 0.0012878877889526351 0.0016854326076529065 -0.0014589422671149055 -0.0015576793048326606; 0.0007239635803662536 0.00024346172904716153 -0.0024096750648439557 0.0007940243060852357; 0.002320308225076724 0.00023562759375918166 0.007624840796732388 -0.0006306379558992351; 0.0036470315314979505 0.0016212181889954844 -0.00251393157775493 0.002289684485585593; -0.0073353565650927785 -0.0020449897917742764 0.003491080287892121 -0.013741345923645823; -0.0002104501319689401 -0.0003858234183143582 0.0045051712747222195 -3.92719223312614e-5; 0.00033418349045190743 -0.0005639291977603692 0.015379903941725859 0.006595063200323842; -0.0023370798111320176 0.0003284085425596085 -0.0015943786174122238 -0.008914392100202995; -0.008765678603957602 0.00023187662391027837 -0.001519299580633708 -0.003796430858632887; 0.0027305490507038913 0.000599735899014808 0.011343201263716001 -0.004577608138018805; -0.010376916001404328 -0.00456162359333042 0.012105411259293885 -0.01278846025734618; 0.0018397756388539932 0.0014729352637121139 -0.0035169570693632315 -0.0015998490258304814; 0.0029219175203259503 0.001739257034488365 -0.0005149817059623669 -0.0025909215858809995; -0.001965284760130314 0.0015785374825356144 -0.0006069857511606733 -0.00219821881275978; 0.00032033258247913735 -0.00025798020795042684 0.01082421581718693 -0.01075391091924657; -0.000293460268859073 0.0019482533579054907 0.010459732530058656 -0.008559046853872078; -0.012318617611505073 -0.006161809467829765 -0.004132795384379822 -0.00029595675265855993; -0.0051856378680646305 -0.0015107896501709397 0.0040265456441313675 -0.0010191161617766563; 0.00013118014979864297 -0.000787928971632483 0.01597756701191326 -0.00793136267530889; -0.0016113765212340324 -7.33427424547173e-5 0.0210518547697063 -0.009419400935426998; -0.009956208196007156 -0.00481086366221152 0.0074555113774001786 -0.0043428620714396765; -0.008261126915915021 -0.006344276309120161 0.0015564118816391935 -0.006710779485139364; 0.0015266831650697353 0.0007354422376225503 0.0004910771824210794 0.001054084874306195; 0.008777127223503491 0.0038805618162346964 -0.013479885793856396 0.00043029522712262434; 0.0028498175899847743 0.0011806267800324505 0.000853458794615925 0.003981869684338035; 0.004016070520843419 0.0017837724256102544 -0.011326651734512382 0.0031597957001893405; 0.003948858799778384 0.0016323806118867471 -0.006892677209652593 0.0009724769062962991; 0.002292298584120738 0.0009344403206429057 -0.004200692032859572 0.003369296656326426; 0.0015687063137883983 0.0006874861400338182 -0.00010724691357676522 0.0015500117520519344; 0.003382513606730918 0.0013635336536355926 0.006106632924822598 0.004364598499839203; 0.0069953817842506785 0.0027816782783378195 -0.008938140708670947 0.011020861304471573; 0.004150298652275991 0.0016515208124795427 -0.004769302492821057 0.006341303049228332; 0.007196359156943736 0.0028885154281337107 -0.012543285317966009 0.0072296394391801475; 0.00588328598079174 0.002337829809000462 -0.011212640648660066 0.006637182966451212; 0.00263814386620444 0.0011058181556187474 -0.0007686370270878558 0.0027171243111618316; 0.004208545377777481 0.0018166341117660785 -0.0041414038861467784 0.001159338319292009; 0.003726764304357646 0.001220952516575207 -0.007686277512764395 0.0034348177918453035; 0.004393939480701817 0.0017519372185008568 -0.010742912491051243 0.0028370467607631553; 0.0048836455528599235 0.0017652635795699628 -0.012242459606510638 0.0010894477109835194; 0.00297371007280959 0.0011539226050821653 -0.005186313967598504 0.0037906859730574044; 0.004743676806900177 0.002123728476316075 -0.0036046729924437836 0.006333692127823337; 0.0021481035188033247 0.0007241479186690785 -0.007265886268770565 0.0004436433404978908; 0.00041273078328325183 6.39973020863696e-5 -0.0012817768113536098 -0.0007018097016022647; 0.002501357265610357 0.00023483865414235618 -0.007382224243867089 0.0027604332546305745; 0.00033739547227819603 -0.0004701304360301899 -0.0008154621561255123 0.0021060499992519794; 0.0012898963743125335 0.0005466730464324534 -0.008997546594791591 0.0006904779808614219; 0.0008384737928569485 -0.0004514419383083872 0.0028482748345335786 0.004722431074377536; 0.0021293453661839793 0.0004772139524893017 -0.005575691447448697 0.004663496790067156; 0.0007853579603687956 6.195606481721725e-5 -0.0023251345925029234 0.0008424639677204089; 0.0015579654177854262 0.000462509645666839 -0.004963183688635742 0.000639768638271226; 0.00380542176999362 0.0015091461760304568 0.00048796129399739536 0.009547467619654531; 0.012624052290056621 0.002461611691125208 0.008168721684362822 0.01408320838621929; -0.035312401805926756 -0.011278569363365095 0.03676354955402072 -0.031109555124367007; -112.88077365642482 -56.84662012968592 572.2199960847366 -1.495676954161905; -23035.0566009115 -14659.614997914887 -8.975922676497527 0.23006892666563608; 0.0016649390524045601 0.00035640148161712997 -0.006972328544533634 -0.012428541266700524])
````

### Incorporating target transformations

Next, suppose that instead of using the raw `:price` as the training target, we want to
use the log-price (a common practice in dealing with house price data). However, suppose
that we still want to report final *predictions* on the original linear scale (and use
these for evaluation purposes). Then we wrap our supervised model using
`TransformedTargetModel`, which has two keyword arguments `transformer` and `inverse`.

````@julia
rgs_log = TransformedTargetModel(rgs, transformer=y->log.(y), inverse=z->exp.(z))
````

````
TransformedTargetModelDeterministic(
  model = RidgeRegressor(
        lambda = 0.1, 
        fit_intercept = true, 
        penalize_intercept = false, 
        scale_penalty_with_samples = true, 
        solver = nothing), 
  transformer = Main.var"##277".var"#6#7"(), 
  inverse = Main.var"##277".var"#8#9"(), 
  cache = true)
````

And here is a revised pipeline model:

````@julia
pipe3 = pipe |> rgs_log
mach = machine(pipe3, X, y)
evaluate!(mach, measure=mae)
````

````
PerformanceEvaluation object with these fields:
  model, tag, measure, operation,
  measurement (per-fold aggregate), uncertainty_radius_95 (1.96*SE),
  per_fold, per_observation,
  fitted_params_per_fold, report_per_fold,
  train_test_rows, resampling, repeats
Tag: DeterministicPipeline-398
Extract:
┌──────────┬───────────┬─────────────┬─────────┐
│ measure  │ operation │ measurement │ 1.96*SE │
├──────────┼───────────┼─────────────┼─────────┤
│ LPLoss(  │ predict   │ 162300.0    │ 4100.0  │
│   p = 1) │           │             │         │
└──────────┴───────────┴─────────────┴─────────┘
┌──────────────────────────────────────────────────────────────┐
│ per_fold                                                     │
├──────────────────────────────────────────────────────────────┤
│ [160000.0, 170000.0, 163000.0, 156000.0, 163000.0, 162000.0] │
└──────────────────────────────────────────────────────────────┘
Apply `describe` to this result for a named tuple summary.
````

MLJ will also allow you to insert *learned* target transformations. For example, we
might want to apply `Standardizer()` to the target, to standardize it, or
`UnivariateBoxCoxTransformer()` to make it look Gaussian. Then instead of specifying a
*function* for `target`, we specify a unsupervised *model* (or model type). One does not
specify `inverse` because only models implementing `inverse_transform` are allowed.

Let's see which of these two options results in a better outcome:

````@julia
box = UnivariateBoxCoxTransformer(n=20)
stand = Standardizer()

rgs_box = TransformedTargetModel(rgs, transformer=box)
pipe4 = pipe |> rgs_box
mach = machine(pipe4, X, y)
evaluate!(mach, measure=mae)
````

````
PerformanceEvaluation object with these fields:
  model, tag, measure, operation,
  measurement (per-fold aggregate), uncertainty_radius_95 (1.96*SE),
  per_fold, per_observation,
  fitted_params_per_fold, report_per_fold,
  train_test_rows, resampling, repeats
Tag: DeterministicPipeline-611
Extract:
┌──────────┬───────────┬─────────────┬──────────┐
│ measure  │ operation │ measurement │ 1.96*SE  │
├──────────┼───────────┼─────────────┼──────────┤
│ LPLoss(  │ predict   │ 510000.0    │ 730000.0 │
│   p = 1) │           │             │          │
└──────────┴───────────┴─────────────┴──────────┘
┌───────────────────────────────────────────────────────────┐
│ per_fold                                                  │
├───────────────────────────────────────────────────────────┤
│ [162000.0, 2.2e6, 181000.0, 161000.0, 176000.0, 172000.0] │
└───────────────────────────────────────────────────────────┘
Apply `describe` to this result for a named tuple summary.
````

````@julia
pipe4.transformed_target_model_deterministic.transformer = stand
evaluate!(mach, measure=mae)
````

````
PerformanceEvaluation object with these fields:
  model, tag, measure, operation,
  measurement (per-fold aggregate), uncertainty_radius_95 (1.96*SE),
  per_fold, per_observation,
  fitted_params_per_fold, report_per_fold,
  train_test_rows, resampling, repeats
Tag: DeterministicPipeline-592
Extract:
┌──────────┬───────────┬─────────────┬─────────┐
│ measure  │ operation │ measurement │ 1.96*SE │
├──────────┼───────────┼─────────────┼─────────┤
│ LPLoss(  │ predict   │ 171500.0    │ 1200.0  │
│   p = 1) │           │             │         │
└──────────┴───────────┴─────────────┴─────────┘
┌──────────────────────────────────────────────────────────────┐
│ per_fold                                                     │
├──────────────────────────────────────────────────────────────┤
│ [171000.0, 172000.0, 173000.0, 170000.0, 173000.0, 171000.0] │
└──────────────────────────────────────────────────────────────┘
Apply `describe` to this result for a named tuple summary.
````

### Tutorial 3 Resources

- From the MLJ manual:
    - [Transformers and other unsupervised models](https://juliaai.github.io/MLJ.jl/dev/transformers/)
    - [Linear pipelines](https://juliaai.github.io/MLJ.jl/dev/linear_pipelines/#Linear-Pipelines)
- From Data Science Tutorials:
    - [Composing models](https://juliaai.github.io/DataScienceTutorials.jl/getting-started/composing-models/)

### Tutorial 3 Exercises

#### Exercise 7

Consider again the Horse Colic classification problem considered in
Exercise 6, but with all features, `Finite` and `Infinite`:

````@julia
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
````

````
┌───────────────┬───────────────────┬───────────────────────────────────┐
│ names         │ scitypes          │ types                             │
├───────────────┼───────────────────┼───────────────────────────────────┤
│ bedrooms      │ OrderedFactor{13} │ CategoricalValue{Int64, UInt32}   │
│ bathrooms     │ OrderedFactor{30} │ CategoricalValue{Float64, UInt32} │
│ sqft_living   │ Continuous        │ Float64                           │
│ sqft_lot      │ Continuous        │ Float64                           │
│ floors        │ OrderedFactor{6}  │ CategoricalValue{Float64, UInt32} │
│ waterfront    │ OrderedFactor{2}  │ CategoricalValue{Int64, UInt32}   │
│ view          │ OrderedFactor{5}  │ CategoricalValue{Int64, UInt32}   │
│ condition     │ OrderedFactor{5}  │ CategoricalValue{Int64, UInt32}   │
│ grade         │ OrderedFactor{12} │ CategoricalValue{Int64, UInt32}   │
│ sqft_above    │ Continuous        │ Float64                           │
│ sqft_basement │ Continuous        │ Float64                           │
│ yr_built      │ Continuous        │ Float64                           │
│ zipcode       │ Multiclass{70}    │ CategoricalValue{Int64, UInt32}   │
│ lat           │ Continuous        │ Float64                           │
│ long          │ Continuous        │ Float64                           │
│ sqft_living15 │ Continuous        │ Float64                           │
│ sqft_lot15    │ Continuous        │ Float64                           │
│ is_renovated  │ OrderedFactor{2}  │ CategoricalValue{Bool, UInt32}    │
└───────────────┴───────────────────┴───────────────────────────────────┘

````

(a) Define a pipeline that:
- uses `Standardizer` to ensure that features that are already
  continuous are centered at zero and have unit variance
- re-encodes the full set of features as `Continuous`, using
  `ContinuousEncoder`
- uses the `KMeans` clustering model from `Clustering.jl`
  to reduce the dimension of the feature space to `k=10`.
- trains a `EvoTreeClassifier` (a gradient tree boosting
  algorithm in `EvoTrees.jl`) on the reduced data, using
  `nrounds=50` and default values for the other
   hyperparameters

(b) Evaluate the pipeline on all data, using 6-fold cross-validation
and `cross_entropy` loss.

(c) Plot a learning curve which examines the effect on this loss
as the tree booster parameter `max_depth` varies from 2 to 10.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

