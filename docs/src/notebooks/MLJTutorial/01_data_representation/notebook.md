```@meta
EditURL = "notebook.jl"
```

# Tutorial 1. Data Representation

> **Goals:**
> 1. Learn how MLJ specifies it's data requirements using "scientific" types
> 2. Understand the options for representing tabular data
> 3. Learn how to inspect and fix the representation of data to meet MLJ requirements

To run the code in this tutorial in a live Julia session, first follow the instructions
given [here](@ref instructions).

### Scientific types

To help you focus on the intended *purpose* or *interpretation* of data, MLJ models
specify data requirements using *scientific types*, instead of machine types. An example
of a scientific type is `OrderedFactor`. The other basic "scalar" scientific types are
illustrated below:

![](scitypes.svg)

A scientific type is an ordinary Julia type (so it can be used for method dispatch, for
example) but it usually has no instances. The `scitype` function is used to articulate
MLJ's convention about how different machine types will be interpreted by MLJ models:

````@julia
using ScientificTypes
scitype(3.14)
````

````
ScientificTypesBase.Continuous
````

````@julia
time = [2.3, 4.5, 4.2, 1.8, 7.1]
scitype(time)
````

````
AbstractVector{Continuous} (alias for AbstractArray{ScientificTypesBase.Continuous, 1})
````

To fix data which MLJ is interpreting incorrectly, we use the
`coerce` method:

````@julia
height = [185, 153, 163, 114, 180]
scitype(height)
````

````
AbstractVector{Count} (alias for AbstractArray{ScientificTypesBase.Count, 1})
````

````@julia
height = coerce(height, Continuous)
````

````
5-element Vector{Float64}:
 185.0
 153.0
 163.0
 114.0
 180.0
````

Here's an example of data we would want interpreted as
`OrderedFactor` but isn't:

````@julia
exam_mark = ["rotten", "great", "bla",  missing, "great"]
scitype(exam_mark)
````

````
AbstractVector{Union{Missing, Textual}} (alias for AbstractArray{Union{Missing, ScientificTypesBase.Textual}, 1})
````

````@julia
exam_mark = coerce(exam_mark, OrderedFactor)
````

````
5-element CategoricalArrays.CategoricalArray{Union{Missing, String},1,UInt32}:
 "rotten"
 "great"
 "bla"
 missing
 "great"
````

````@julia
levels(exam_mark)
````

````
3-element CategoricalArrays.CategoricalArray{String,1,UInt32}:
 "bla"
 "great"
 "rotten"
````

Use `levels!` to put the classes in the right order:

````@julia
levels!(exam_mark, ["rotten", "bla", "great"])
exam_mark[1] < exam_mark[2]
````

````
true
````

When sub-sampling, no levels are lost:

````@julia
levels(exam_mark[1:2])
````

````
3-element CategoricalArrays.CategoricalArray{String,1,UInt32}:
 "rotten"
 "bla"
 "great"
````

**Note on binary data.** There is no separate scientific type for binary data. Binary
data is `OrderedFactor{2}` or `Multiclass{2}`. If a binary measure like `truepositive`
is a applied to `OrderedFactor{2}` then the "positive" class is assumed to appear
*second* in the ordering. If such a measure is applied to `Multiclass{2}` data, a
warning is issued. A single `OrderedFactor` can be coerced to a single `Continuous`
variable, for models that require this, while a `Multiclass` variable can only be
one-hot encoded.

See also [Working with Categorical
Data](https://juliaai.github.io/MLJ.jl/stable/working_with_categorical_data/#Working-with-Categorical-Data)
from the MLJ manual.

### Two-dimensional data

Whenever it makes sense, MLJ Models generally expect two-dimensional data to be
*tabular*. Most tabular formats implementing the [Tables.jl
API](https://juliadata.github.io/Tables.jl/stable/) (see this
[list](https://github.com/JuliaData/Tables.jl/blob/master/INTEGRATIONS.md)) have a
scientific type of `Table` and can be used with such models.

Probably the simplest example of a table is the julia native *column
table*, which is just a named tuple of equal-length vectors:

````@julia
column_table = (h=height, e=exam_mark, t=time)
````

````
(h = [185.0, 153.0, 163.0, 114.0, 180.0], e = Union{Missing, CategoricalArrays.CategoricalValue{String, UInt32}}[CategoricalValue(CategoricalArrays.CategoricalPool{String, UInt32}(["rotten", "bla", "great"], true), 1), CategoricalValue(CategoricalArrays.CategoricalPool{String, UInt32}(["rotten", "bla", "great"], true), 3), CategoricalValue(CategoricalArrays.CategoricalPool{String, UInt32}(["rotten", "bla", "great"], true), 2), missing, CategoricalValue(CategoricalArrays.CategoricalPool{String, UInt32}(["rotten", "bla", "great"], true), 3)], t = [2.3, 4.5, 4.2, 1.8, 7.1])
````

While a table has a `scitype`, the general user will want to inspect column scitypes
using MLJ's `schema` method:

````@julia
schema(column_table)
````

````
┌───────┬──────────────────────────────────┬────────────────────────────────────
│ names │ scitypes                         │ types                             ⋯
├───────┼──────────────────────────────────┼────────────────────────────────────
│ h     │ Continuous                       │ Float64                           ⋯
│ e     │ Union{Missing, OrderedFactor{3}} │ Union{Missing, CategoricalValue{S ⋯
│ t     │ Continuous                       │ Float64                           ⋯
└───────┴──────────────────────────────────┴────────────────────────────────────
                                                                1 column omitted

````

Here are other examples of tables:

````@julia
dict_table = Dict(:h => height, :e => exam_mark, :t => time)
schema(dict_table)
````

````
┌───────┬──────────────────────────────────┬────────────────────────────────────
│ names │ scitypes                         │ types                             ⋯
├───────┼──────────────────────────────────┼────────────────────────────────────
│ e     │ Union{Missing, OrderedFactor{3}} │ Union{Missing, CategoricalValue{S ⋯
│ h     │ Continuous                       │ Float64                           ⋯
│ t     │ Continuous                       │ Float64                           ⋯
└───────┴──────────────────────────────────┴────────────────────────────────────
                                                                1 column omitted

````

(To control column order here, instead use `LittleDict` from
OrderedCollections.jl.)

````@julia
row_table = [(a=1, b=3.4),
             (a=2, b=4.5),
             (a=3, b=5.6)]
schema(row_table)
````

````
┌───────┬────────────┬─────────┐
│ names │ scitypes   │ types   │
├───────┼────────────┼─────────┤
│ a     │ Count      │ Int64   │
│ b     │ Continuous │ Float64 │
└───────┴────────────┴─────────┘

````

````@julia
import DataFrames
df = DataFrames.DataFrame(column_table)
````

```@raw html
<div><div style = "float: left;"><span>5×3 DataFrame</span></div><div style = "clear: both;"></div></div><div class = "data-frame" style = "overflow-x: scroll;"><table class = "data-frame" style = "margin-bottom: 6px;"><thead><tr class = "columnLabelRow"><th class = "stubheadLabel" style = "font-weight: bold; text-align: right;">Row</th><th style = "text-align: left;">h</th><th style = "text-align: left;">e</th><th style = "text-align: left;">t</th></tr><tr class = "columnLabelRow"><th class = "stubheadLabel" style = "font-weight: bold; text-align: right;"></th><th title = "Float64" style = "text-align: left;">Float64</th><th title = "Union{Missing, CategoricalArrays.CategoricalValue{String, UInt32}}" style = "text-align: left;">Cat…?</th><th title = "Float64" style = "text-align: left;">Float64</th></tr></thead><tbody><tr class = "dataRow"><td class = "rowLabel" style = "font-weight: bold; text-align: right;">1</td><td style = "text-align: right;">185.0</td><td style = "text-align: left;">rotten</td><td style = "text-align: right;">2.3</td></tr><tr class = "dataRow"><td class = "rowLabel" style = "font-weight: bold; text-align: right;">2</td><td style = "text-align: right;">153.0</td><td style = "text-align: left;">great</td><td style = "text-align: right;">4.5</td></tr><tr class = "dataRow"><td class = "rowLabel" style = "font-weight: bold; text-align: right;">3</td><td style = "text-align: right;">163.0</td><td style = "text-align: left;">bla</td><td style = "text-align: right;">4.2</td></tr><tr class = "dataRow"><td class = "rowLabel" style = "font-weight: bold; text-align: right;">4</td><td style = "text-align: right;">114.0</td><td style = "font-style: italic; text-align: left;">missing</td><td style = "text-align: right;">1.8</td></tr><tr class = "dataRow"><td class = "rowLabel" style = "font-weight: bold; text-align: right;">5</td><td style = "text-align: right;">180.0</td><td style = "text-align: left;">great</td><td style = "text-align: right;">7.1</td></tr></tbody></table></div>
```

````@julia
schema(df)
````

````
┌───────┬──────────────────────────────────┬────────────────────────────────────
│ names │ scitypes                         │ types                             ⋯
├───────┼──────────────────────────────────┼────────────────────────────────────
│ h     │ Continuous                       │ Float64                           ⋯
│ e     │ Union{Missing, OrderedFactor{3}} │ Union{Missing, CategoricalValue{S ⋯
│ t     │ Continuous                       │ Float64                           ⋯
└───────┴──────────────────────────────────┴────────────────────────────────────
                                                                1 column omitted

````

A schema is itself a table. If we convert it to a dataframe, we can get a nicer display
in some contexts (e.g., in documentation or a jupyter notebook):

````@julia
schema(df) |> DataFrames.DataFrame
````

```@raw html
<div><div style = "float: left;"><span>3×3 DataFrame</span></div><div style = "clear: both;"></div></div><div class = "data-frame" style = "overflow-x: scroll;"><table class = "data-frame" style = "margin-bottom: 6px;"><thead><tr class = "columnLabelRow"><th class = "stubheadLabel" style = "font-weight: bold; text-align: right;">Row</th><th style = "text-align: left;">names</th><th style = "text-align: left;">scitypes</th><th style = "text-align: left;">types</th></tr><tr class = "columnLabelRow"><th class = "stubheadLabel" style = "font-weight: bold; text-align: right;"></th><th title = "Symbol" style = "text-align: left;">Symbol</th><th title = "Type" style = "text-align: left;">Type</th><th title = "Type" style = "text-align: left;">Type</th></tr></thead><tbody><tr class = "dataRow"><td class = "rowLabel" style = "font-weight: bold; text-align: right;">1</td><td style = "text-align: left;">h</td><td style = "text-align: left;">Continuous</td><td style = "text-align: left;">Float64</td></tr><tr class = "dataRow"><td class = "rowLabel" style = "font-weight: bold; text-align: right;">2</td><td style = "text-align: left;">e</td><td style = "text-align: left;">Union{Missing, OrderedFactor{3}}</td><td style = "text-align: left;">Union{Missing, CategoricalValue{String, UInt32}}</td></tr><tr class = "dataRow"><td class = "rowLabel" style = "font-weight: bold; text-align: right;">3</td><td style = "text-align: left;">t</td><td style = "text-align: left;">Continuous</td><td style = "text-align: left;">Float64</td></tr></tbody></table></div>
```

Most MLJ models do not accept a matrix in lieu of a table, but you can
wrap a matrix as a table:

````@julia
using Tables
matrix_table = Tables.table(rand(2,3))
schema(matrix_table)
````

````
┌─────────┬────────────┬─────────┐
│ names   │ scitypes   │ types   │
├─────────┼────────────┼─────────┤
│ Column1 │ Continuous │ Float64 │
│ Column2 │ Continuous │ Float64 │
│ Column3 │ Continuous │ Float64 │
└─────────┴────────────┴─────────┘

````

### Fixing scientific types in tabular data

To show how we can correct the scientific types of data in tables, let's look more
closely at a cleaned up version of the UCI Horse Colic Data set. (The cleaning work-flow
is described
[here](https://juliaai.github.io/DataScienceTutorials.jl/end-to-end/horse/#dealing_with_missing_values).)

````@julia
import Downloads
import CSV
url = "https://raw.githubusercontent.com/ablaom/"*
    "MachineLearningInJulia2020/"*
    "for-MLJ-version-0.16/data/horse.csv"
csv_file = Downloads.download(url)
````

````
"/tmp/jl_kOThHT/horse.csv"
````

Entering these lines of code downloads the data to a temporary file at the location
shown above. We'll read this data into memory as a dataframe, provided by the
DataFrames.jl package; see [this
tutorial](https://juliaai.github.io/DataScienceTutorials.jl/data/dataframe/) for a
quick-start introduction.

````@julia
horse = CSV.read(csv_file, DataFrames.DataFrame)
first(horse, 4)
````

```@raw html
<div><div style = "float: left;"><span>4×16 DataFrame</span></div><div style = "clear: both;"></div></div><div class = "data-frame" style = "overflow-x: scroll;"><table class = "data-frame" style = "margin-bottom: 6px;"><thead><tr class = "columnLabelRow"><th class = "stubheadLabel" style = "font-weight: bold; text-align: right;">Row</th><th style = "text-align: left;">surgery</th><th style = "text-align: left;">age</th><th style = "text-align: left;">rectal_temperature</th><th style = "text-align: left;">pulse</th><th style = "text-align: left;">respiratory_rate</th><th style = "text-align: left;">temperature_extremities</th><th style = "text-align: left;">mucous_membranes</th><th style = "text-align: left;">capillary_refill_time</th><th style = "text-align: left;">pain</th><th style = "text-align: left;">peristalsis</th><th style = "text-align: left;">abdominal_distension</th><th style = "text-align: left;">packed_cell_volume</th><th style = "text-align: left;">total_protein</th><th style = "text-align: left;">outcome</th><th style = "text-align: left;">surgical_lesion</th><th style = "text-align: left;">cp_data</th></tr><tr class = "columnLabelRow"><th class = "stubheadLabel" style = "font-weight: bold; text-align: right;"></th><th title = "Int64" style = "text-align: left;">Int64</th><th title = "Int64" style = "text-align: left;">Int64</th><th title = "Float64" style = "text-align: left;">Float64</th><th title = "Int64" style = "text-align: left;">Int64</th><th title = "Int64" style = "text-align: left;">Int64</th><th title = "Int64" style = "text-align: left;">Int64</th><th title = "Int64" style = "text-align: left;">Int64</th><th title = "Int64" style = "text-align: left;">Int64</th><th title = "Int64" style = "text-align: left;">Int64</th><th title = "Int64" style = "text-align: left;">Int64</th><th title = "Int64" style = "text-align: left;">Int64</th><th title = "Float64" style = "text-align: left;">Float64</th><th title = "Float64" style = "text-align: left;">Float64</th><th title = "Int64" style = "text-align: left;">Int64</th><th title = "Int64" style = "text-align: left;">Int64</th><th title = "Int64" style = "text-align: left;">Int64</th></tr></thead><tbody><tr class = "dataRow"><td class = "rowLabel" style = "font-weight: bold; text-align: right;">1</td><td style = "text-align: right;">2</td><td style = "text-align: right;">1</td><td style = "text-align: right;">38.5</td><td style = "text-align: right;">66</td><td style = "text-align: right;">66</td><td style = "text-align: right;">3</td><td style = "text-align: right;">1</td><td style = "text-align: right;">2</td><td style = "text-align: right;">5</td><td style = "text-align: right;">4</td><td style = "text-align: right;">4</td><td style = "text-align: right;">45.0</td><td style = "text-align: right;">8.4</td><td style = "text-align: right;">2</td><td style = "text-align: right;">2</td><td style = "text-align: right;">2</td></tr><tr class = "dataRow"><td class = "rowLabel" style = "font-weight: bold; text-align: right;">2</td><td style = "text-align: right;">1</td><td style = "text-align: right;">1</td><td style = "text-align: right;">39.2</td><td style = "text-align: right;">88</td><td style = "text-align: right;">88</td><td style = "text-align: right;">3</td><td style = "text-align: right;">4</td><td style = "text-align: right;">1</td><td style = "text-align: right;">3</td><td style = "text-align: right;">4</td><td style = "text-align: right;">2</td><td style = "text-align: right;">50.0</td><td style = "text-align: right;">85.0</td><td style = "text-align: right;">3</td><td style = "text-align: right;">2</td><td style = "text-align: right;">2</td></tr><tr class = "dataRow"><td class = "rowLabel" style = "font-weight: bold; text-align: right;">3</td><td style = "text-align: right;">2</td><td style = "text-align: right;">1</td><td style = "text-align: right;">38.3</td><td style = "text-align: right;">40</td><td style = "text-align: right;">40</td><td style = "text-align: right;">1</td><td style = "text-align: right;">3</td><td style = "text-align: right;">1</td><td style = "text-align: right;">3</td><td style = "text-align: right;">3</td><td style = "text-align: right;">1</td><td style = "text-align: right;">33.0</td><td style = "text-align: right;">6.7</td><td style = "text-align: right;">1</td><td style = "text-align: right;">2</td><td style = "text-align: right;">1</td></tr><tr class = "dataRow"><td class = "rowLabel" style = "font-weight: bold; text-align: right;">4</td><td style = "text-align: right;">1</td><td style = "text-align: right;">9</td><td style = "text-align: right;">39.1</td><td style = "text-align: right;">164</td><td style = "text-align: right;">164</td><td style = "text-align: right;">4</td><td style = "text-align: right;">6</td><td style = "text-align: right;">2</td><td style = "text-align: right;">2</td><td style = "text-align: right;">4</td><td style = "text-align: right;">4</td><td style = "text-align: right;">48.0</td><td style = "text-align: right;">7.2</td><td style = "text-align: right;">2</td><td style = "text-align: right;">1</td><td style = "text-align: right;">1</td></tr></tbody></table></div>
```

From [the UCI
docs](http://archive.ics.uci.edu/ml/datasets/Horse+Colic) we can
surmise how each variable ought to be interpreted (a step in our
work-flow that cannot reliably be left to the computer):

variable                    | scientific type (interpretation)
----------------------------|-----------------------------------
`:surgery`                  | Multiclass
`:age`                      | Multiclass
`:rectal_temperature`       | Continuous
`:pulse`                    | Continuous
`:respiratory_rate`         | Continuous
`:temperature_extremities`  | OrderedFactor
`:mucous_membranes`         | Multiclass
`:capillary_refill_time`    | Multiclass
`:pain`                     | OrderedFactor
`:peristalsis`              | OrderedFactor
`:abdominal_distension`     | OrderedFactor
`:packed_cell_volume`       | Continuous
`:total_protein`            | Continuous
`:outcome`                  | Multiclass
`:surgical_lesion`          | OrderedFactor
`:cp_data`                  | Multiclass

Let's see how MLJ will actually interpret the data, as it is
currently encoded:

````@julia
schema(horse)
````

````
┌─────────────────────────┬────────────┬─────────┐
│ names                   │ scitypes   │ types   │
├─────────────────────────┼────────────┼─────────┤
│ surgery                 │ Count      │ Int64   │
│ age                     │ Count      │ Int64   │
│ rectal_temperature      │ Continuous │ Float64 │
│ pulse                   │ Count      │ Int64   │
│ respiratory_rate        │ Count      │ Int64   │
│ temperature_extremities │ Count      │ Int64   │
│ mucous_membranes        │ Count      │ Int64   │
│ capillary_refill_time   │ Count      │ Int64   │
│ pain                    │ Count      │ Int64   │
│ peristalsis             │ Count      │ Int64   │
│ abdominal_distension    │ Count      │ Int64   │
│ packed_cell_volume      │ Continuous │ Float64 │
│ total_protein           │ Continuous │ Float64 │
│ outcome                 │ Count      │ Int64   │
│ surgical_lesion         │ Count      │ Int64   │
│ cp_data                 │ Count      │ Int64   │
└─────────────────────────┴────────────┴─────────┘

````

As a first correction step, we can get MLJ to "guess" the
appropriate fix, using the `autotype` method:

````@julia
autotype(horse)
````

````
Dict{Symbol, Type} with 11 entries:
  :abdominal_distension => OrderedFactor
  :pain => OrderedFactor
  :surgery => OrderedFactor
  :mucous_membranes => OrderedFactor
  :surgical_lesion => OrderedFactor
  :outcome => OrderedFactor
  :capillary_refill_time => OrderedFactor
  :age => OrderedFactor
  :temperature_extremities => OrderedFactor
  :peristalsis => OrderedFactor
  :cp_data => OrderedFactor
````

Okay, this is not perfect, but a step in the right direction, which
we implement like this:

````@julia
coerce!(horse, autotype(horse));
schema(horse)
````

````
┌─────────────────────────┬──────────────────┬─────────────────────────────────┐
│ names                   │ scitypes         │ types                           │
├─────────────────────────┼──────────────────┼─────────────────────────────────┤
│ surgery                 │ OrderedFactor{2} │ CategoricalValue{Int64, UInt32} │
│ age                     │ OrderedFactor{2} │ CategoricalValue{Int64, UInt32} │
│ rectal_temperature      │ Continuous       │ Float64                         │
│ pulse                   │ Count            │ Int64                           │
│ respiratory_rate        │ Count            │ Int64                           │
│ temperature_extremities │ OrderedFactor{4} │ CategoricalValue{Int64, UInt32} │
│ mucous_membranes        │ OrderedFactor{6} │ CategoricalValue{Int64, UInt32} │
│ capillary_refill_time   │ OrderedFactor{3} │ CategoricalValue{Int64, UInt32} │
│ pain                    │ OrderedFactor{5} │ CategoricalValue{Int64, UInt32} │
│ peristalsis             │ OrderedFactor{4} │ CategoricalValue{Int64, UInt32} │
│ abdominal_distension    │ OrderedFactor{4} │ CategoricalValue{Int64, UInt32} │
│ packed_cell_volume      │ Continuous       │ Float64                         │
│ total_protein           │ Continuous       │ Float64                         │
│ outcome                 │ OrderedFactor{3} │ CategoricalValue{Int64, UInt32} │
│ surgical_lesion         │ OrderedFactor{2} │ CategoricalValue{Int64, UInt32} │
│ cp_data                 │ OrderedFactor{2} │ CategoricalValue{Int64, UInt32} │
└─────────────────────────┴──────────────────┴─────────────────────────────────┘

````

All remaining `Count` data should be `Continuous`:

````@julia
coerce!(horse, Count => Continuous);
schema(horse)
````

````
┌─────────────────────────┬──────────────────┬─────────────────────────────────┐
│ names                   │ scitypes         │ types                           │
├─────────────────────────┼──────────────────┼─────────────────────────────────┤
│ surgery                 │ OrderedFactor{2} │ CategoricalValue{Int64, UInt32} │
│ age                     │ OrderedFactor{2} │ CategoricalValue{Int64, UInt32} │
│ rectal_temperature      │ Continuous       │ Float64                         │
│ pulse                   │ Continuous       │ Float64                         │
│ respiratory_rate        │ Continuous       │ Float64                         │
│ temperature_extremities │ OrderedFactor{4} │ CategoricalValue{Int64, UInt32} │
│ mucous_membranes        │ OrderedFactor{6} │ CategoricalValue{Int64, UInt32} │
│ capillary_refill_time   │ OrderedFactor{3} │ CategoricalValue{Int64, UInt32} │
│ pain                    │ OrderedFactor{5} │ CategoricalValue{Int64, UInt32} │
│ peristalsis             │ OrderedFactor{4} │ CategoricalValue{Int64, UInt32} │
│ abdominal_distension    │ OrderedFactor{4} │ CategoricalValue{Int64, UInt32} │
│ packed_cell_volume      │ Continuous       │ Float64                         │
│ total_protein           │ Continuous       │ Float64                         │
│ outcome                 │ OrderedFactor{3} │ CategoricalValue{Int64, UInt32} │
│ surgical_lesion         │ OrderedFactor{2} │ CategoricalValue{Int64, UInt32} │
│ cp_data                 │ OrderedFactor{2} │ CategoricalValue{Int64, UInt32} │
└─────────────────────────┴──────────────────┴─────────────────────────────────┘

````

We'll correct the remaining truant entries manually:

````@julia
coerce!(horse,
        :surgery               => Multiclass,
        :age                   => Multiclass,
        :mucous_membranes      => Multiclass,
        :capillary_refill_time => Multiclass,
        :outcome               => Multiclass,
        :cp_data               => Multiclass);
schema(horse)
````

````
┌─────────────────────────┬──────────────────┬─────────────────────────────────┐
│ names                   │ scitypes         │ types                           │
├─────────────────────────┼──────────────────┼─────────────────────────────────┤
│ surgery                 │ Multiclass{2}    │ CategoricalValue{Int64, UInt32} │
│ age                     │ Multiclass{2}    │ CategoricalValue{Int64, UInt32} │
│ rectal_temperature      │ Continuous       │ Float64                         │
│ pulse                   │ Continuous       │ Float64                         │
│ respiratory_rate        │ Continuous       │ Float64                         │
│ temperature_extremities │ OrderedFactor{4} │ CategoricalValue{Int64, UInt32} │
│ mucous_membranes        │ Multiclass{6}    │ CategoricalValue{Int64, UInt32} │
│ capillary_refill_time   │ Multiclass{3}    │ CategoricalValue{Int64, UInt32} │
│ pain                    │ OrderedFactor{5} │ CategoricalValue{Int64, UInt32} │
│ peristalsis             │ OrderedFactor{4} │ CategoricalValue{Int64, UInt32} │
│ abdominal_distension    │ OrderedFactor{4} │ CategoricalValue{Int64, UInt32} │
│ packed_cell_volume      │ Continuous       │ Float64                         │
│ total_protein           │ Continuous       │ Float64                         │
│ outcome                 │ Multiclass{3}    │ CategoricalValue{Int64, UInt32} │
│ surgical_lesion         │ OrderedFactor{2} │ CategoricalValue{Int64, UInt32} │
│ cp_data                 │ Multiclass{2}    │ CategoricalValue{Int64, UInt32} │
└─────────────────────────┴──────────────────┴─────────────────────────────────┘

````

### Resources for this tutorial

- From the MLJ manual:
  - [A preview of data type specification in
    MLJ](https://juliaai.github.io/MLJ.jl/dev/getting_started/#A-preview-of-data-type-specification-in-MLJ-1)
  - [Data containers and scientific
    types](https://juliaai.github.io/MLJ.jl/dev/getting_started/#Data-containers-and-scientific-types-1)
  - [Working with Categorical Data](https://juliaai.github.io/MLJ.jl/dev/working_with_categorical_data/)
- [Summary](https://juliaai.github.io/ScientificTypes.jl/dev/#Summary-of-the-default-convention) of the MLJ convention for representing scientific types
- [ScientificTypes.jl](https://juliaai.github.io/ScientificTypes.jl/dev/)
- From Data Science Tutorials:
  - [Data interpretation: Scientific Types](https://juliaai.github.io/DataScienceTutorials.jl/data/scitype/)
  - [Horse colic data](https://juliaai.github.io/DataScienceTutorials.jl/end-to-end/horse/) [UCI Horse Colic Data Set](http://archive.ics.uci.edu/ml/datasets/Horse+Colic)
- [Documentation](https://juliaai.github.io/StatisticalMeasures.jl/stable/) on
  performance metrics.

### Tutorial 1 Exercises

#### Exercise 1

Try to guess how each code snippet below will evaluate:

````@julia
scitype(42);
````

````@julia
questions = ["who", "why", "what", "when"]
scitype(questions);
````

````@julia
elscitype(questions);
````

````@julia
t = (3.141, 42, "how")
scitype(t);
````

````@julia
A = rand(2, 3)
````

````
2×3 Matrix{Float64}:
 0.832781  0.999535  0.163212
 0.685279  0.998338  0.966773
````

````@julia
scitype(A);
````

````@julia
elscitype(A);
````

````@julia
using SparseArrays
Asparse = sparse(A)
````

````
2×3 SparseArrays.SparseMatrixCSC{Float64, Int64} with 6 stored entries:
 0.832781  0.999535  0.163212
 0.685279  0.998338  0.966773
````

````@julia
scitype(Asparse);
````

````@julia
C = coerce(A, Multiclass);
````

````@julia
scitype(C);
````

````@julia
elscitype(C);
````

````@julia
v = [1, 2, missing, 4]
scitype(v);
````

````@julia
elscitype(v);
````

````@julia
scitype(v[1:2]);
````

Can you guess at the general behavior of
`scitype` with respect to tuples, abstract arrays and missing
values? The answers are
[here](https://github.com/juliaai/ScientificTypesBase.jl#2-the-scitype-and-scitype-methods)
(ignore "Property 1").

#### Exercise 2

Coerce the following vector to make MLJ recognize it as a vector of
ordered factors (with an appropriate ordering):

````@julia
quality = ["good", "poor", "poor", "excellent", missing, "good", "excellent"];
````

#### Exercise 3 (fixing scitypes in a table)

Fix the scitypes for the [House Prices in King
County](https://mlr3gallery.mlr-org.com/posts/2020-01-30-house-prices-in-king-county/)
dataset:

````@julia
url = "https://raw.githubusercontent.com/ablaom/"*
    "MachineLearningInJulia2020/for-MLJ-version-0.16/"*
    "data/house.csv";
house = CSV.read(Downloads.download(url), DataFrames.DataFrame)
first(house, 4)
````

```@raw html
<div><div style = "float: left;"><span>4×19 DataFrame</span></div><div style = "clear: both;"></div></div><div class = "data-frame" style = "overflow-x: scroll;"><table class = "data-frame" style = "margin-bottom: 6px;"><thead><tr class = "columnLabelRow"><th class = "stubheadLabel" style = "font-weight: bold; text-align: right;">Row</th><th style = "text-align: left;">price</th><th style = "text-align: left;">bedrooms</th><th style = "text-align: left;">bathrooms</th><th style = "text-align: left;">sqft_living</th><th style = "text-align: left;">sqft_lot</th><th style = "text-align: left;">floors</th><th style = "text-align: left;">waterfront</th><th style = "text-align: left;">view</th><th style = "text-align: left;">condition</th><th style = "text-align: left;">grade</th><th style = "text-align: left;">sqft_above</th><th style = "text-align: left;">sqft_basement</th><th style = "text-align: left;">yr_built</th><th style = "text-align: left;">zipcode</th><th style = "text-align: left;">lat</th><th style = "text-align: left;">long</th><th style = "text-align: left;">sqft_living15</th><th style = "text-align: left;">sqft_lot15</th><th style = "text-align: left;">is_renovated</th></tr><tr class = "columnLabelRow"><th class = "stubheadLabel" style = "font-weight: bold; text-align: right;"></th><th title = "Float64" style = "text-align: left;">Float64</th><th title = "Int64" style = "text-align: left;">Int64</th><th title = "Float64" style = "text-align: left;">Float64</th><th title = "Int64" style = "text-align: left;">Int64</th><th title = "Int64" style = "text-align: left;">Int64</th><th title = "Float64" style = "text-align: left;">Float64</th><th title = "Int64" style = "text-align: left;">Int64</th><th title = "Int64" style = "text-align: left;">Int64</th><th title = "Int64" style = "text-align: left;">Int64</th><th title = "Int64" style = "text-align: left;">Int64</th><th title = "Int64" style = "text-align: left;">Int64</th><th title = "Int64" style = "text-align: left;">Int64</th><th title = "Int64" style = "text-align: left;">Int64</th><th title = "Int64" style = "text-align: left;">Int64</th><th title = "Float64" style = "text-align: left;">Float64</th><th title = "Float64" style = "text-align: left;">Float64</th><th title = "Int64" style = "text-align: left;">Int64</th><th title = "Int64" style = "text-align: left;">Int64</th><th title = "Bool" style = "text-align: left;">Bool</th></tr></thead><tbody><tr class = "dataRow"><td class = "rowLabel" style = "font-weight: bold; text-align: right;">1</td><td style = "text-align: right;">221900.0</td><td style = "text-align: right;">3</td><td style = "text-align: right;">1.0</td><td style = "text-align: right;">1180</td><td style = "text-align: right;">5650</td><td style = "text-align: right;">1.0</td><td style = "text-align: right;">0</td><td style = "text-align: right;">0</td><td style = "text-align: right;">3</td><td style = "text-align: right;">7</td><td style = "text-align: right;">1180</td><td style = "text-align: right;">0</td><td style = "text-align: right;">1955</td><td style = "text-align: right;">98178</td><td style = "text-align: right;">47.5112</td><td style = "text-align: right;">-122.257</td><td style = "text-align: right;">1340</td><td style = "text-align: right;">5650</td><td style = "text-align: right;">true</td></tr><tr class = "dataRow"><td class = "rowLabel" style = "font-weight: bold; text-align: right;">2</td><td style = "text-align: right;">538000.0</td><td style = "text-align: right;">3</td><td style = "text-align: right;">2.25</td><td style = "text-align: right;">2570</td><td style = "text-align: right;">7242</td><td style = "text-align: right;">2.0</td><td style = "text-align: right;">0</td><td style = "text-align: right;">0</td><td style = "text-align: right;">3</td><td style = "text-align: right;">7</td><td style = "text-align: right;">2170</td><td style = "text-align: right;">400</td><td style = "text-align: right;">1951</td><td style = "text-align: right;">98125</td><td style = "text-align: right;">47.721</td><td style = "text-align: right;">-122.319</td><td style = "text-align: right;">1690</td><td style = "text-align: right;">7639</td><td style = "text-align: right;">false</td></tr><tr class = "dataRow"><td class = "rowLabel" style = "font-weight: bold; text-align: right;">3</td><td style = "text-align: right;">180000.0</td><td style = "text-align: right;">2</td><td style = "text-align: right;">1.0</td><td style = "text-align: right;">770</td><td style = "text-align: right;">10000</td><td style = "text-align: right;">1.0</td><td style = "text-align: right;">0</td><td style = "text-align: right;">0</td><td style = "text-align: right;">3</td><td style = "text-align: right;">6</td><td style = "text-align: right;">770</td><td style = "text-align: right;">0</td><td style = "text-align: right;">1933</td><td style = "text-align: right;">98028</td><td style = "text-align: right;">47.7379</td><td style = "text-align: right;">-122.233</td><td style = "text-align: right;">2720</td><td style = "text-align: right;">8062</td><td style = "text-align: right;">true</td></tr><tr class = "dataRow"><td class = "rowLabel" style = "font-weight: bold; text-align: right;">4</td><td style = "text-align: right;">604000.0</td><td style = "text-align: right;">4</td><td style = "text-align: right;">3.0</td><td style = "text-align: right;">1960</td><td style = "text-align: right;">5000</td><td style = "text-align: right;">1.0</td><td style = "text-align: right;">0</td><td style = "text-align: right;">0</td><td style = "text-align: right;">5</td><td style = "text-align: right;">7</td><td style = "text-align: right;">1050</td><td style = "text-align: right;">910</td><td style = "text-align: right;">1965</td><td style = "text-align: right;">98136</td><td style = "text-align: right;">47.5208</td><td style = "text-align: right;">-122.393</td><td style = "text-align: right;">1360</td><td style = "text-align: right;">5000</td><td style = "text-align: right;">true</td></tr></tbody></table></div>
```

(Two features in the original data set have been deemed uninformative
and dropped, namely `:id` and `:date`. The original feature
`:yr_renovated` has been replaced by the `Bool` feature `is_renovated`.)

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

