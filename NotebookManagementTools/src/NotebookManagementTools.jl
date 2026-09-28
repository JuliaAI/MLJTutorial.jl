"""
    NotebookManagementTools

Module providing tools for generating markdown from Literate.jl-compliant julia scripts,
after testing those scripts.

Intended to facilitate the separate execution and generation of notebooks that are part of
a larger Documenter.jl documentation deployment. In this scenario, notebook "ground truth"
is always a julia script (with Literate.jl-compatible narrative). Furthermore, when used
with appropriate continuous integration workflows, notebooks can be generated on a
need-to basis, to speed up document generation and the development of new tutorials.

The kind of repository we have in mind will be structured using Julia's "workspace"
package management structure. For a stand-alone collection of tutorials following Julia's
workspace package pattern, this might look something like this:

```
├── Project.toml     # <---- root project
├── docs
│   ├── make.jl
│   ├── Project.toml # <--- project for generating documentation wrapping the notebooks
│   └── src
│       └── notebooks
│           ├── tutorial1
│           │   ├── Project.toml  # <--- tutorial-specific project
│           │   ├── runtests.jl
│           │   └── notebook.jl
│           └── tutorial2
│               ├── Project.toml
│               ├── runtests.jl
│               ├── notebook.jl
│               └── notebook.md
```

!!! note

    If you are not using Julia 1.12 or higher with a workspace project structure, you will
    need to explicitly add Literate to each notebook's project.


# Tools

- [`generate`](@ref): The main tool, used to generate markdown from annotated julia script.

- [`notebook_dirs`](@ref): Used in continuous integration to list notebook directories.

- [`notebook_dirs_containing`](@ref): Used in continuous integration to find tutorial
  directories containing files known to have changed in a pull-request.

# Advanced Tools

- [`set_path_to_literate`](@ref): For pointing the generator to a version of Literate.jl
  different from the one in the NotebookManagementTools project.

- [`path_to_literate`](@ref): For inspecting the above.


"""
module NotebookManagementTools

using Distributed

const NOTEBOOK_MANAGEMENT_TOOLS_PATH = joinpath(@__DIR__, "..")

include("utilities.jl")
include("path_to_literate.jl")
include("generate.jl")
include("init.jl")

export generate, notebook_dirs_containing, notebook_dirs,
    set_path_to_literate, path_to_literate

end # module
