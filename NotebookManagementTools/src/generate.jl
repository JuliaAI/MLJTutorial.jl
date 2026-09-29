ERR_BAD_SCRIPT(file) = ErrorException(
    "$file appears to be missing, or is not a regular file. "
)

INFO_MISSING_TESTS(name) = "No runtests.jl file found. Skipping tests for $name. "

"""
    generate(notebook_dir, path_to_literate=path_to_literate(); tests=true)

Attempt to generate a new markdown file from the file named "notebook.jl" and contained in
`notebook_dir`, using Literate.jl and the project at `notebook_dir`, after activating and
instantiating the Julia project in `notebook_dir`. A version of Literate.jl is used
consistent with the project specified at `path_to_literate` (which is pushed to
`LOAD_PATH` to make it available).

Literate.jl executes all code cells and wraps the input cell in regular julia fencing. In
that way, when the markdown is included in a Documenter.jl project, no code is
re-executed.

# Testing

If `tests==true`, then, *before* attempting markdown generation, run, in a new Julia
process, the code appearing in the file "runtests.jl" appearing in the directory
`notebook_dir`, which should additionally contain the julia script `notebook.jl`.

!!! note

    The test file `runtests.jl` should begin with "using Test; include("notebook.jl")" and
    end with "true".



# Return value

Returns the path to the generated markdown, unless the Julia process exits abnormally, in
which case an empty string is returned.

"""
function generate(
    notebook_dir,
    path_to_literate=NotebookManagementTools.path_to_literate();
    tests=true,
    )
    name = splitpath(notebook_dir) |> last

    script_file = joinpath(notebook_dir, "notebook.jl")
    markdown_file = joinpath(notebook_dir, "notebook.md")
    isfile(script_file) || throw(ERR_BAD_SCRIPT(script_file))

    test_file = joinpath(notebook_dir, "runtests.jl")
    test_file_exists = isfile(test_file)
    test_file_exists || @info(INFO_MISSING_TESTS(name))

    success = true

    testing = tests && test_file_exists

    # Code to be executed in new remote process must be split into two remotecall's to
    # prevent World Age issues (`using Test` cannot be in same call as `@testset`, etc).

    setup = quote
        # Next if-end block makes sure the standard library is available.
        # When calling generate(…) from
        # NotebookManagementTools/test/runtests.jl,
        # the standard library
        # is mysteriously disappearing from LOAD_PATH here so that Pkg is not
        # available.

        if !("@stdlib" in LOAD_PATH)
            push!(LOAD_PATH, "@stdlib")
        end

        using Pkg
        Pkg.activate($notebook_dir)
        push!(LOAD_PATH, $path_to_literate)

        # warn about missing Literate:
        if VERSION < v"1.12"
            literate_lost = !haskey(Pkg.project().dependencies, "Literate")
            literate_lost && @warn "You are generating notebook markdown using "*
                "julia version < 1.12. You may need to explicitly "*
                "add Literate to your notebook projects. "
        end

        Pkg.instantiate()

        using Literate
        using Test
    end

    program = quote
        if $testing
            @info "Testing "*$name*"."
            @testset $name begin
                include($test_file)
                Test.@test true
            end
        end

        @info "Generating markdown for "*$name*"."
        Literate.markdown(
            $script_file,
            $notebook_dir,
            execute=true,
            config=Dict("codefence" => Pair("````@julia", "````" )),
        )
    end

    id = addprocs(1) |> only
    future = try
        future1 = remotecall(Main.eval, id, setup)
        fetch(future1)
        future2 = remotecall(Main.eval, id, program)
        fetch(future2)
        rmprocs(id)
    catch ex
        success = false
        rmprocs(id)
        ex isa RemoteException || rethrow(ex)
    end
    return success ? markdown_file : ""
end

"""
    generate(notebook_dirs::AbstractVector, path_to_literate=path_to_literate(); tests=true)

Attempts to generate each notebook with directory in the provided list
`notebook_dirs`. Returns paths to re-generated markdown files, or `nothing` if there is a
failed notebook test or markdown generation failure.

"""
function generate(
    notebook_dirs::AbstractVector,
    path_to_literate=NotebookManagementTools.path_to_literate();
    kwargs...,
    )
    generated_files =  map(notebook_dirs) do dir
        generate(dir, path_to_literate; kwargs...)
    end
    "" in generated_files && return nothing
    filter(generated_files) do file
        !isempty(file)
    end
end
