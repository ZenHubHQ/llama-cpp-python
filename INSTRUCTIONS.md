## Syncing with upstream repo

See [here](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork#syncing-a-fork-branch-from-the-web-ui) for more details. 

1. On the GitHub UI, create a new branch `repo-sync`, if the branch doesn't exist already.

2. Click on the "Sync fork" button and then click on the "Update branch" button. This will import all the commits from the upstream repo.

3. Create a local branch `repo-sync` and pull the contents from the remote `repo-sync` branch.

4. Solve for any conflicts if they arise. Otherwise, proceed to the next step.

5. Update all the git submodles:

```
git submodule update --recursive
```

6. Since changes have probably been made to the vendor libraries (`llama_cpp`, `kompute`), we need to recompile the `llama_cpp` package. Navigate to the `vendor/llama.cpp` folder and clean the build cache:

```
make clean
```
6. Navigate back to the root directory and type the following to recompile the `llama_cpp` package and build the dependenies again:

```
make deps && make build
```
7. Launch the `llama_cpp_python` server using the following command:
```
python -m llama_cpp.server --model $MODEL --n_gpu_layers -1
```
NOTE: Modify the launch arguments as needed. Make sure the `MODEL` environment variable points to an absolute path containing a `.gguf` model. 

8. If the server launches without issues, then you can proceed to create a PR with the latest changes