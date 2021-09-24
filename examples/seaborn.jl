### A Pluto.jl notebook ###
# v0.16.1

using Markdown
using InteractiveUtils

# ╔═╡ e9dcc3e9-7c0f-40aa-b633-1242aacca66c
using Pkg; Pkg.activate("tmp", shared=true)

# ╔═╡ 545b2750-1bc5-11ec-1d6a-275133eec22d
using PythonCall, RDatasets

# ╔═╡ dbf0b8fc-d9d1-492a-892e-79f1ea86c345
iris = dataset("datasets", "iris");

# ╔═╡ 41f6b405-3436-4058-979e-bcf7be661e9a
sns = pyimport("seaborn"); sns.set_theme();

# ╔═╡ 9f8ddcfe-778e-4e33-9308-c4acb409770f
sns.pairplot(pytable(iris), hue="Species")

# ╔═╡ Cell order:
# ╠═e9dcc3e9-7c0f-40aa-b633-1242aacca66c
# ╠═545b2750-1bc5-11ec-1d6a-275133eec22d
# ╠═dbf0b8fc-d9d1-492a-892e-79f1ea86c345
# ╠═41f6b405-3436-4058-979e-bcf7be661e9a
# ╠═9f8ddcfe-778e-4e33-9308-c4acb409770f
