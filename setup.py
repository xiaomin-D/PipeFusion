from setuptools import find_packages, setup


if __name__ == "__main__":
    with open("README.md", "r") as f:
        long_description = f.read()
    fp = open("pipefuser/__version__.py", "r").read()
    version = eval(fp.strip().split()[-1])

    setup(
        name="pipefusion",
        author="Jiannan Wang, Jiarui Fang, Aoyu Li, Pengcheng Yang",
        author_email="muyangli@mit.edu",
        packages=find_packages(),
        install_requires=["torch", "diffusers", "transformers", "tqdm"],
        url="https://github.com/PipeFusion/PipeFusion.",
        description="DistriFusion: Distributed Parallel Inference for High-Resolution Diffusion Models",
        long_description=long_description,
        long_description_content_type="text/markdown",
        version=version,
        classifiers=[
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent",
        ],
        include_package_data=True,
        python_requires=">=3.10",
    )
