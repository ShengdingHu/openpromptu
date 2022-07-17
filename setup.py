
import setuptools
import os

def get_requirements(path):
    ret = []
    with open(os.path.join(path, "requirements.txt"), encoding="utf-8") as freq:
        for line in freq.readlines():
            ret.append( line.strip() )
    return ret


path = os.path.dirname(os.path.abspath(__file__))
print(path)
requires =  get_requirements(path)
print(requires)

# with open('README.md', 'r') as f:
setuptools.setup(
    name = 'openpromptu',
    version = '0.1.1',
    description = "The model-agnostic and core version of OpenPrompt, keeping only the hard prompt and verbalizer. Should be used together with OpenDelta",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author = '',
    author_email = 'shengdinghu@gmail.com',
    license="Apache",
    url="https://github.com/thunlp/OpenPromptU",
    keywords = ['PLM', 'Parameter-efficient-Learning', 'AI', 'NLP'],
    python_requires=">=3.6.0",
    install_requires=requires,
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ]
)