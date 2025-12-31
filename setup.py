# -*- coding: utf-8 -*-
# @File : setup.py

from setuptools import setup, find_packages

# Package requirements (edit as needed)
requirements = [
    "torch>=1.7",
    "torchvision",
    "mmdet3d==1.0.0rc4",
    "mmseg",
    "mmcv-full",
    "mmdet",
    "shapely",
    "pyquaternion",
    "nuscenes-dev-kit",
    "numpy",
    "opencv-python",
    "pandas",
    "scipy",
    "tqdm",
    "matplotlib",
    "seaborn",
]

# Authors (keep as lists for easy multi-author maintenance)
authors = [
    "yangzhao",
    "shiyinan",
]
author_emails = [
    "lingxi.yz950701@gmail.com",
    "syn1211@outlook.com"
]

setup(
    name="ssbev",
    version="1.0",
    author=", ".join(authors),
    author_email=", ".join(author_emails),
    description="Semi Supervised Learning about Bevdet, Bevdepth, Bevformer, etc.",
    # 你要安装的包，通过 setuptools.find_packages 找到当前目录下有哪些包
    packages=find_packages(),
    install_requires=requirements,
)
