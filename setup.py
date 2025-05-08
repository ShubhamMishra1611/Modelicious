from setuptools import setup, find_packages

setup(
    name="llmserve",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "torch",
        "pyyaml",
    ],
    extras_require={
        "test": [
            "pytest==7.4.3",
            "pytest-asyncio==0.21.1",
            "httpx==0.25.1",
        ],
    },
) 