from setuptools import setup

setup(
    name="fastserve",
    version="0.0.1",
    author="Yinmin Zhong, Shengyu Liu, Yuanhang Sun",
    description="High performance inference engine for LLMs.",
    packages=["fastserve", "fastserve.glm4_moe"],
    package_data={
        "fastserve.glm4_moe": [
            "csrc/*.cpp",
            "csrc/*.cu",
            "csrc/include/minisgl/*",
        ]
    },
    include_package_data=True,
    zip_safe=False,
)
